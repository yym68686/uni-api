use memchr::memchr2;
use pyo3::exceptions::{PyIOError, PyOverflowError, PyTypeError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyBool, PyBytes, PyDict, PyFloat, PyInt, PyList, PyString, PyTuple};
use std::fs::OpenOptions;
use std::io::{BufWriter, Write};

const ERROR_NONE: u8 = 0;
const ERROR_MAX_DEPTH: u8 = 1;
const ERROR_MAX_SCALAR_BYTES: u8 = 2;
const ERROR_MAX_ESTIMATED_BYTES: u8 = 3;

const PHASE_NONE: u8 = 0;
const PHASE_CHUNK_RAW_CHARGE: u8 = 1;
const PHASE_STRUCTURAL_ITEM_SCAN: u8 = 2;
const PHASE_DEPTH_SCAN: u8 = 3;
const PHASE_SCALAR_SCAN: u8 = 4;

#[derive(Clone, Copy)]
struct Limits {
    raw_memory_multiplier: u64,
    token_memory_bytes: u64,
    max_depth: u64,
    max_scalar_bytes: u64,
    max_estimated_bytes: u64,
}

#[derive(Clone, Copy, Debug)]
struct State {
    raw_bytes: u64,
    tokens: u64,
    depth: u64,
    peak_depth: u64,
    in_string: bool,
    escaped: bool,
    scalar_active: bool,
    scalar_bytes: u64,
}

#[derive(Clone, Copy, Debug)]
struct ScanError {
    code: u8,
    phase: u8,
}

impl State {
    fn estimated_bytes(self, limits: Limits) -> Option<u128> {
        let raw = u128::from(self.raw_bytes) * u128::from(limits.raw_memory_multiplier);
        let structural = u128::from(self.tokens) * u128::from(limits.token_memory_bytes);
        raw.checked_add(structural)
    }

    fn exceeds_estimated_limit(self, limits: Limits) -> bool {
        self.estimated_bytes(limits)
            .map(|value| value > u128::from(limits.max_estimated_bytes))
            .unwrap_or(true)
    }

    fn finish_scalar(&mut self) {
        self.scalar_active = false;
        self.scalar_bytes = 0;
    }

    fn count_token(&mut self, limits: Limits) -> Result<(), ScanError> {
        self.tokens = self.tokens.saturating_add(1);
        if self.exceeds_estimated_limit(limits) {
            return Err(ScanError {
                code: ERROR_MAX_ESTIMATED_BYTES,
                phase: PHASE_STRUCTURAL_ITEM_SCAN,
            });
        }
        Ok(())
    }
}

fn is_outside_special(value: u8) -> bool {
    matches!(
        value,
        b'"' | b'{' | b'}' | b'[' | b']' | b' ' | b'\t' | b'\n' | b'\r' | b',' | b':'
    )
}

fn scan_json_chunk(
    bytes: &[u8],
    limits: Limits,
    mut state: State,
) -> Result<State, (State, ScanError)> {
    let chunk_len = match u64::try_from(bytes.len()) {
        Ok(value) => value,
        Err(_) => {
            return Err((
                state,
                ScanError {
                    code: ERROR_MAX_ESTIMATED_BYTES,
                    phase: PHASE_CHUNK_RAW_CHARGE,
                },
            ));
        }
    };
    state.raw_bytes = state.raw_bytes.saturating_add(chunk_len);
    if state.exceeds_estimated_limit(limits) {
        return Err((
            state,
            ScanError {
                code: ERROR_MAX_ESTIMATED_BYTES,
                phase: PHASE_CHUNK_RAW_CHARGE,
            },
        ));
    }

    let mut position = 0usize;
    while position < bytes.len() {
        if state.in_string {
            if state.escaped {
                state.escaped = false;
                position += 1;
                continue;
            }

            let Some(relative) = memchr2(b'"', b'\\', &bytes[position..]) else {
                break;
            };
            position += relative;
            let value = bytes[position];
            position += 1;
            if value == b'\\' {
                if position < bytes.len() {
                    position += 1;
                } else {
                    state.escaped = true;
                }
            } else {
                state.in_string = false;
            }
            continue;
        }

        let run_end = bytes[position..]
            .iter()
            .position(|value| is_outside_special(*value))
            .map(|relative| position + relative)
            .unwrap_or(bytes.len());
        if run_end > position {
            if !state.scalar_active {
                state.scalar_active = true;
                state.scalar_bytes = 0;
                if let Err(error) = state.count_token(limits) {
                    return Err((state, error));
                }
            }
            let run_length = u64::try_from(run_end - position).unwrap_or(u64::MAX);
            let scalar_bytes = state.scalar_bytes.saturating_add(run_length);
            if scalar_bytes > limits.max_scalar_bytes {
                state.scalar_bytes = limits.max_scalar_bytes.saturating_add(1);
                return Err((
                    state,
                    ScanError {
                        code: ERROR_MAX_SCALAR_BYTES,
                        phase: PHASE_SCALAR_SCAN,
                    },
                ));
            }
            state.scalar_bytes = scalar_bytes;
            position = run_end;
        }
        if position == bytes.len() {
            break;
        }

        let value = bytes[position];
        position += 1;
        if value == b'"' {
            state.finish_scalar();
            if let Err(error) = state.count_token(limits) {
                return Err((state, error));
            }
            state.in_string = true;
            continue;
        }

        if matches!(value, b'{' | b'[') {
            state.finish_scalar();
            if let Err(error) = state.count_token(limits) {
                return Err((state, error));
            }
            state.depth = state.depth.saturating_add(1);
            state.peak_depth = state.peak_depth.max(state.depth);
            if state.depth > limits.max_depth {
                return Err((
                    state,
                    ScanError {
                        code: ERROR_MAX_DEPTH,
                        phase: PHASE_DEPTH_SCAN,
                    },
                ));
            }
            continue;
        }

        if matches!(value, b'}' | b']') {
            state.finish_scalar();
            state.depth = state.depth.saturating_sub(1);
            continue;
        }

        state.finish_scalar();
    }

    Ok(state)
}

type NativeResult = (u8, u8, u64, u64, u64, u64, bool, bool, bool, u64, u128);

fn result_tuple(state: State, limits: Limits, error: Option<ScanError>) -> NativeResult {
    let error = error.unwrap_or(ScanError {
        code: ERROR_NONE,
        phase: PHASE_NONE,
    });
    (
        error.code,
        error.phase,
        state.raw_bytes,
        state.tokens,
        state.depth,
        state.peak_depth,
        state.in_string,
        state.escaped,
        state.scalar_active,
        state.scalar_bytes,
        state.estimated_bytes(limits).unwrap_or(u128::MAX),
    )
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn scan_json_memory_chunk(
    py: Python<'_>,
    chunk: &Bound<'_, PyBytes>,
    raw_memory_multiplier: u64,
    token_memory_bytes: u64,
    max_depth: u64,
    max_scalar_bytes: u64,
    max_estimated_bytes: u64,
    raw_bytes: u64,
    tokens: u64,
    depth: u64,
    peak_depth: u64,
    in_string: bool,
    escaped: bool,
    scalar_active: bool,
    scalar_bytes: u64,
) -> PyResult<NativeResult> {
    let bytes = chunk.as_bytes();
    let limits = Limits {
        raw_memory_multiplier,
        token_memory_bytes,
        max_depth,
        max_scalar_bytes,
        max_estimated_bytes,
    };
    let state = State {
        raw_bytes,
        tokens,
        depth,
        peak_depth,
        in_string,
        escaped,
        scalar_active,
        scalar_bytes,
    };

    if state.estimated_bytes(limits).is_none() {
        return Err(PyOverflowError::new_err(
            "JSON memory estimate exceeds the native integer envelope",
        ));
    }

    let outcome = py.detach(|| scan_json_chunk(bytes, limits, state));
    Ok(match outcome {
        Ok(state) => result_tuple(state, limits, None),
        Err((state, error)) => result_tuple(state, limits, Some(error)),
    })
}

fn write_all_json(writer: &mut impl Write, bytes: &[u8]) -> PyResult<()> {
    writer
        .write_all(bytes)
        .map_err(|error| PyIOError::new_err(error.to_string()))
}

fn write_json_scalar<T: serde::Serialize + ?Sized>(
    writer: &mut impl Write,
    value: &T,
) -> PyResult<()> {
    serde_json::to_writer(writer, value).map_err(|error| PyIOError::new_err(error.to_string()))
}

fn write_python_json_value(writer: &mut impl Write, value: &Bound<'_, PyAny>) -> PyResult<()> {
    if value.is_none() {
        return write_all_json(writer, b"null");
    }
    if value.is_instance_of::<PyBool>() {
        return if value.extract::<bool>()? {
            write_all_json(writer, b"true")
        } else {
            write_all_json(writer, b"false")
        };
    }
    if let Ok(text) = value.cast::<PyString>() {
        return write_json_scalar(writer, text.to_str()?);
    }
    if value.is_instance_of::<PyInt>() {
        let rendered = value.str()?;
        return write_all_json(writer, rendered.to_str()?.as_bytes());
    }
    if value.is_instance_of::<PyFloat>() {
        let number = value.extract::<f64>()?;
        if number.is_nan() {
            return write_all_json(writer, b"NaN");
        }
        if number == f64::INFINITY {
            return write_all_json(writer, b"Infinity");
        }
        if number == f64::NEG_INFINITY {
            return write_all_json(writer, b"-Infinity");
        }
        return write_json_scalar(writer, &number);
    }
    if let Ok(mapping) = value.cast::<PyDict>() {
        write_all_json(writer, b"{")?;
        let mut first = true;
        for (key, item) in mapping.iter() {
            let key = key
                .cast::<PyString>()
                .map_err(|_| PyTypeError::new_err("native JSON object keys must be strings"))?;
            if !first {
                write_all_json(writer, b",")?;
            }
            first = false;
            write_json_scalar(writer, key.to_str()?)?;
            write_all_json(writer, b":")?;
            write_python_json_value(writer, &item)?;
        }
        return write_all_json(writer, b"}");
    }
    if let Ok(sequence) = value.cast::<PyList>() {
        write_all_json(writer, b"[")?;
        let mut first = true;
        for item in sequence.iter() {
            if !first {
                write_all_json(writer, b",")?;
            }
            first = false;
            write_python_json_value(writer, &item)?;
        }
        return write_all_json(writer, b"]");
    }
    if let Ok(sequence) = value.cast::<PyTuple>() {
        write_all_json(writer, b"[")?;
        let mut first = true;
        for item in sequence.iter() {
            if !first {
                write_all_json(writer, b",")?;
            }
            first = false;
            write_python_json_value(writer, &item)?;
        }
        return write_all_json(writer, b"]");
    }
    Err(PyTypeError::new_err(format!(
        "native JSON serializer does not support {}",
        value.get_type().name()?
    )))
}

/// Serialize a Python JSON graph directly into a bounded Rust file buffer.
///
/// Unlike ``json.dumps(...).encode()``, this never materializes a whole-document
/// string or bytes object.  It is intentionally limited to the JSON-native
/// values produced by FastAPI's parser and provider overrides.
#[pyfunction]
fn write_json_file(payload: &Bound<'_, PyAny>, path: &str) -> PyResult<u64> {
    let file = OpenOptions::new()
        .write(true)
        .truncate(true)
        .open(path)
        .map_err(|error| PyIOError::new_err(error.to_string()))?;
    let mut writer = BufWriter::with_capacity(64 * 1024, file);
    write_python_json_value(&mut writer, payload)?;
    writer
        .flush()
        .map_err(|error| PyIOError::new_err(error.to_string()))?;
    writer
        .get_ref()
        .metadata()
        .map(|metadata| metadata.len())
        .map_err(|error| PyIOError::new_err(error.to_string()))
}

#[pymodule]
fn _uni_api_native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add("IMPLEMENTATION_VERSION", 2u8)?;
    module.add_function(wrap_pyfunction!(scan_json_memory_chunk, module)?)?;
    module.add_function(wrap_pyfunction!(write_json_file, module)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn defaults() -> Limits {
        Limits {
            raw_memory_multiplier: 5,
            token_memory_bytes: 1024,
            max_depth: 128,
            max_scalar_bytes: 4096,
            max_estimated_bytes: 256 * 1024 * 1024,
        }
    }

    fn initial() -> State {
        State {
            raw_bytes: 0,
            tokens: 0,
            depth: 0,
            peak_depth: 0,
            in_string: false,
            escaped: false,
            scalar_active: false,
            scalar_bytes: 0,
        }
    }

    #[test]
    fn counts_dense_objects() {
        let mut payload = Vec::from(&b"["[..]);
        for index in 0..10_000 {
            if index != 0 {
                payload.push(b',');
            }
            payload.extend_from_slice(b"{}");
        }
        payload.push(b']');
        let state = scan_json_chunk(&payload, defaults(), initial()).unwrap();
        assert_eq!(state.tokens, 10_001);
    }

    #[test]
    fn preserves_escape_state_across_chunks() {
        let first = scan_json_chunk(br#"{"value":"tail\"#, defaults(), initial()).unwrap();
        assert!(first.in_string);
        assert!(first.escaped);
        let second = scan_json_chunk(br#""quoted"}"#, defaults(), first).unwrap();
        assert!(!second.in_string);
        assert_eq!(second.depth, 0);
    }
}
