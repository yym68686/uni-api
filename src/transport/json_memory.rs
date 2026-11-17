//! Allocation-free structural pass before constructing serde_json's DOM.
//! A wire-size multiplier alone underestimates dense arrays and small objects.
use std::io::{self, BufRead};

pub(super) fn estimate(mut input: impl BufRead) -> io::Result<u64> {
    let mut nodes = 0_u64;
    let mut entries = 0_u64;
    let mut object_roots = 0_u64;
    let mut string_bytes = 0_u64;
    let mut in_string = false;
    let mut escaped = false;
    let mut scalar = false;
    let mut object_start = false;
    loop {
        let buffer = input.fill_buf()?;
        if buffer.is_empty() {
            break;
        }
        for &byte in buffer {
            if in_string {
                string_bytes = string_bytes.saturating_add(1);
                if escaped {
                    escaped = false;
                } else if byte == b'\\' {
                    escaped = true;
                } else if byte == b'"' {
                    in_string = false;
                }
                continue;
            }
            if object_start && !byte.is_ascii_whitespace() {
                if byte != b'}' {
                    object_roots = object_roots.saturating_add(1);
                }
                object_start = false;
            }
            match byte {
                b'"' => {
                    nodes = nodes.saturating_add(1);
                    in_string = true;
                    scalar = false;
                }
                b'[' | b'{' => {
                    nodes = nodes.saturating_add(1);
                    scalar = false;
                    object_start = byte == b'{';
                }
                b':' => {
                    entries = entries.saturating_add(1);
                    scalar = false;
                }
                b',' | b']' | b'}' | b' ' | b'\n' | b'\r' | b'\t' => scalar = false,
                _ if !scalar => {
                    nodes = nodes.saturating_add(1);
                    scalar = true;
                }
                _ => {}
            }
        }
        let consumed = buffer.len();
        input.consume(consumed);
    }
    // Arrays: up to twice the node count for geometric Vec growth, plus old
    // storage during reallocation. Objects: BTree nodes, keys and links even
    // at low occupancy, including a full root node for each nonempty object.
    // Strings: owned data, parser scratch and growth copies.
    Ok(nodes
        .saturating_mul((std::mem::size_of::<serde_json::Value>() as u64).saturating_mul(3))
        .saturating_add(entries.saturating_mul(256))
        .saturating_add(object_roots.saturating_mul(1024))
        .saturating_add(string_bytes.saturating_mul(4))
        .saturating_add(32 * 1024))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dense_array_budget_covers_backing_storage_and_reallocation() {
        let json = format!("[{}0]", "0,".repeat(1_048_575));
        let value: serde_json::Value = serde_json::from_str(&json).unwrap();
        let array = value.as_array().unwrap();
        let backing = array.capacity() * std::mem::size_of::<serde_json::Value>();
        assert!(estimate(json.as_bytes()).unwrap() >= (backing * 2) as u64);
    }

    #[test]
    fn scanner_handles_escapes_and_buffer_boundaries() {
        let json = br#"{"a\"\\:":[true,null,123,"x:y"]}"#;
        let expected = estimate(json.as_slice()).unwrap();
        assert!(serde_json::from_slice::<serde_json::Value>(json).is_ok());
        assert_eq!(
            estimate(io::BufReader::with_capacity(1, json.as_slice())).unwrap(),
            expected
        );
        assert!(expected > 32 * 1024);
    }

    #[test]
    fn skinny_nested_objects_include_each_sparse_btree_root() {
        let json = format!("{}0{}", "{\"key\":".repeat(100), "}".repeat(100));
        assert!(serde_json::from_str::<serde_json::Value>(&json).is_ok());
        assert!(estimate(json.as_bytes()).unwrap() >= 100 * 1024 + 32 * 1024);
        // Empty maps do not allocate BTree nodes.
        assert!(estimate(b"[{},{},{}]".as_slice()).unwrap() < 34 * 1024);
    }
}
