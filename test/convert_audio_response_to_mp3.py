#!/usr/bin/env python3
import argparse
import base64
import json
import os
import re
import subprocess
import sys
import tempfile
import wave
import io


def _b64decode_relaxed(data: str) -> bytes:
    s = (data or "").strip()
    s = re.sub(r"\s+", "", s)
    # If the base64 was truncated, len%4 can be 1 (invalid). Trim until decodable.
    while len(s) % 4 == 1:
        s = s[:-1]
    s += "=" * ((4 - (len(s) % 4)) % 4)
    return base64.b64decode(s)


def _parse_rate(mime_type: str) -> int | None:
    if not mime_type:
        return None
    m = re.search(r"rate=(\d+)", mime_type)
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def _pcm_l16_to_wav(pcm_bytes: bytes, sample_rate: int, channels: int = 1) -> bytes:
    buf = tempfile.SpooledTemporaryFile(max_size=10 * 1024 * 1024)
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)  # L16 => 16-bit = 2 bytes
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    buf.seek(0)
    return buf.read()


def _guess_container(audio_bytes: bytes) -> str:
    if audio_bytes.startswith(b"RIFF") and b"WAVE" in audio_bytes[:16]:
        return "wav"
    if audio_bytes.startswith(b"ID3") or audio_bytes[:2] == b"\xff\xfb":
        return "mp3"
    return "unknown"


def _extract_openai_audio(json_obj: dict) -> tuple[bytes, str | None]:
    audio = (
        json_obj.get("choices", [{}])[0]
        .get("message", {})
        .get("audio", {})
    )
    b64 = audio.get("data")
    if not b64:
        raise ValueError("No `choices[0].message.audio.data` found in JSON.")
    audio_bytes = _b64decode_relaxed(b64)
    return audio_bytes, audio.get("format")


def _extract_gemini_inline_audio(json_obj: dict) -> bytes:
    parts = (
        json_obj.get("candidates", [{}])[0]
        .get("content", {})
        .get("parts", [])
    )
    for part in parts:
        inline = part.get("inlineData") or {}
        mime = inline.get("mimeType") or ""
        b64 = inline.get("data")
        if b64 and mime.lower().startswith("audio/"):
            pcm = _b64decode_relaxed(b64)
            if "l16" in mime.lower() and "pcm" in mime.lower():
                rate = _parse_rate(mime) or 24000
                return _pcm_l16_to_wav(pcm, sample_rate=rate, channels=1)
            raise ValueError(f"Unsupported Gemini audio mimeType: {mime!r} (only PCM/L16 supported)")
    raise ValueError("No Gemini `inlineData` audio part found in JSON.")


def _ffmpeg_wav_to_mp3(wav_path: str, mp3_path: str, bitrate: str | None, qscale: int | None) -> None:
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", wav_path]
    if bitrate:
        cmd += ["-b:a", bitrate]
    if qscale is not None:
        cmd += ["-q:a", str(qscale)]
    cmd += [mp3_path]
    subprocess.run(cmd, check=True)

def _validate_wav_bytes(wav_bytes: bytes) -> None:
    try:
        with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
            wf.getparams()
    except Exception as e:
        raise ValueError(
            "Decoded WAV bytes are invalid (often caused by truncated base64 in the JSON). "
            "Make sure you're using the full `audio.data` field from the real response."
        ) from e


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert chat/completions audio response JSON into an MP3 file.",
    )
    parser.add_argument("input", help="Path to JSON response file")
    parser.add_argument("-o", "--output", help="Path to output .mp3 (default: <input>.mp3)")
    parser.add_argument(
        "--source",
        choices=["auto", "openai", "gemini"],
        default="auto",
        help="Which JSON schema to parse (default: auto)",
    )
    parser.add_argument("--bitrate", default=None, help="MP3 bitrate for ffmpeg, e.g. 128k, 192k")
    parser.add_argument("--qscale", type=int, default=2, help="ffmpeg VBR quality (lower is better), default: 2")
    parser.add_argument("--write-wav", default=None, help="Optional path to also write the decoded WAV")
    args = parser.parse_args()

    in_path = args.input
    out_path = args.output or (os.path.splitext(in_path)[0] + ".mp3")

    with open(in_path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    audio_bytes = None
    declared_format = None
    if args.source in ["auto", "openai"] and isinstance(obj, dict) and "choices" in obj:
        audio_bytes, declared_format = _extract_openai_audio(obj)
    elif args.source in ["auto", "gemini"] and isinstance(obj, dict) and "candidates" in obj:
        audio_bytes = _extract_gemini_inline_audio(obj)
        declared_format = "wav"
    else:
        if args.source == "openai":
            audio_bytes, declared_format = _extract_openai_audio(obj)
        elif args.source == "gemini":
            audio_bytes = _extract_gemini_inline_audio(obj)
            declared_format = "wav"
        else:
            raise ValueError("Unable to auto-detect JSON schema; try `--source openai` or `--source gemini`.")

    container = declared_format or _guess_container(audio_bytes)
    if container == "mp3":
        with open(out_path, "wb") as f:
            f.write(audio_bytes)
        return 0

    if container != "wav":
        raise ValueError(f"Unsupported audio container: {container!r}. Expected WAV or MP3 bytes.")

    _validate_wav_bytes(audio_bytes)

    if args.write_wav:
        with open(args.write_wav, "wb") as f:
            f.write(audio_bytes)

    with tempfile.TemporaryDirectory() as td:
        wav_path = os.path.join(td, "in.wav")
        with open(wav_path, "wb") as f:
            f.write(audio_bytes)
        _ffmpeg_wav_to_mp3(wav_path, out_path, bitrate=args.bitrate, qscale=args.qscale)

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        raise SystemExit(1)
