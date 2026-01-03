import os
import sys
import json
import base64
import asyncio

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.request import prepare_request_payload
from core.utils import gemini_audio_inline_data_to_wav_base64, generate_no_stream_response


async def test_request_mapping_to_gemini():
    provider = {
        "provider": "gemini",
        "base_url": "https://generativelanguage.googleapis.com/v1beta",
        "api": "test-key",
        "model": ["gemini-2.5-flash-preview-tts"],
    }

    request_data = {
        "model": "gemini-2.5-flash-preview-tts",
        "modalities": ["text", "audio"],
        "audio": {"voice": "Kore", "format": "wav"},
        "messages": [{"role": "user", "content": "Say cheerfully: Have a wonderful day!"}],
        "stream": True,  # should be forced to non-stream in main.py; payload mapping should still be correct
    }

    url, headers, payload, engine = await prepare_request_payload(provider, request_data)
    assert engine == "gemini"
    assert "/models/gemini-2.5-flash-preview-tts:" in url
    assert "modalities" not in payload
    assert "audio" not in payload
    assert "thinkingConfig" not in payload["generationConfig"]
    assert payload["generationConfig"]["responseModalities"] == ["AUDIO"]
    assert payload["generationConfig"]["speechConfig"]["voiceConfig"]["prebuiltVoiceConfig"]["voiceName"] == "Kore"
    assert payload["model"] == "gemini-2.5-flash-preview-tts"


async def test_response_mapping_to_openai_chat():
    here = os.path.dirname(os.path.abspath(__file__))
    sample_path = os.path.join(here, "json_str", "gemini", "audio_reponse.json")
    with open(sample_path, "r", encoding="utf-8") as f:
        gemini_resp = json.load(f)

    inline = gemini_resp["candidates"][0]["content"]["parts"][0]["inlineData"]
    wav_b64 = gemini_audio_inline_data_to_wav_base64(inline["mimeType"], inline["data"])
    assert wav_b64, "expected PCM->WAV conversion"
    assert base64.b64decode(wav_b64)[:4] == b"RIFF"

    timestamp = 0
    audio_obj = {
        "id": "audio_test",
        "data": wav_b64,
        "expires_at": None,
        "transcript": None,
        "format": "wav",
    }
    out = await generate_no_stream_response(timestamp, "gemini-2.5-flash-preview-tts", content=None, role="assistant", total_tokens=57, prompt_tokens=8, completion_tokens=49, audio=audio_obj)
    out_json = json.loads(out)
    assert out_json["choices"][0]["message"]["audio"]["format"] == "wav"
    assert out_json["choices"][0]["message"]["content"] is None


if __name__ == "__main__":
    asyncio.run(test_request_mapping_to_gemini())
    asyncio.run(test_response_mapping_to_openai_chat())
