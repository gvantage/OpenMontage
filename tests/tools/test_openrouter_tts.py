"""OpenRouter Despina speech adapter. requests.post is mocked."""

from __future__ import annotations

import json
import wave
from pathlib import Path

import pytest
import requests

from lib.pcm_wav import pcm_s16le_to_wav
from tools.audio.openrouter_tts import MODEL_ID, SPEECH_URL, OpenRouterTTS
from tools.base_tool import ToolStatus
from tools.provider_pricing import PriceQuoteRequired


class _Response:
    def __init__(self, content=b"", status=200, headers=None, payload=None):
        self.content = content
        self.status_code = status
        self.headers = headers or {}
        self._payload = payload
        self.text = ""

    def json(self):
        if self._payload is None:
            raise ValueError("not json")
        return self._payload


@pytest.fixture
def api_key(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-openrouter-secret")
    return "sk-openrouter-secret"


def test_unavailable_without_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENMONTAGE_OPENROUTER_TTS", raising=False)
    assert OpenRouterTTS().get_status() == ToolStatus.UNAVAILABLE


def test_key_alone_does_not_join_selector_ranking(api_key, monkeypatch):
    monkeypatch.delenv("OPENMONTAGE_OPENROUTER_TTS", raising=False)
    assert OpenRouterTTS().get_status() == ToolStatus.UNAVAILABLE


def test_selector_opt_in(api_key, monkeypatch):
    monkeypatch.setenv("OPENMONTAGE_OPENROUTER_TTS", "1")
    assert OpenRouterTTS().get_status() == ToolStatus.AVAILABLE


def test_cost_is_unquoted():
    with pytest.raises(PriceQuoteRequired):
        OpenRouterTTS().estimate_cost({"text": "你好"})


def test_posts_pcm_contract_and_wraps_wav(api_key, tmp_path, monkeypatch):
    pcm = b"\x10\x00" * 240
    seen = {}

    def post(url, headers=None, json=None, timeout=None):
        seen["url"] = url
        seen["headers"] = headers
        seen["json"] = json
        seen["timeout"] = timeout
        return _Response(
            content=pcm,
            headers={"Content-Type": "audio/pcm", "X-Generation-Id": "gen-1"},
        )

    monkeypatch.setattr("tools.audio.openrouter_tts.requests.post", post)
    output = tmp_path / "line.wav"
    result = OpenRouterTTS().execute(
        {"text": "你好", "output_path": str(output)}
    )

    assert result.success is True
    assert seen["url"] == SPEECH_URL
    assert seen["json"] == {
        "model": MODEL_ID,
        "input": "你好",
        "voice": "Despina",
        "response_format": "pcm",
    }
    assert seen["headers"]["Authorization"] == "Bearer sk-openrouter-secret"
    assert "api.deepseek.com" not in seen["url"]
    assert output.is_file()
    with wave.open(str(output), "rb") as handle:
        assert handle.getframerate() == 24000
        assert handle.getnchannels() == 1
        assert handle.getsampwidth() == 2
        assert handle.readframes(240) == pcm
    assert result.data["sample_rate_source"] == "gemini-3.8-contract"
    assert result.data["pcm_container"] == "wrapped"
    assert result.data["generation_id"] == "gen-1"
    assert result.cost_usd is None


def test_content_type_rate_is_honored(api_key, tmp_path, monkeypatch):
    pcm = b"\x00\x00" * 16

    def post(url, headers=None, json=None, timeout=None):
        return _Response(content=pcm, headers={"Content-Type": "audio/L16;rate=16000"})

    monkeypatch.setattr("tools.audio.openrouter_tts.requests.post", post)
    output = tmp_path / "rate.wav"
    result = OpenRouterTTS().execute({"text": "a", "output_path": str(output)})
    assert result.success is True
    assert result.data["sample_rate"] == 16000
    assert result.data["sample_rate_source"] == "content-type"


def test_existing_wav_is_not_double_wrapped(api_key, tmp_path, monkeypatch):
    wav = pcm_s16le_to_wav(b"\x02\x00" * 10, 24000, 1)

    def post(url, headers=None, json=None, timeout=None):
        return _Response(content=wav, headers={"Content-Type": "audio/wav"})

    monkeypatch.setattr("tools.audio.openrouter_tts.requests.post", post)
    output = tmp_path / "already.wav"
    result = OpenRouterTTS().execute({"text": "a", "output_path": str(output)})
    assert result.success is True
    assert Path(output).read_bytes() == wav
    assert result.data["pcm_container"] == "wav"


def test_mp3_is_rejected(api_key, tmp_path, monkeypatch):
    def post(url, headers=None, json=None, timeout=None):
        return _Response(content=b"ID3fake", headers={"Content-Type": "audio/mpeg"})

    monkeypatch.setattr("tools.audio.openrouter_tts.requests.post", post)
    result = OpenRouterTTS().execute(
        {"text": "a", "output_path": str(tmp_path / "no.wav")}
    )
    assert result.success is False
    assert "PCM" in result.error


def test_error_redacts_key(api_key, monkeypatch):
    def post(url, headers=None, json=None, timeout=None):
        raise requests.RequestException("connect failed sk-openrouter-secret")

    monkeypatch.setattr("tools.audio.openrouter_tts.requests.post", post)
    result = OpenRouterTTS().execute({"text": "a", "output_path": "out.wav"})
    assert result.success is False
    assert "sk-openrouter-secret" not in result.error
    assert "[REDACTED]" in result.error


def test_unknown_model_rejected(api_key):
    result = OpenRouterTTS().execute(
        {"text": "a", "model": "deepseek-flash", "output_path": "out.wav"}
    )
    assert result.success is False
    assert MODEL_ID in result.error


def test_http_error_uses_json_message(api_key, monkeypatch):
    def post(url, headers=None, json=None, timeout=None):
        return _Response(
            status=401,
            headers={"Content-Type": "application/json"},
            payload={"error": {"code": 401, "message": "bad key sk-openrouter-secret"}},
        )

    monkeypatch.setattr("tools.audio.openrouter_tts.requests.post", post)
    result = OpenRouterTTS().execute({"text": "a", "output_path": "out.wav"})
    assert result.success is False
    assert "sk-openrouter-secret" not in result.error
    body = json.dumps(result.error)
    assert "bad key" in body
