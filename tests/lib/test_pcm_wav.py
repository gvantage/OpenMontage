"""WAV wrapping for OpenRouter PCM. No network."""

from __future__ import annotations

import io
import wave

import pytest

from lib.pcm_wav import (
    GEMINI_TTS_PCM_RATE,
    PcmWavError,
    pcm_s16le_to_wav,
    sample_rate_from_content_type,
    wav_info,
)


def test_default_contract_is_24k_mono_s16le():
    pcm = b"\x00\x01" * 48
    wav = pcm_s16le_to_wav(pcm)
    info = wav_info(wav)
    assert info["sample_rate"] == GEMINI_TTS_PCM_RATE == 24000
    assert info["channels"] == 1
    assert info["sample_width"] == 2
    assert info["frames"] == 48
    with wave.open(io.BytesIO(wav), "rb") as handle:
        assert handle.readframes(48) == pcm


def test_odd_pcm_is_rejected():
    with pytest.raises(PcmWavError):
        pcm_s16le_to_wav(b"\x00")


def test_content_type_rate_overrides_the_contract_default():
    rate, source = sample_rate_from_content_type("audio/L16; rate=16000")
    assert rate == 16000
    assert source == "content-type"
    rate, source = sample_rate_from_content_type("audio/pcm")
    assert rate == 24000
    assert source == "gemini-3.8-contract"
