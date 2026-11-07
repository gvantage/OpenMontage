"""Wrap headerless 16-bit PCM as a WAV container.

OpenRouter's speech endpoint returns a raw byte stream. For
``google/gemini-3.8-flash-tts`` this workflow requests ``response_format=pcm``.

Checked against public docs on 2026-11-07, without a live API call:

- https://openrouter.ai/docs/api/api-reference/tts/create-speech
  says ``audio/pcm`` is 16-bit little-endian raw audio. The reference does not
  publish a separate sample-rate field.
- https://openrouter.ai/google/gemini-3.8-flash-tts/llms.txt
  lists Despina and ``response_format`` ``pcm`` (default) or ``mp3``. It does
  not restate the sample rate.
- https://openrouter.ai/docs/guides/overview/multimodal/tts
  says Gemini TTS on this endpoint returns PCM.
- https://ai.google.dev/gemini-api/docs/speech-generation
  documents Gemini 3.8 TTS linear PCM as 24 kHz, mono, 16-bit signed
  little-endian. OpenRouter model-family copy describes the same 24 kHz /
  16-bit mono PCM layout.

The default container is therefore 24000 Hz, 1 channel, 16-bit. An explicit
``rate`` on the response ``Content-Type`` overrides that default so the WAV
header matches the bytes that were actually returned.
"""

from __future__ import annotations

import io
import re
import wave


GEMINI_TTS_PCM_RATE = 24000
GEMINI_TTS_PCM_CHANNELS = 1
GEMINI_TTS_PCM_WIDTH = 2


class PcmWavError(ValueError):
    """Raised when bytes are not a whole number of 16-bit PCM frames."""


def sample_rate_from_content_type(content_type: str | None) -> tuple[int, str]:
    """Return ``(rate, source)`` for a speech response.

    ``source`` is ``content-type`` when the header carries a rate, otherwise
    ``gemini-3.8-contract``.
    """

    if content_type:
        match = re.search(
            r"(?:^|[;\s])(?:rate|samplerate|sample_rate)=(\d+)",
            content_type.lower(),
        )
        if match:
            rate = int(match.group(1))
            if rate < 8000 or rate > 96000:
                raise PcmWavError(f"PCM sample rate {rate} is outside 8000..96000")
            return rate, "content-type"
    return GEMINI_TTS_PCM_RATE, "gemini-3.8-contract"


def pcm_s16le_to_wav(
    pcm: bytes,
    sample_rate: int = GEMINI_TTS_PCM_RATE,
    channels: int = GEMINI_TTS_PCM_CHANNELS,
) -> bytes:
    """Pack little-endian signed 16-bit PCM into a WAV byte string."""

    if channels < 1:
        raise PcmWavError("PCM channel count must be positive")
    if sample_rate < 1:
        raise PcmWavError("PCM sample rate must be positive")
    frame_bytes = GEMINI_TTS_PCM_WIDTH * channels
    if not pcm or len(pcm) % frame_bytes != 0:
        raise PcmWavError(
            "PCM length is not a whole number of 16-bit frames "
            f"({len(pcm)} bytes, frame {frame_bytes})"
        )
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(GEMINI_TTS_PCM_WIDTH)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm)
    return buffer.getvalue()


def wav_info(data: bytes) -> dict[str, int]:
    """Read a WAV container and require mono 16-bit PCM."""

    if len(data) < 12 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise PcmWavError("audio is not a RIFF WAVE container")
    with wave.open(io.BytesIO(data), "rb") as handle:
        channels = handle.getnchannels()
        width = handle.getsampwidth()
        rate = handle.getframerate()
        frames = handle.getnframes()
    if channels != GEMINI_TTS_PCM_CHANNELS or width != GEMINI_TTS_PCM_WIDTH:
        raise PcmWavError(
            f"WAV must be mono 16-bit PCM, got channels={channels} width={width}"
        )
    if rate < 1 or frames < 1:
        raise PcmWavError("WAV has no PCM frames")
    return {
        "sample_rate": rate,
        "channels": channels,
        "sample_width": width,
        "frames": frames,
    }
