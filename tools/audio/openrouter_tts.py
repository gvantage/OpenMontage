"""OpenRouter speech for Gemini 3.8 Flash TTS, default voice Despina.

POST https://openrouter.ai/api/v1/audio/speech

The adapter always sends ``response_format=pcm`` and writes a WAV file.
Raw PCM is wrapped by ``lib.pcm_wav``. A body that is already mono 16-bit
WAV is stored after the same check. Compressed audio is rejected.

This module does not call DeepSeek and does not route DeepSeek through
OpenRouter. ``OPENROUTER_API_KEY`` is read from the environment only.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import requests

from lib.pcm_wav import (
    GEMINI_TTS_PCM_CHANNELS,
    GEMINI_TTS_PCM_RATE,
    PcmWavError,
    pcm_s16le_to_wav,
    sample_rate_from_content_type,
    wav_info,
)
from tools.base_tool import (
    BaseTool,
    Determinism,
    ResourceProfile,
    RetryPolicy,
    ToolResult,
    ToolRuntime,
    ToolStability,
    ToolStatus,
    ToolTier,
)
from tools.provider_pricing import PriceQuoteRequired


SPEECH_URL = "https://openrouter.ai/api/v1/audio/speech"
MODEL_ID = "google/gemini-3.8-flash-tts"
DEFAULT_VOICE = "Despina"
VOICES = (
    "Zephyr",
    "Puck",
    "Charon",
    "Kore",
    "Fenrir",
    "Leda",
    "Orus",
    "Aoede",
    "Callirrhoe",
    "Autonoe",
    "Enceladus",
    "Iapetus",
    "Umbriel",
    "Algieba",
    "Despina",
    "Erinome",
    "Algenib",
    "Rasalgethi",
    "Laomedeia",
    "Achernar",
    "Alnilam",
    "Schedar",
    "Gacrux",
    "Pulcherrima",
    "Achird",
    "Zubenelgenubi",
    "Vindemiatrix",
    "Sadachbia",
    "Sadaltager",
    "Sulafat",
)


def _redact(text: str, secret: str) -> str:
    if secret and text:
        return text.replace(secret, "[REDACTED]")
    return text


class OpenRouterTTS(BaseTool):
    name = "openrouter_tts"
    version = "0.1.0"
    tier = ToolTier.VOICE
    capability = "tts"
    provider = "openrouter"
    stability = ToolStability.EXPERIMENTAL
    determinism = Determinism.STOCHASTIC
    runtime = ToolRuntime.API
    dependencies = ["env:OPENROUTER_API_KEY"]
    install_instructions = (
        "Export OPENROUTER_API_KEY. Direct dubbing uses that key. "
        "Set OPENMONTAGE_OPENROUTER_TTS=1 as well before tts_selector may "
        f"rank this tool. Model {MODEL_ID}, news voice {DEFAULT_VOICE}. "
        "Docs: https://openrouter.ai/docs/api/api-reference/tts/create-speech"
    )
    agent_skills = ["deepseek-director"]
    capabilities = ["text_to_speech", "voice_selection"]
    supports = {
        "voice_cloning": False,
        "multilingual": True,
        "offline": False,
        "native_audio": True,
    }
    best_for = [
        "OpenRouter Gemini 3.8 Flash TTS dubbing with the Despina studio voice",
    ]
    not_good_for = [
        "offline production",
        "replacing a narration provider the user already approved",
    ]
    input_schema = {
        "type": "object",
        "required": ["text"],
        "properties": {
            "text": {"type": "string", "minLength": 1},
            "voice": {
                "type": "string",
                "enum": list(VOICES),
                "default": DEFAULT_VOICE,
            },
            "voice_id": {
                "type": "string",
                "description": "Alias of voice. Must match voice when both are set.",
            },
            "model": {"type": "string", "enum": [MODEL_ID], "default": MODEL_ID},
            "model_id": {"type": "string", "enum": [MODEL_ID]},
            "instructions": {
                "type": "string",
                "description": "Delivery note supported by Gemini TTS on OpenRouter.",
            },
            "output_path": {"type": "string"},
        },
    }
    resource_profile = ResourceProfile(
        cpu_cores=1, ram_mb=256, vram_mb=0, disk_mb=20, network_required=True
    )
    retry_policy = RetryPolicy(max_retries=0)
    idempotency_key_fields = ["text", "voice", "model", "instructions"]
    side_effects = [
        "calls OpenRouter /api/v1/audio/speech",
        "writes a WAV file",
    ]
    user_visible_verification = [
        "Listen to the WAV and confirm it matches the section timing",
    ]

    def get_status(self) -> ToolStatus:
        # The key alone must not change tts_selector ranking. The director
        # dub command calls execute() directly. Selector opt-in is separate.
        if not os.environ.get("OPENROUTER_API_KEY"):
            return ToolStatus.UNAVAILABLE
        flag = os.environ.get("OPENMONTAGE_OPENROUTER_TTS", "").strip().lower()
        if flag not in {"1", "true", "yes"}:
            return ToolStatus.UNAVAILABLE
        return ToolStatus.AVAILABLE

    def estimate_cost(self, inputs: dict[str, Any]) -> float:
        raise PriceQuoteRequired(
            "OpenRouter Gemini TTS bills tokens. Obtain a current quote; "
            "this adapter does not invent a per-character price."
        )

    def execute(self, inputs: dict[str, Any]) -> ToolResult:
        key = os.environ.get("OPENROUTER_API_KEY", "")
        if not key:
            return ToolResult(
                success=False,
                error="OPENROUTER_API_KEY is not set. " + self.install_instructions,
            )
        try:
            prepared = self._prepare(inputs)
            content, content_type, generation_id = self._request(prepared, key)
            wav, meta = self._to_wav(content, content_type)
        except (ValueError, PcmWavError, requests.RequestException) as exc:
            return ToolResult(
                success=False,
                error="openrouter_tts: " + _redact(str(exc), key),
            )
        path = Path(prepared["output_path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".part")
        temporary.write_bytes(wav)
        temporary.replace(path)
        data = {
            "output": str(path),
            "provider": self.provider,
            "model": MODEL_ID,
            "voice": prepared["voice"],
            "response_format": "pcm",
            "format": "wav",
            "sample_rate": meta["sample_rate"],
            "channels": meta["channels"],
            "sample_width": meta["sample_width"],
            "sample_rate_source": meta["sample_rate_source"],
            "pcm_container": meta["pcm_container"],
            "cost_status": "unquoted",
        }
        if generation_id:
            data["generation_id"] = generation_id
        return ToolResult(
            success=True,
            data=data,
            artifacts=[str(path)],
            model=MODEL_ID,
            cost_usd=None,
        )

    def _prepare(self, inputs: dict[str, Any]) -> dict[str, Any]:
        text = inputs.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text is required")
        if len(text) > 4000:
            raise ValueError("text exceeds 4000 characters")
        voice = inputs.get("voice") or inputs.get("voice_id") or DEFAULT_VOICE
        if inputs.get("voice") and inputs.get("voice_id") and inputs["voice"] != inputs["voice_id"]:
            raise ValueError("voice and voice_id disagree")
        if voice not in VOICES:
            raise ValueError(f"voice {voice!r} is not a documented Gemini TTS voice")
        model = inputs.get("model") or inputs.get("model_id") or MODEL_ID
        if inputs.get("model") and inputs.get("model_id") and inputs["model"] != inputs["model_id"]:
            raise ValueError("model and model_id disagree")
        if model != MODEL_ID:
            raise ValueError(f"model must be {MODEL_ID}")
        instructions = inputs.get("instructions")
        if instructions is not None:
            if not isinstance(instructions, str):
                raise ValueError("instructions must be a string")
            if len(instructions) > 500:
                raise ValueError("instructions exceeds 500 characters")
        output_path = inputs.get("output_path") or "openrouter_despina.wav"
        if Path(output_path).suffix.lower() != ".wav":
            raise ValueError("output_path must end in .wav")
        return {
            "text": text,
            "voice": voice,
            "instructions": instructions or "",
            "output_path": output_path,
        }

    def _request(self, prepared: dict[str, Any], key: str):
        payload: dict[str, Any] = {
            "model": MODEL_ID,
            "input": prepared["text"],
            "voice": prepared["voice"],
            "response_format": "pcm",
        }
        if prepared["instructions"]:
            payload["instructions"] = prepared["instructions"]
        response = requests.post(
            SPEECH_URL,
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=180,
        )
        if response.status_code >= 400:
            raise ValueError(self._http_error(response, key))
        content = response.content or b""
        if not content:
            raise ValueError("OpenRouter returned an empty audio body")
        content_type = response.headers.get("Content-Type", "")
        generation_id = response.headers.get("X-Generation-Id", "")
        return content, content_type, generation_id

    def _http_error(self, response, key: str) -> str:
        detail = ""
        content_type = (response.headers.get("Content-Type") or "").lower()
        if "json" in content_type:
            try:
                payload = response.json()
            except Exception:
                payload = None
            if isinstance(payload, dict):
                error = payload.get("error")
                if isinstance(error, dict):
                    detail = str(error.get("message") or "")
                elif payload.get("message"):
                    detail = str(payload["message"])
        if not detail:
            detail = "speech request failed"
        return _redact(f"HTTP {response.status_code}: {detail[:400]}", key)

    def _to_wav(self, content: bytes, content_type: str) -> tuple[bytes, dict[str, Any]]:
        lowered = (content_type or "").lower()
        if "mpeg" in lowered or "mp3" in lowered or content[:3] == b"ID3":
            raise ValueError(
                "OpenRouter returned compressed audio. This adapter requested PCM "
                "and will not wrap MP3 as WAV."
            )
        if content[:4] == b"RIFF":
            info = wav_info(content)
            return content, {
                "sample_rate": info["sample_rate"],
                "channels": info["channels"],
                "sample_width": info["sample_width"],
                "sample_rate_source": "wav-header",
                "pcm_container": "wav",
            }
        if lowered and "pcm" not in lowered and "l16" not in lowered and "wav" not in lowered:
            raise ValueError(f"unexpected audio Content-Type {content_type!r}")
        rate, source = sample_rate_from_content_type(content_type)
        wav = pcm_s16le_to_wav(content, rate, GEMINI_TTS_PCM_CHANNELS)
        info = wav_info(wav)
        if info["sample_rate"] != rate:
            raise ValueError("WAV header sample rate does not match the PCM rate")
        return wav, {
            "sample_rate": rate,
            "channels": GEMINI_TTS_PCM_CHANNELS,
            "sample_width": info["sample_width"],
            "sample_rate_source": source,
            "pcm_container": "wrapped",
            "contract_rate": GEMINI_TTS_PCM_RATE,
        }
