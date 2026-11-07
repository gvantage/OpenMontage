"""DeepSeek director and renderable news spec. No live API calls."""

from __future__ import annotations

import json
import shutil
import wave
from pathlib import Path

import pytest
import requests

from lib.deepseek_client import (
    CHAT_URL,
    DeepSeekDirectorError,
    complete_json,
    resolve_chat_url,
    resolve_model,
)
from lib.deepseek_director import build_messages, dub_project, plan_project
from lib.news_production import (
    NEWS_MAX_SECONDS,
    NEWS_MIN_SECONDS,
    ProductionSpecError,
    compile_project,
    import_narration_wavs,
    load_evidence_brief,
    parse_director_document,
    render_project,
    smoke_project,
)
from lib.pcm_wav import pcm_s16le_to_wav
from schemas.artifacts import validate_artifact


ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = ROOT / "examples" / "ai-news-evidence.template.json"


class _Response:
    def __init__(self, payload=None, status=200):
        self.status_code = status
        self._payload = payload or {}
        self.text = ""

    def json(self):
        return self._payload


def _short_brief() -> dict:
    return {
        "title": "短规格",
        "hook": "用于测试时间轴。",
        "tone": "清楚",
        "style": "clean-professional",
        "target_platform": "generic",
        "target_duration_seconds": 2,
        "language": "zh",
        "evidence": [
            {
                "id": "e1",
                "claim": "事实一",
                "narration": "第一句旁白保持不变。",
                "source": "来源一",
            },
            {
                "id": "e2",
                "claim": "事实二",
                "narration": "第二句旁白保持不变。",
                "source": "来源二",
            },
        ],
    }


def test_news_template_compiles_two_projects(tmp_path):
    brief = load_evidence_brief(TEMPLATE)
    assert NEWS_MIN_SECONDS <= brief["target_duration_seconds"] <= NEWS_MAX_SECONDS
    first = compile_project(brief, "ai-news-a", tmp_path)
    second = compile_project(brief, "ai-news-b", tmp_path)
    assert first["duration_seconds"] == 150
    assert second["project_id"] == "ai-news-b"
    for project_id in ("ai-news-a", "ai-news-b"):
        project = tmp_path / project_id
        script = json.loads((project / "artifacts" / "script.json").read_text(encoding="utf-8"))
        edit = json.loads((project / "artifacts" / "edit_decisions.json").read_text(encoding="utf-8"))
        validate_artifact("script", script)
        validate_artifact("scene_plan", json.loads((project / "artifacts" / "scene_plan.json").read_text(encoding="utf-8")))
        validate_artifact("asset_manifest", json.loads((project / "artifacts" / "asset_manifest.json").read_text(encoding="utf-8")))
        validate_artifact("edit_decisions", edit)
        validate_artifact("brief", json.loads((project / "artifacts" / "brief.json").read_text(encoding="utf-8")))
        assert script["total_duration_seconds"] == 150
        assert [section["text"] for section in script["sections"]] == [
            item["narration"] for item in brief["evidence"]
        ]
        assert edit["render_runtime"] == "ffmpeg"
        assert edit["audio"]["narration"]["segments"][0]["start_seconds"] == 0
        assert edit["audio"]["narration"]["segments"][-1]["end_seconds"] == 150
        checkpoint = json.loads((project / "checkpoint_idea.json").read_text(encoding="utf-8"))
        assert checkpoint["status"] == "awaiting_human"
        assert checkpoint["human_approved"] is False


def test_duration_outside_news_window_is_rejected(tmp_path):
    brief = _short_brief()
    brief["target_duration_seconds"] = 90
    with pytest.raises(ProductionSpecError):
        compile_project(brief, "too-short", tmp_path)


def test_shell_operation_is_rejected():
    with pytest.raises(ProductionSpecError):
        parse_director_document(
            {
                "summary": "no",
                "operations": [{"op": "shell", "command": "rm -rf /"}],
            }
        )


def test_model_cannot_rewrite_narration(tmp_path, monkeypatch):
    brief_path = tmp_path / "brief.json"
    brief_path.write_text(json.dumps(_short_brief()), encoding="utf-8")

    def fake_complete(messages, **kwargs):
        return {
            "summary": "rewrite",
            "operations": [
                {
                    "op": "set_timeline",
                    "sections": [
                        {
                            "evidence_id": "e1",
                            "start_seconds": 0,
                            "end_seconds": 1,
                            "narration": "被改写的旁白",
                        }
                    ],
                }
            ],
        }

    monkeypatch.setattr("lib.deepseek_director.complete_json", fake_complete)
    with pytest.raises(Exception) as caught:
        plan_project(brief_path, "rewritten", tmp_path, allow_short=True)
    assert not (tmp_path / "rewritten").exists()
    assert "旁白" not in str(caught.value) or "rejected" in str(caught.value).lower() or "Additional" in str(caught.value) or "narration" in str(caught.value)


def test_plan_uses_model_timings_and_keeps_brief_text(tmp_path, monkeypatch):
    brief = _short_brief()
    brief_path = tmp_path / "brief.json"
    brief_path.write_text(json.dumps(brief), encoding="utf-8")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-plan-secret")

    def fake_complete(messages, **kwargs):
        blob = json.dumps(messages, ensure_ascii=False)
        assert "sk-plan-secret" not in blob
        assert "json" in messages[0]["content"]
        assert "第一句旁白保持不变。" in messages[1]["content"]
        return {
            "summary": "把第二条放在前面。",
            "_deepseek_model": "deepseek-flash",
            "operations": [
                {"op": "use_pipeline", "pipeline": "documentary-montage"},
                {"op": "lock_runtime", "render_runtime": "ffmpeg"},
                {
                    "op": "set_timeline",
                    "sections": [
                        {"evidence_id": "e2", "start_seconds": 0, "end_seconds": 0.8},
                        {"evidence_id": "e1", "start_seconds": 0.8, "end_seconds": 2},
                    ],
                },
            ],
        }

    monkeypatch.setattr("lib.deepseek_director.complete_json", fake_complete)
    report = plan_project(brief_path, "planned", tmp_path, allow_short=True)
    script = json.loads(
        (tmp_path / "planned" / "artifacts" / "script.json").read_text(encoding="utf-8")
    )
    assert [section["text"] for section in script["sections"]] == [
        "第二句旁白保持不变。",
        "第一句旁白保持不变。",
    ]
    assert script["sections"][0]["end_seconds"] == 0.8
    assert report["pipeline"] == "documentary-montage"
    messages = build_messages(brief, ["documentary-montage"])
    assert "sk-plan-secret" not in json.dumps(messages)


def test_deepseek_client_posts_only_to_official_host(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-direct-secret")
    seen = {}

    def post(url, headers=None, json=None, timeout=None):
        seen["url"] = url
        seen["body"] = json
        seen["auth"] = headers["Authorization"]
        return _Response(
            {
                "model": "deepseek-flash",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": json_dumps_ops(),
                        },
                    }
                ],
            }
        )

    monkeypatch.setattr("lib.deepseek_client.requests.post", post)
    parsed = complete_json([{"role": "user", "content": "return json"}])
    assert seen["url"] == CHAT_URL
    assert "openrouter.ai" not in seen["url"]
    assert seen["body"]["model"] == "deepseek-flash"
    assert seen["body"]["response_format"] == {"type": "json_object"}
    assert seen["auth"] == "Bearer sk-direct-secret"
    assert parsed["operations"][0]["op"] == "lock_runtime"


def json_dumps_ops() -> str:
    return json.dumps(
        {
            "summary": "lock",
            "operations": [{"op": "lock_runtime", "render_runtime": "ffmpeg"}],
        }
    )


def test_openrouter_host_and_foreign_model_are_refused(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_BASE", "https://openrouter.ai/api/v1")
    with pytest.raises(DeepSeekDirectorError):
        resolve_chat_url()
    monkeypatch.delenv("DEEPSEEK_API_BASE", raising=False)
    monkeypatch.setenv("DEEPSEEK_MODEL", "google/gemini-3.8-flash-tts")
    with pytest.raises(DeepSeekDirectorError):
        resolve_model()


def test_empty_and_truncated_json_are_refused(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-direct-secret")

    def empty(url, headers=None, json=None, timeout=None):
        return _Response(
            {"choices": [{"finish_reason": "stop", "message": {"content": "   "}}]}
        )

    monkeypatch.setattr("lib.deepseek_client.requests.post", empty)
    with pytest.raises(DeepSeekDirectorError, match="empty content"):
        complete_json([{"role": "user", "content": "json"}])

    def truncated(url, headers=None, json=None, timeout=None):
        return _Response(
            {"choices": [{"finish_reason": "length", "message": {"content": "{"}}]}
        )

    monkeypatch.setattr("lib.deepseek_client.requests.post", truncated)
    with pytest.raises(DeepSeekDirectorError, match="max_tokens"):
        complete_json([{"role": "user", "content": "json"}])


def test_request_exception_redacts_key(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-direct-secret")

    def post(url, headers=None, json=None, timeout=None):
        raise requests.RequestException("timeout sk-direct-secret")

    monkeypatch.setattr("lib.deepseek_client.requests.post", post)
    with pytest.raises(DeepSeekDirectorError) as caught:
        complete_json([{"role": "user", "content": "json"}])
    assert "sk-direct-secret" not in str(caught.value)


def test_long_spec_does_not_render_without_an_explicit_flag(tmp_path):
    brief = load_evidence_brief(TEMPLATE)
    info = compile_project(brief, "ai-news-a", tmp_path)
    with pytest.raises(ProductionSpecError, match="smoke"):
        render_project(Path(info["project_dir"]), placeholders=True)


def test_overflow_is_not_trimmed(tmp_path):
    info = compile_project(_short_brief(), "align", tmp_path, allow_short=True)
    project = Path(info["project_dir"])
    long_wav = project / "assets" / "audio" / "narr-e1.wav"
    short_wav = project / "assets" / "audio" / "narr-e2.wav"
    long_wav.parent.mkdir(parents=True, exist_ok=True)
    long_wav.write_bytes(pcm_s16le_to_wav(b"\x00\x01" * int(24000 * 3)))
    short_wav.write_bytes(pcm_s16le_to_wav(b"\x00\x01" * 2400))
    report = import_narration_wavs(project, {"e1": long_wav, "e2": short_wav})
    assert report["status"] == "overflow"
    assert not (project / "assets" / "audio" / "narration-mix.wav").exists()


def test_dub_aligns_short_take_without_changing_words(tmp_path, monkeypatch):
    info = compile_project(_short_brief(), "dubbed", tmp_path, allow_short=True)
    project = Path(info["project_dir"])
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-openrouter-secret")
    pcm = b"\x01\x00" * 4800  # 0.2s at 24 kHz

    def post(url, headers=None, json=None, timeout=None):
        assert url == "https://openrouter.ai/api/v1/audio/speech"
        assert json["voice"] == "Despina"
        assert json["model"] == "google/gemini-3.8-flash-tts"
        assert json["response_format"] == "pcm"
        assert json["input"] in {"第一句旁白保持不变。", "第二句旁白保持不变。"}
        return type("R", (), {
            "content": pcm,
            "status_code": 200,
            "headers": {"Content-Type": "audio/pcm"},
        })()

    monkeypatch.setattr("tools.audio.openrouter_tts.requests.post", post)
    report = dub_project("dubbed", tmp_path)
    assert report["alignment"]["status"] == "aligned"
    mix = project / "assets" / "audio" / "narration-mix.wav"
    with wave.open(str(mix), "rb") as handle:
        assert handle.getframerate() == 24000
        assert abs(handle.getnframes() / handle.getframerate() - 2) < 0.02
    script = json.loads((project / "artifacts" / "script.json").read_text(encoding="utf-8"))
    assert script["sections"][0]["text"] == "第一句旁白保持不变。"


@pytest.mark.skipif(shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None, reason="ffmpeg required")
def test_smoke_renders_timeline_through_video_compose(tmp_path):
    report = smoke_project(tmp_path)
    output = Path(report["output"])
    assert output.is_file()
    assert output.stat().st_size > 1000
    assert abs(report["duration_seconds"] - 2) < 0.35
    rendered = json.loads(
        (Path(report["project_dir"]) / "artifacts" / "render_report.json").read_text(encoding="utf-8")
    )
    validate_artifact("render_report", rendered)
    assert rendered["metadata"]["operation"] == "compose"
    assert rendered["metadata"]["narration_voice"] == "Despina"
