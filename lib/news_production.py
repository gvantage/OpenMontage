"""Build a renderable OpenMontage timeline from an evidence brief.

The brief is the only source of narration text and factual claims. DeepSeek
may order those evidence ids and set their timings. It cannot invent claims,
choose a shell command, or change the spoken words.

Outputs are schema-valid artifacts plus an idea checkpoint. Placeholder
plates and silent WAV files exist so ``video_compose`` operation ``compose``
can render the timeline before any paid narration call. Despina WAV files,
once imported, are aligned into the same slots.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import wave
from pathlib import Path
from typing import Any

import jsonschema

from lib.checkpoint import init_project, write_checkpoint
from lib.pcm_wav import GEMINI_TTS_PCM_RATE, pcm_s16le_to_wav
from lib.pipeline_loader import list_pipelines, load_pipeline_readonly
from schemas.artifacts import validate_artifact
from styles.playbook_loader import list_playbooks


NEWS_MIN_SECONDS = 120
NEWS_MAX_SECONDS = 180
MODEL_ID = "google/gemini-3.8-flash-tts"
VOICE = "Despina"
PIPELINE_DEFAULT = "documentary-montage"

BRIEF_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "title",
        "hook",
        "tone",
        "style",
        "target_platform",
        "target_duration_seconds",
        "evidence",
    ],
    "properties": {
        "title": {"type": "string", "minLength": 1, "maxLength": 200},
        "hook": {"type": "string", "minLength": 1, "maxLength": 500},
        "core_message": {"type": "string", "maxLength": 500},
        "cta": {"type": "string", "maxLength": 300},
        "tone": {"type": "string", "minLength": 1, "maxLength": 120},
        "style": {"type": "string", "minLength": 1, "maxLength": 120},
        "target_audience": {"type": "string", "maxLength": 200},
        "target_platform": {
            "type": "string",
            "enum": ["youtube", "instagram", "tiktok", "linkedin", "generic"],
        },
        "target_duration_seconds": {"type": "number"},
        "language": {"type": "string", "maxLength": 16},
        "pipeline": {"type": "string", "maxLength": 80},
        "evidence": {
            "type": "array",
            "minItems": 2,
            "maxItems": 12,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["id", "claim", "narration", "source"],
                "properties": {
                    "id": {"type": "string", "pattern": "^[A-Za-z0-9_-]{1,40}$"},
                    "claim": {"type": "string", "minLength": 1, "maxLength": 500},
                    "narration": {"type": "string", "minLength": 1, "maxLength": 2000},
                    "source": {"type": "string", "minLength": 1, "maxLength": 500},
                    "visual_note": {"type": "string", "maxLength": 500},
                },
            },
        },
    },
}

OPERATION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["summary", "operations"],
    "properties": {
        "summary": {"type": "string", "minLength": 1, "maxLength": 2000},
        "_deepseek_model": {"type": "string"},
        "operations": {
            "type": "array",
            "minItems": 1,
            "maxItems": 8,
            "items": {
                "oneOf": [
                    {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["op", "pipeline"],
                        "properties": {
                            "op": {"const": "use_pipeline"},
                            "pipeline": {"type": "string", "minLength": 1, "maxLength": 80},
                        },
                    },
                    {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["op", "render_runtime"],
                        "properties": {
                            "op": {"const": "lock_runtime"},
                            "render_runtime": {"const": "ffmpeg"},
                        },
                    },
                    {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["op", "evidence_ids"],
                        "properties": {
                            "op": {"const": "order_evidence"},
                            "evidence_ids": {
                                "type": "array",
                                "minItems": 2,
                                "maxItems": 12,
                                "uniqueItems": True,
                                "items": {"type": "string"},
                            },
                        },
                    },
                    {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["op", "sections"],
                        "properties": {
                            "op": {"const": "set_timeline"},
                            "sections": {
                                "type": "array",
                                "minItems": 2,
                                "maxItems": 12,
                                "items": {
                                    "type": "object",
                                    "additionalProperties": False,
                                    "required": [
                                        "evidence_id",
                                        "start_seconds",
                                        "end_seconds",
                                    ],
                                    "properties": {
                                        "evidence_id": {"type": "string"},
                                        "start_seconds": {"type": "number", "minimum": 0},
                                        "end_seconds": {"type": "number", "minimum": 0},
                                    },
                                },
                            },
                        },
                    },
                ]
            },
        },
    },
}

_PLATE_COLORS = ("0x102033", "0x1d3a2f", "0x3a2a1d", "0x2a2040")


class ProductionSpecError(ValueError):
    """The brief or director operations cannot become a renderable spec."""


def load_evidence_brief(path: Path, *, allow_short: bool = False) -> dict[str, Any]:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProductionSpecError(f"Could not read evidence brief: {exc}") from exc
    return validate_evidence_brief(data, allow_short=allow_short)


def validate_evidence_brief(data: dict[str, Any], *, allow_short: bool = False) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise ProductionSpecError("Evidence brief must be a JSON object")
    try:
        jsonschema.validate(data, BRIEF_SCHEMA)
    except jsonschema.ValidationError as exc:
        raise ProductionSpecError(f"Evidence brief is invalid: {exc.message}") from exc
    duration = float(data["target_duration_seconds"])
    low = 1 if allow_short else NEWS_MIN_SECONDS
    high = NEWS_MAX_SECONDS
    if duration < low or duration > high:
        raise ProductionSpecError(
            f"target_duration_seconds must be between {low} and {high}"
        )
    ids = [item["id"] for item in data["evidence"]]
    if len(ids) != len(set(ids)):
        raise ProductionSpecError("Evidence ids must be unique")
    return data


def idea_pipelines() -> list[str]:
    names: list[str] = []
    for name in list_pipelines():
        try:
            manifest = load_pipeline_readonly(name)
        except Exception:
            continue
        stages = manifest.get("stages") or []
        if stages and stages[0].get("name") == "idea":
            names.append(name)
    return sorted(names)


def parse_director_document(document: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(document, dict):
        raise ProductionSpecError("Director response must be a JSON object")
    try:
        jsonschema.validate(document, OPERATION_SCHEMA)
    except jsonschema.ValidationError as exc:
        raise ProductionSpecError(
            "Director operations were rejected: " + exc.message
        ) from exc
    return document


def resolve_operations(
    document: dict[str, Any],
    brief: dict[str, Any],
) -> tuple[str, list[dict[str, Any]]]:
    """Return ``(pipeline, sections)`` from a validated director document."""

    parse_director_document(document)
    known = set(idea_pipelines())
    pipeline = brief.get("pipeline") or PIPELINE_DEFAULT
    order = [item["id"] for item in brief["evidence"]]
    timeline: list[dict[str, Any]] | None = None
    for operation in document["operations"]:
        kind = operation["op"]
        if kind == "use_pipeline":
            pipeline = operation["pipeline"]
        elif kind == "lock_runtime":
            continue
        elif kind == "order_evidence":
            order = list(operation["evidence_ids"])
        elif kind == "set_timeline":
            timeline = list(operation["sections"])
        else:
            raise ProductionSpecError(f"Unsupported operation {kind!r}")
    if pipeline not in known:
        raise ProductionSpecError(
            f"Pipeline {pipeline!r} is not an OpenMontage pipeline whose first "
            "stage is idea. The director will not write a later stage first."
        )
    evidence_ids = {item["id"] for item in brief["evidence"]}
    if set(order) != evidence_ids or len(order) != len(evidence_ids):
        raise ProductionSpecError("order_evidence must list each evidence id once")
    if timeline is None:
        sections = even_sections(order, float(brief["target_duration_seconds"]))
    else:
        sections = _validate_timeline(
            timeline,
            evidence_ids,
            float(brief["target_duration_seconds"]),
        )
        timeline_order = [section["evidence_id"] for section in sections]
        if "order_evidence" in {op["op"] for op in document["operations"]}:
            if timeline_order != order:
                raise ProductionSpecError(
                    "order_evidence and set_timeline list different orders"
                )
    return pipeline, sections


def even_sections(evidence_ids: list[str], total_seconds: float) -> list[dict[str, Any]]:
    total_ms = _ms(total_seconds)
    count = len(evidence_ids)
    if count < 1 or total_ms < count:
        raise ProductionSpecError("Timeline is too short for the evidence list")
    base, remainder = divmod(total_ms, count)
    cursor = 0
    sections = []
    for index, evidence_id in enumerate(evidence_ids):
        duration = base + (1 if index < remainder else 0)
        if duration < 1 and total_seconds >= NEWS_MIN_SECONDS:
            raise ProductionSpecError("Each news section must be at least 1 second")
        sections.append(
            {
                "evidence_id": evidence_id,
                "start_seconds": cursor / 1000,
                "end_seconds": (cursor + duration) / 1000,
            }
        )
        cursor += duration
    return sections


def _validate_timeline(
    sections: list[dict[str, Any]],
    evidence_ids: set[str],
    total_seconds: float,
) -> list[dict[str, Any]]:
    seen: list[str] = []
    cursor = 0
    normalized: list[dict[str, Any]] = []
    minimum_ms = 1000 if total_seconds >= NEWS_MIN_SECONDS else 400
    for section in sections:
        evidence_id = section["evidence_id"]
        if evidence_id not in evidence_ids:
            raise ProductionSpecError(
                f"Timeline evidence id {evidence_id!r} is not in the brief"
            )
        start = _ms(section["start_seconds"])
        end = _ms(section["end_seconds"])
        if start != cursor:
            raise ProductionSpecError("Timeline sections must be contiguous from 0")
        if end - start < minimum_ms:
            raise ProductionSpecError(
                f"Section {evidence_id} is shorter than {minimum_ms} ms"
            )
        seen.append(evidence_id)
        normalized.append(
            {
                "evidence_id": evidence_id,
                "start_seconds": start / 1000,
                "end_seconds": end / 1000,
            }
        )
        cursor = end
    if set(seen) != evidence_ids or len(seen) != len(evidence_ids):
        raise ProductionSpecError("Timeline must include each evidence id once")
    if cursor != _ms(total_seconds):
        raise ProductionSpecError(
            "Timeline length must equal target_duration_seconds. "
            "Narration text was not changed to force a fit."
        )
    return normalized


def compile_project(
    brief: dict[str, Any],
    project_id: str,
    projects_dir: Path,
    *,
    pipeline: str | None = None,
    sections: list[dict[str, Any]] | None = None,
    allow_short: bool = False,
    frame_size: tuple[int, int] = (1280, 720),
    director_summary: str = "",
) -> dict[str, Any]:
    """Write the project marker, idea checkpoint, and renderable artifacts."""

    validate_evidence_brief(brief, allow_short=allow_short)
    _check_project_id(project_id)
    chosen = pipeline or brief.get("pipeline") or PIPELINE_DEFAULT
    if chosen not in set(idea_pipelines()):
        raise ProductionSpecError(
            f"Pipeline {chosen!r} cannot take an idea checkpoint from this director"
        )
    timeline = sections or even_sections(
        [item["id"] for item in brief["evidence"]],
        float(brief["target_duration_seconds"]),
    )
    by_id = {item["id"]: item for item in brief["evidence"]}
    root = Path(projects_dir)
    playbook = brief["style"] if brief["style"] in set(list_playbooks()) else None
    project_dir = init_project(
        project_id,
        title=brief["title"],
        pipeline_type=chosen,
        pipeline_dir=root,
        style_playbook=playbook,
    )
    artifacts = _build_artifacts(
        brief,
        project_id,
        chosen,
        timeline,
        by_id,
        frame_size,
        director_summary,
    )
    artifact_dir = project_dir / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    for name, payload in artifacts.items():
        validate_artifact(name, payload)
        _write_json(artifact_dir / f"{name}.json", payload)
    write_checkpoint(
        root,
        project_id,
        "idea",
        "awaiting_human",
        {"brief": artifacts["brief"], "decision_log": artifacts["decision_log"]},
        pipeline_type=chosen,
        style_playbook=playbook,
        human_approval_required=True,
        human_approved=False,
        metadata={
            "director": "deepseek-direct",
            "narration_model": MODEL_ID,
            "narration_voice": VOICE,
            "spec_status": "renderable-placeholder-visuals",
        },
    )
    return {
        "project_dir": str(project_dir),
        "project_id": project_id,
        "pipeline": chosen,
        "duration_seconds": float(brief["target_duration_seconds"]),
        "section_count": len(timeline),
        "awaiting_human": True,
    }


def materialize_placeholders(
    project_dir: Path,
    *,
    max_seconds: float | None = None,
) -> dict[str, Any]:
    """Write silent WAV files and color plates.

    A short proof writes plates under ``assets/video/proof`` so the spec's
    full-length plate paths stay untouched.
    """

    project_dir = Path(project_dir)
    edit = _read_json(project_dir / "artifacts" / "edit_decisions.json")
    script = _read_json(project_dir / "artifacts" / "script.json")
    cuts = _clip_cuts(edit["cuts"], max_seconds)
    proof = max_seconds is not None
    audio_dir = project_dir / "assets" / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    size = _frame_size(edit)
    mix_parts: list[Path] = []
    resolved_cuts = []
    alignments = []
    for index, cut in enumerate(cuts):
        duration = float(cut["out_seconds"]) - float(cut["in_seconds"])
        plate_name = Path(cut["source"]).name
        plate = (
            project_dir / "assets" / "video" / "proof" / plate_name
            if proof
            else project_dir / cut["source"]
        )
        _write_plate(plate, duration, _PLATE_COLORS[index % len(_PLATE_COLORS)], size)
        evidence_id = plate_name.removeprefix("plate-").removesuffix(".mp4")
        wav_path = (
            audio_dir / "proof" / f"narr-{evidence_id}.wav"
            if proof
            else audio_dir / f"narr-{evidence_id}.wav"
        )
        _write_silence(wav_path, duration)
        mix_parts.append(wav_path)
        resolved = dict(cut)
        resolved["source"] = str(plate.resolve())
        resolved_cuts.append(resolved)
        alignments.append(
            {
                "section_id": f"sec-{evidence_id}",
                "status": "placeholder",
                "slot_seconds": duration,
                "wav": str(wav_path.relative_to(project_dir)),
            }
        )
    mix_name = "proof-mix.wav" if proof else "narration-mix.wav"
    mix = audio_dir / mix_name
    _concat_wavs(mix_parts, mix)
    report = {
        "mode": "placeholder",
        "status": "placeholder",
        "voice": VOICE,
        "model": MODEL_ID,
        "sample_rate": GEMINI_TTS_PCM_RATE,
        "mix": f"assets/audio/{mix_name}",
        "sections": alignments,
    }
    if not proof:
        _write_json(project_dir / "artifacts" / "narration_alignment.json", report)
    else:
        _write_json(project_dir / "artifacts" / "narration_alignment_proof.json", report)
    unused_sections = len(script["sections"]) - len(cuts)
    return {"mix": mix, "cuts": resolved_cuts, "clipped_sections": max(0, unused_sections)}


def import_narration_wavs(project_dir: Path, wavs_by_evidence: dict[str, Path]) -> dict[str, Any]:
    """Align already-written Despina WAV files onto the spec timeline.

    Speech longer than its slot is reported and not trimmed. The mix is
    written only when every section fits.
    """

    project_dir = Path(project_dir)
    script = _read_json(project_dir / "artifacts" / "script.json")
    manifest = _read_json(project_dir / "artifacts" / "asset_manifest.json")
    audio_dir = project_dir / "assets" / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    aligned_paths: list[Path] = []
    sections_report = []
    overflow = False
    for section in script["sections"]:
        evidence_id = section["id"].removeprefix("sec-")
        source = wavs_by_evidence.get(evidence_id)
        if source is None or not Path(source).is_file():
            raise ProductionSpecError(f"Missing narration WAV for {evidence_id}")
        slot = float(section["end_seconds"]) - float(section["start_seconds"])
        destination = audio_dir / f"aligned-{evidence_id}.wav"
        fit = _fit_wav(Path(source), destination, slot)
        sections_report.append({"section_id": section["id"], **fit})
        if fit["status"] == "overflow":
            overflow = True
            if destination.exists():
                destination.unlink()
        else:
            aligned_paths.append(destination)
            _mark_narration_asset(manifest, evidence_id, Path(source), fit)
    report: dict[str, Any] = {
        "mode": "dub",
        "status": "overflow" if overflow else "aligned",
        "voice": VOICE,
        "model": MODEL_ID,
        "sample_rate": GEMINI_TTS_PCM_RATE,
        "sections": sections_report,
    }
    if not overflow:
        mix = audio_dir / "narration-mix.wav"
        _concat_wavs(aligned_paths, mix)
        report["mix"] = "assets/audio/narration-mix.wav"
        _write_json(project_dir / "artifacts" / "asset_manifest.json", manifest)
    _write_json(project_dir / "artifacts" / "narration_alignment.json", report)
    return report


def render_project(
    project_dir: Path,
    *,
    placeholders: bool = False,
    max_seconds: float | None = None,
    full_timeline: bool = False,
) -> dict[str, Any]:
    """Render the spec with ``video_compose`` operation ``compose``.

    Placeholder plates are not a finished news film. A timeline longer than
    15 seconds is refused unless ``full_timeline`` or ``max_seconds`` is set,
    so a spec compile does not start a long encode by itself.
    """

    project_dir = Path(project_dir)
    edit = _read_json(project_dir / "artifacts" / "edit_decisions.json")
    script = _read_json(project_dir / "artifacts" / "script.json")
    duration = float(script["total_duration_seconds"])
    if duration > 15 and not full_timeline and max_seconds is None:
        raise ProductionSpecError(
            "This timeline is a full news spec. Run `smoke` for the short "
            "render check, pass --max-seconds for a short proof, or pass "
            "--full-timeline to encode placeholder plates for the whole spec. "
            "Placeholder plates are not the finished film."
        )
    alignment_path = project_dir / "artifacts" / "narration_alignment.json"
    alignment = _read_json(alignment_path) if alignment_path.exists() else None
    if alignment and alignment.get("status") == "overflow":
        raise ProductionSpecError(
            "Narration is longer than its timeline slot. The mix was not written "
            "and speech was not trimmed. Retime the spec before rendering."
        )
    use_dub = bool(alignment and alignment.get("mode") == "dub" and max_seconds is None)
    if use_dub:
        mix = project_dir / "assets" / "audio" / "narration-mix.wav"
        if not mix.is_file():
            raise ProductionSpecError("Dub alignment exists but the narration mix is missing")
        cuts = []
        for cut in edit["cuts"]:
            source = project_dir / cut["source"]
            if not source.is_file():
                raise ProductionSpecError(
                    f"Cut source missing: {cut['source']}. "
                    "Footage plates are not generated during dub."
                )
            cuts.append({**cut, "source": str(source.resolve())})
    elif placeholders or (alignment and alignment.get("mode") == "placeholder" and max_seconds is None):
        materialized = materialize_placeholders(project_dir, max_seconds=max_seconds)
        mix = materialized["mix"]
        cuts = materialized["cuts"]
    else:
        raise ProductionSpecError(
            "No narration mix. Run dub, or render with placeholders for a silent proof."
        )
    output = project_dir / "renders" / (
        "timeline-proof.mp4" if max_seconds else "timeline.mp4"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    from tools.video.video_compose import VideoCompose

    composed = dict(edit)
    composed["cuts"] = cuts
    result = VideoCompose().execute(
        {
            "operation": "compose",
            "edit_decisions": composed,
            "audio_path": str(mix),
            "output_path": str(output),
            "preset": "ultrafast",
            "crf": 30,
        }
    )
    if not result.success:
        raise ProductionSpecError(f"video_compose failed: {result.error}")
    probed = _probe_duration(output)
    report = {
        "version": "1.0",
        "outputs": [
            {
                "path": str(output),
                "format": "mp4",
                "codec": "libx264",
                "audio_codec": "aac",
                "resolution": _resolution(edit),
                "fps": 30,
                "duration_seconds": probed if probed is not None else duration,
            }
        ],
        "warnings": [
            "Placeholder plates stand in for footage. This file is a timeline render, not a finished news film.",
        ],
        "render_grammar": "documentary-montage"
        if edit.get("renderer_family") == "documentary-montage"
        else None,
        "metadata": {
            "tool": "video_compose",
            "operation": "compose",
            "render_runtime": "ffmpeg",
            "narration_voice": VOICE,
            "narration_model": MODEL_ID,
        },
    }
    if report["render_grammar"] is None:
        report.pop("render_grammar")
    validate_artifact("render_report", report)
    _write_json(project_dir / "artifacts" / "render_report.json", report)
    return {
        "output": str(output),
        "duration_seconds": report["outputs"][0]["duration_seconds"],
        "render_report": str(project_dir / "artifacts" / "render_report.json"),
    }


def smoke_project(dest: Path) -> dict[str, Any]:
    """Compile a 2 second fixture and render it. No API calls."""

    brief = {
        "title": "Render smoke fixture",
        "hook": "This fixture checks the timeline path.",
        "tone": "neutral",
        "style": "clean-professional",
        "target_platform": "generic",
        "target_duration_seconds": 2,
        "language": "zh",
        "evidence": [
            {
                "id": "e1",
                "claim": "Smoke claim one",
                "narration": "第一段烟雾测试旁白。",
                "source": "fixture",
            },
            {
                "id": "e2",
                "claim": "Smoke claim two",
                "narration": "第二段烟雾测试旁白。",
                "source": "fixture",
            },
        ],
    }
    info = compile_project(
        brief,
        "despina-render-smoke",
        dest,
        allow_short=True,
        frame_size=(320, 180),
        director_summary="Offline render smoke. Not a news film.",
    )
    rendered = render_project(
        Path(info["project_dir"]),
        placeholders=True,
        full_timeline=True,
    )
    return {**info, **rendered}


def _build_artifacts(
    brief: dict[str, Any],
    project_id: str,
    pipeline: str,
    timeline: list[dict[str, Any]],
    by_id: dict[str, dict[str, Any]],
    frame_size: tuple[int, int],
    director_summary: str,
) -> dict[str, dict[str, Any]]:
    key_points = [by_id[section["evidence_id"]]["claim"] for section in timeline]
    openmontage_brief = {
        "version": "1.0",
        "title": brief["title"],
        "hook": brief["hook"],
        "key_points": key_points,
        "core_message": brief.get("core_message") or brief["hook"],
        "tone": brief["tone"],
        "style": brief["style"],
        "target_platform": brief["target_platform"],
        "target_duration_seconds": float(brief["target_duration_seconds"]),
        "metadata": {
            "thematic_question": brief["title"],
            "language": brief.get("language") or "",
            "director": "deepseek-direct",
            "director_summary": director_summary,
            "music_plan": {
                "source": "none",
                "note": "This workflow carries narration. It does not generate music.",
            },
            "narration": {
                "provider": "openrouter",
                "tool": "openrouter_tts",
                "model": MODEL_ID,
                "voice": VOICE,
                "response_format": "pcm",
                "sample_rate_hz": GEMINI_TTS_PCM_RATE,
                "channels": 1,
                "sample_format": "s16le",
            },
            "visual_status": "placeholder_plates",
            "evidence": [
                {
                    "id": item["id"],
                    "claim": item["claim"],
                    "source": item["source"],
                }
                for item in brief["evidence"]
            ],
        },
    }
    if brief.get("target_audience"):
        openmontage_brief["target_audience"] = brief["target_audience"]
    if brief.get("cta"):
        openmontage_brief["cta"] = brief["cta"]

    sections = []
    scenes = []
    assets = []
    cuts = []
    narration_segments = []
    for section in timeline:
        item = by_id[section["evidence_id"]]
        section_id = f"sec-{item['id']}"
        scene_id = f"scene-{item['id']}"
        start = section["start_seconds"]
        end = section["end_seconds"]
        duration = end - start
        sections.append(
            {
                "id": section_id,
                "label": item["claim"],
                "text": item["narration"],
                "start_seconds": start,
                "end_seconds": end,
                "speaker_directions": "Clear measured narration for the Despina studio voice.",
                "delivery_cues": {
                    "pace": "measured",
                    "energy": "steady",
                    "delivery_note": brief["tone"],
                    "provider_text": item["narration"],
                },
                "source_ref": item["source"],
            }
        )
        scenes.append(
            {
                "id": scene_id,
                "type": "broll",
                "description": item.get("visual_note")
                or f"Placeholder plate for evidence {item['id']}: {item['claim']}",
                "start_seconds": start,
                "end_seconds": end,
                "script_section_id": section_id,
                "narrative_role": "evidence",
                "required_assets": [
                    {
                        "type": "video",
                        "description": item["claim"],
                        "source": "provided",
                    },
                    {
                        "type": "narration",
                        "description": "OpenRouter Despina narration",
                        "source": "generate",
                    },
                ],
            }
        )
        plate_rel = f"assets/video/plate-{item['id']}.mp4"
        narr_rel = f"assets/audio/narr-{item['id']}.wav"
        assets.append(
            {
                "id": f"plate-{item['id']}",
                "type": "video",
                "path": plate_rel,
                "source_tool": "video_compose",
                "scene_id": scene_id,
                "generation_summary": "Color placeholder plate. Replace with footage before calling this a film.",
                "duration_seconds": duration,
                "resolution": f"{frame_size[0]}x{frame_size[1]}",
                "format": "mp4",
            }
        )
        assets.append(
            {
                "id": f"narr-{item['id']}",
                "type": "narration",
                "path": narr_rel,
                "source_tool": "openrouter_tts",
                "scene_id": scene_id,
                "provider": "openrouter",
                "model": MODEL_ID,
                "format": "wav",
                "duration_seconds": duration,
                "generation_summary": "Slot reserved for OpenRouter Despina PCM wrapped as WAV.",
                "voice_performance": {
                    "source_section_id": section_id,
                    "delivery_cues_applied": True,
                    "provider_text_used": True,
                    "provider_settings": {
                        "model": MODEL_ID,
                        "voice": VOICE,
                        "response_format": "pcm",
                        "sample_rate": GEMINI_TTS_PCM_RATE,
                        "endpoint": "https://openrouter.ai/api/v1/audio/speech",
                    },
                    "sample_approved": False,
                },
            }
        )
        cuts.append(
            {
                "id": f"cut-{item['id']}",
                "source": plate_rel,
                "in_seconds": 0,
                "out_seconds": duration,
                "reason": f"Timeline slot for evidence {item['id']}",
            }
        )
        narration_segments.append(
            {
                "asset_id": f"narr-{item['id']}",
                "start_seconds": start,
                "end_seconds": end,
            }
        )

    script = {
        "version": "1.0",
        "title": brief["title"],
        "total_duration_seconds": float(brief["target_duration_seconds"]),
        "voice_performance": {
            "performance_intent": "News narration on the Despina voice",
            "pacing_profile": "technical",
            "energy_curve": "steady",
            "pause_policy": "Silence pads a short take. A long take is not trimmed.",
            "sample_section_id": sections[0]["id"],
            "provider_notes": {
                "openrouter_tts": (
                    f"{MODEL_ID} voice {VOICE}, response_format pcm, "
                    f"{GEMINI_TTS_PCM_RATE} Hz mono s16le WAV"
                ),
            },
        },
        "sections": sections,
        "metadata": {
            "language": brief.get("language") or "",
            "pipeline": pipeline,
        },
    }
    scene_plan = {
        "version": "1.0",
        "scenes": scenes,
        "metadata": {"pipeline": pipeline, "visual_status": "placeholder_plates"},
    }
    asset_manifest = {
        "version": "1.0",
        "assets": assets,
        "metadata": {
            "narration_cost_status": "unquoted",
            "visual_status": "placeholder_plates",
        },
    }
    edit = {
        "version": "1.0",
        "render_runtime": "ffmpeg",
        "cuts": cuts,
        "audio": {"narration": {"segments": narration_segments}},
        "metadata": {
            "compose_target": {
                "width": frame_size[0],
                "height": frame_size[1],
                "fit": "pad",
            },
            "proposal_render_runtime": "ffmpeg",
            "pipeline": pipeline,
            "narration_voice": VOICE,
            "narration_model": MODEL_ID,
            "visual_status": "placeholder_plates",
        },
    }
    if pipeline == "documentary-montage":
        edit["renderer_family"] = "documentary-montage"
    decision_log = _decision_log(project_id, pipeline)
    return {
        "brief": openmontage_brief,
        "script": script,
        "scene_plan": scene_plan,
        "asset_manifest": asset_manifest,
        "edit_decisions": edit,
        "decision_log": decision_log,
    }


def _decision_log(project_id: str, pipeline: str) -> dict[str, Any]:
    others = [name for name in idea_pipelines() if name != pipeline]
    rejected = others[0] if others else "animated-explainer"
    return {
        "version": "1.0",
        "project_id": project_id,
        "decisions": [
            {
                "decision_id": "d-001",
                "stage": "idea",
                "category": "pipeline_selection",
                "subject": "OpenMontage pipeline",
                "options_considered": [
                    {
                        "option_id": pipeline,
                        "label": pipeline,
                        "score": 1,
                        "reason": "OpenMontage pipeline whose first stage is idea",
                    },
                    {
                        "option_id": rejected,
                        "label": rejected,
                        "score": 0.4,
                        "reason": "Also an OpenMontage pipeline",
                        "rejected_because": "The director locked a single pipeline for this spec",
                    },
                ],
                "selected": pipeline,
                "reason": "The evidence brief is compiled into this pipeline's idea checkpoint.",
                "user_visible": True,
                "user_approved": False,
                "confidence": 0.7,
            },
            {
                "decision_id": "d-002",
                "stage": "idea",
                "category": "render_runtime_selection",
                "subject": "render runtime",
                "options_considered": [
                    {
                        "option_id": "ffmpeg",
                        "label": "ffmpeg via video_compose compose",
                        "score": 1,
                        "reason": "Renders the plate timeline and the narration mix",
                    },
                    {
                        "option_id": "remotion",
                        "label": "remotion",
                        "score": 0.4,
                        "reason": "OpenMontage can render Remotion compositions",
                        "rejected_because": "This spec does not include an atelier entry",
                    },
                    {
                        "option_id": "hyperframes",
                        "label": "hyperframes",
                        "score": 0.4,
                        "reason": "OpenMontage can render HyperFrames compositions",
                        "rejected_because": "This spec does not include a HyperFrames composition",
                    },
                ],
                "selected": "ffmpeg",
                "reason": "video_compose operation compose is the render path for this spec.",
                "user_visible": True,
                "user_approved": False,
                "confidence": 0.8,
            },
            {
                "decision_id": "d-003",
                "stage": "idea",
                "category": "voice_selection",
                "subject": "Narration TTS provider",
                "options_considered": [
                    {
                        "option_id": "openrouter-despina",
                        "label": f"OpenRouter {MODEL_ID} {VOICE}",
                        "score": 1,
                        "reason": "Requested dubbing voice on the OpenRouter speech endpoint",
                    },
                    {
                        "option_id": "silent-placeholder",
                        "label": "Silent placeholder WAV",
                        "score": 0.2,
                        "reason": "Lets the timeline render before a paid dub",
                        "rejected_because": "Silence is only the render proof, not the narration",
                    },
                ],
                "selected": "openrouter-despina",
                "reason": "Section WAV slots are spoken by Despina through openrouter_tts.",
                "user_visible": True,
                "user_approved": False,
                "confidence": 0.9,
            },
        ],
    }


def _fit_wav(source: Path, destination: Path, slot_seconds: float) -> dict[str, Any]:
    with wave.open(str(source), "rb") as handle:
        rate = handle.getframerate()
        channels = handle.getnchannels()
        width = handle.getsampwidth()
        frames = handle.readframes(handle.getnframes())
    if channels != 1 or width != 2:
        raise ProductionSpecError(
            f"{source.name} must be mono 16-bit PCM WAV, got channels={channels} width={width}"
        )
    if rate != GEMINI_TTS_PCM_RATE:
        raise ProductionSpecError(
            f"{source.name} is {rate} Hz. Despina PCM is wrapped at {GEMINI_TTS_PCM_RATE} Hz. "
            "Refusing to resample."
        )
    slot_frames = int(round(slot_seconds * rate))
    src_frames = len(frames) // 2
    src_seconds = src_frames / rate
    if src_frames > slot_frames:
        return {
            "status": "overflow",
            "source_seconds": src_seconds,
            "slot_seconds": slot_seconds,
            "wav": str(source),
        }
    padded = frames + (b"\x00\x00" * (slot_frames - src_frames))
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(pcm_s16le_to_wav(padded, rate, 1))
    status = "exact" if src_frames == slot_frames else "padded"
    return {
        "status": status,
        "source_seconds": src_seconds,
        "slot_seconds": slot_seconds,
        "wav": str(destination),
    }


def _mark_narration_asset(manifest: dict, evidence_id: str, source: Path, fit: dict) -> None:
    del source
    for asset in manifest["assets"]:
        if asset["id"] != f"narr-{evidence_id}":
            continue
        asset["duration_seconds"] = fit["source_seconds"]
        asset["generation_summary"] = (
            "OpenRouter Despina PCM wrapped as WAV and aligned to the section slot."
        )
        asset["voice_performance"]["sample_approved"] = False
        asset["voice_performance"]["review_notes"] = fit["status"]
        break


def _write_silence(path: Path, duration: float) -> None:
    frames = max(1, int(round(duration * GEMINI_TTS_PCM_RATE)))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pcm_s16le_to_wav(b"\x00\x00" * frames, GEMINI_TTS_PCM_RATE, 1))


def _concat_wavs(parts: list[Path], destination: Path) -> None:
    buffers = []
    rate = None
    for part in parts:
        with wave.open(str(part), "rb") as handle:
            if handle.getnchannels() != 1 or handle.getsampwidth() != 2:
                raise ProductionSpecError(f"{part.name} is not mono 16-bit WAV")
            if rate is None:
                rate = handle.getframerate()
            elif handle.getframerate() != rate:
                raise ProductionSpecError("Narration WAV sample rates differ")
            buffers.append(handle.readframes(handle.getnframes()))
    if rate is None:
        raise ProductionSpecError("No narration sections to mix")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(pcm_s16le_to_wav(b"".join(buffers), rate, 1))


def _write_plate(path: Path, duration: float, color: str, size: tuple[int, int]) -> None:
    if shutil.which("ffmpeg") is None:
        raise ProductionSpecError("ffmpeg is required to render timeline plates")
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = size
    command = [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"color=c={color}:s={width}x{height}:r=30",
        "-f",
        "lavfi",
        "-i",
        "anullsrc=channel_layout=stereo:sample_rate=48000",
        "-t",
        f"{duration:.3f}",
        "-shortest",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-preset",
        "ultrafast",
        "-crf",
        "30",
        "-c:a",
        "aac",
        str(path),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0 or not path.is_file():
        detail = (completed.stderr or completed.stdout or "ffmpeg failed")[-400:]
        raise ProductionSpecError(f"Could not write plate {path.name}: {detail}")


def _clip_cuts(cuts: list[dict[str, Any]], max_seconds: float | None) -> list[dict[str, Any]]:
    if max_seconds is None:
        return [dict(cut) for cut in cuts]
    if max_seconds <= 0:
        raise ProductionSpecError("max_seconds must be positive")
    remaining = float(max_seconds)
    clipped = []
    for cut in cuts:
        duration = float(cut["out_seconds"]) - float(cut["in_seconds"])
        if remaining <= 0:
            break
        take = min(duration, remaining)
        cloned = dict(cut)
        cloned["out_seconds"] = float(cut["in_seconds"]) + take
        clipped.append(cloned)
        remaining -= take
    if not clipped:
        raise ProductionSpecError("max_seconds clipped away every cut")
    return clipped


def _frame_size(edit: dict[str, Any]) -> tuple[int, int]:
    target = (edit.get("metadata") or {}).get("compose_target") or {}
    return int(target.get("width", 1280)), int(target.get("height", 720))


def _resolution(edit: dict[str, Any]) -> str:
    width, height = _frame_size(edit)
    return f"{width}x{height}"


def _probe_duration(path: Path) -> float | None:
    if shutil.which("ffprobe") is None:
        return None
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=nw=1:nk=1",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        return None
    try:
        return float(completed.stdout.strip())
    except ValueError:
        return None


def _check_project_id(project_id: str) -> None:
    if not project_id or len(project_id) > 64:
        raise ProductionSpecError("project_id must be 1..64 characters")
    allowed = set("abcdefghijklmnopqrstuvwxyz0123456789-")
    if project_id[0] not in set("abcdefghijklmnopqrstuvwxyz0123456789"):
        raise ProductionSpecError("project_id must start with a letter or digit")
    if any(char not in allowed for char in project_id):
        raise ProductionSpecError("project_id may contain only lowercase letters, digits, and hyphens")


def _ms(value: float) -> int:
    return int(round(float(value) * 1000))


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProductionSpecError(f"Could not read {path.name}: {exc}") from exc
    if not isinstance(data, dict):
        raise ProductionSpecError(f"{path.name} must be a JSON object")
    return data


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
