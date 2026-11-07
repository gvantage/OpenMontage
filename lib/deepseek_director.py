"""Opt-in DeepSeek director for an OpenMontage news timeline.

The default OpenMontage director is still the coding agent. This module is
the direct DeepSeek path: it reads an evidence brief, asks DeepSeek for a
closed set of operations, and writes artifacts that ``video_compose`` can
render. Narration is a separate OpenRouter call.

Commands:

    python -m lib.deepseek_director compile --offline --brief FILE --project-id ID
    python -m lib.deepseek_director plan --brief FILE --project-id ID
    python -m lib.deepseek_director dub --project-id ID
    python -m lib.deepseek_director render --project-id ID --placeholders
    python -m lib.deepseek_director smoke
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from lib.deepseek_client import (
    DeepSeekDirectorError,
    complete_json,
    redact_secret,
    resolve_model,
)
from lib.news_production import (
    MODEL_ID,
    VOICE,
    ProductionSpecError,
    compile_project,
    idea_pipelines,
    import_narration_wavs,
    load_evidence_brief,
    render_project,
    resolve_operations,
    smoke_project,
)
from lib.pcm_wav import GEMINI_TTS_PCM_RATE


class DirectorError(RuntimeError):
    """User-facing director failure. The message is safe to print."""


def build_messages(brief: dict, pipelines: list[str]) -> list[dict[str, str]]:
    evidence = [
        {
            "id": item["id"],
            "claim": item["claim"],
            "narration": item["narration"],
            "source": item["source"],
        }
        for item in brief["evidence"]
    ]
    system = (
        "You are the OpenMontage production director. Reply with one json object "
        "and no markdown. You may only emit operations use_pipeline, lock_runtime, "
        "order_evidence, and set_timeline. lock_runtime must be ffmpeg. "
        "use_pipeline must be one of the supplied OpenMontage pipelines. "
        "order_evidence and set_timeline may only use evidence ids from the brief. "
        "Do not write narration, claims, shell commands, file paths, or tool arguments. "
        "The timeline must start at 0, be contiguous, and end at target_duration_seconds. "
        "Example json: "
        '{"summary":"Order the brief.","operations":['
        '{"op":"use_pipeline","pipeline":"documentary-montage"},'
        '{"op":"lock_runtime","render_runtime":"ffmpeg"},'
        '{"op":"set_timeline","sections":['
        '{"evidence_id":"e1","start_seconds":0,"end_seconds":75},'
        '{"evidence_id":"e2","start_seconds":75,"end_seconds":150}]}]}'
    )
    user = {
        "target_duration_seconds": brief["target_duration_seconds"],
        "title": brief["title"],
        "pipelines": pipelines,
        "evidence": evidence,
    }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
    ]


def plan_project(
    brief_path: Path,
    project_id: str,
    projects_dir: Path,
    *,
    allow_short: bool = False,
) -> dict:
    brief = load_evidence_brief(brief_path, allow_short=allow_short)
    pipelines = idea_pipelines()
    messages = build_messages(brief, pipelines)
    try:
        document = complete_json(messages, model=resolve_model())
        pipeline, sections = resolve_operations(document, brief)
    except (DeepSeekDirectorError, ProductionSpecError) as exc:
        raise DirectorError(str(exc)) from exc
    try:
        report = compile_project(
            brief,
            project_id,
            projects_dir,
            pipeline=pipeline,
            sections=sections,
            allow_short=allow_short,
            director_summary=str(document.get("summary") or ""),
        )
    except ProductionSpecError as exc:
        raise DirectorError(str(exc)) from exc
    report["director_summary"] = document.get("summary")
    report["deepseek_model"] = document.get("_deepseek_model")
    return report


def compile_offline(
    brief_path: Path,
    project_id: str,
    projects_dir: Path,
    *,
    allow_short: bool = False,
) -> dict:
    try:
        brief = load_evidence_brief(brief_path, allow_short=allow_short)
        return compile_project(
            brief,
            project_id,
            projects_dir,
            allow_short=allow_short,
            director_summary="Compiled from the evidence file without a model call.",
        )
    except ProductionSpecError as exc:
        raise DirectorError(str(exc)) from exc


def dub_project(project_id: str, projects_dir: Path) -> dict:
    project_dir = Path(projects_dir) / project_id
    script_path = project_dir / "artifacts" / "script.json"
    if not script_path.is_file():
        raise DirectorError(
            f"No script at {script_path}. Compile the evidence brief first."
        )
    script = json.loads(script_path.read_text(encoding="utf-8"))
    from tools.audio.openrouter_tts import MODEL_ID as TOOL_MODEL
    from tools.audio.openrouter_tts import OpenRouterTTS

    if TOOL_MODEL != MODEL_ID:
        raise DirectorError("openrouter_tts model drifted from the production spec")
    tool = OpenRouterTTS()
    wavs: dict[str, Path] = {}
    for section in script["sections"]:
        evidence_id = section["id"].removeprefix("sec-")
        output = project_dir / "assets" / "audio" / f"narr-{evidence_id}.wav"
        result = tool.execute(
            {
                "text": section["delivery_cues"]["provider_text"],
                "voice": VOICE,
                "model": MODEL_ID,
                "instructions": section.get("speaker_directions") or "",
                "output_path": str(output),
            }
        )
        if not result.success:
            secret = os.environ.get("OPENROUTER_API_KEY", "")
            raise DirectorError(redact_secret(result.error or "dub failed", secret))
        if result.data.get("voice") != VOICE or result.data.get("model") != MODEL_ID:
            raise DirectorError("Dub result did not use Despina on the locked model")
        if result.data.get("sample_rate") != GEMINI_TTS_PCM_RATE and result.data.get(
            "sample_rate_source"
        ) == "gemini-3.8-contract":
            raise DirectorError("PCM was wrapped at an unexpected contract rate")
        wavs[evidence_id] = output
    try:
        alignment = import_narration_wavs(project_dir, wavs)
    except ProductionSpecError as exc:
        raise DirectorError(str(exc)) from exc
    return {"project_dir": str(project_dir), "alignment": alignment}


def render_existing(
    project_id: str,
    projects_dir: Path,
    *,
    placeholders: bool,
    max_seconds: float | None,
    full_timeline: bool,
) -> dict:
    project_dir = Path(projects_dir) / project_id
    if not (project_dir / "artifacts" / "edit_decisions.json").is_file():
        raise DirectorError(f"No edit decisions in {project_dir}")
    try:
        return render_project(
            project_dir,
            placeholders=placeholders,
            max_seconds=max_seconds,
            full_timeline=full_timeline,
        )
    except ProductionSpecError as exc:
        raise DirectorError(str(exc)) from exc


def run_smoke(dest: Path) -> dict:
    try:
        return smoke_project(dest)
    except ProductionSpecError as exc:
        raise DirectorError(str(exc)) from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m lib.deepseek_director",
        description="Direct DeepSeek director and OpenRouter Despina timeline.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add_projects(command: argparse.ArgumentParser) -> None:
        command.add_argument(
            "--projects-dir",
            type=Path,
            default=Path("projects"),
            help="Directory that will contain projects/<project-id>/",
        )

    compile_cmd = sub.add_parser("compile", help="Write a spec from the evidence file")
    compile_cmd.add_argument("--brief", type=Path, required=True)
    compile_cmd.add_argument("--project-id", required=True)
    compile_cmd.add_argument(
        "--offline",
        action="store_true",
        help="Required. Compile does not call DeepSeek.",
    )
    add_projects(compile_cmd)

    plan_cmd = sub.add_parser("plan", help="Ask DeepSeek to order the evidence, then write the spec")
    plan_cmd.add_argument("--brief", type=Path, required=True)
    plan_cmd.add_argument("--project-id", required=True)
    add_projects(plan_cmd)

    dub_cmd = sub.add_parser("dub", help="Synthesize Despina narration and align it")
    dub_cmd.add_argument("--project-id", required=True)
    add_projects(dub_cmd)

    render_cmd = sub.add_parser("render", help="Render the timeline with video_compose")
    render_cmd.add_argument("--project-id", required=True)
    render_cmd.add_argument("--placeholders", action="store_true")
    render_cmd.add_argument("--max-seconds", type=float)
    render_cmd.add_argument("--full-timeline", action="store_true")
    add_projects(render_cmd)

    smoke_cmd = sub.add_parser("smoke", help="Offline 2 second render through video_compose")
    add_projects(smoke_cmd)

    args = parser.parse_args(argv)
    try:
        if args.command == "compile":
            if not args.offline:
                raise DirectorError(
                    "compile requires --offline. Use plan to call the DeepSeek API."
                )
            report = compile_offline(args.brief, args.project_id, args.projects_dir)
        elif args.command == "plan":
            report = plan_project(args.brief, args.project_id, args.projects_dir)
        elif args.command == "dub":
            report = dub_project(args.project_id, args.projects_dir)
        elif args.command == "render":
            report = render_existing(
                args.project_id,
                args.projects_dir,
                placeholders=args.placeholders,
                max_seconds=args.max_seconds,
                full_timeline=args.full_timeline,
            )
        elif args.command == "smoke":
            report = run_smoke(args.projects_dir)
        else:
            raise DirectorError(f"Unknown command {args.command}")
    except DirectorError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
