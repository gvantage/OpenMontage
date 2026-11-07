---
name: deepseek-director
description: Opt-in direct DeepSeek production director and OpenRouter Despina narration for an OpenMontage timeline.
---

# DeepSeek director and Despina narration

Use this when the user wants DeepSeek to direct an OpenMontage production or
wants narration from OpenRouter voice Despina. The coding agent remains the
default director. Do not change `config.yaml` `llm.provider` to pretend the
runtime calls DeepSeek.

## Credentials

Read `DEEPSEEK_API_KEY` and `OPENROUTER_API_KEY` from the environment only.
Do not write them into briefs, checkpoints, logs, or command lines. A missing
key stops that call. Do not send DeepSeek traffic to OpenRouter, and do not
send Despina speech to `api.deepseek.com`.

`openrouter_tts` stays out of `tts_selector` ranking until
`OPENMONTAGE_OPENROUTER_TTS=1`. The `dub` command calls the tool directly and
does not need that flag.

## Direct DeepSeek

`lib.deepseek_director` posts to `https://api.deepseek.com/chat/completions`.
Models are `deepseek-flash` (default) and `deepseek-v4-pro`. The response must
be JSON operations:

- `use_pipeline` — an OpenMontage pipeline whose first stage is `idea`
- `lock_runtime` — only `ffmpeg`
- `order_evidence` — a permutation of brief evidence ids
- `set_timeline` — contiguous timings that end at `target_duration_seconds`

Narration text is copied from the evidence file. Reject shell, paths, and
any operation outside that list. Apply nothing until the whole document
validates.

News specs must be 120–180 seconds. Compile one project per brief:

```bash
python -m lib.deepseek_director compile --offline \
  --brief briefs/news-a.json --project-id ai-news-a --projects-dir projects
python -m lib.deepseek_director plan \
  --brief briefs/news-a.json --project-id ai-news-a --projects-dir projects
```

`plan` spends DeepSeek tokens. `compile --offline` does not.

## Despina

`openrouter_tts` POSTs `https://openrouter.ai/api/v1/audio/speech` with model
`google/gemini-3.8-flash-tts`, voice `Despina`, and `response_format` `pcm`.
Raw 16-bit little-endian mono PCM is wrapped as WAV at 24000 Hz unless the
response `Content-Type` includes another `rate`. Do not label MP3 as WAV.

```bash
python -m lib.deepseek_director dub --project-id ai-news-a --projects-dir projects
```

A take longer than its slot is an overflow. Do not trim the words. A shorter
take is padded with silence so the mix stays on the timeline.

## Render

The spec writes `brief`, `script`, `scene_plan`, `asset_manifest`, and
`edit_decisions` with `render_runtime: ffmpeg`. The idea checkpoint stays
`awaiting_human`. Placeholder color plates are not footage and not a finished
film.

```bash
python -m lib.deepseek_director smoke --projects-dir /tmp/openmontage-smoke
python -m lib.deepseek_director render --project-id ai-news-a \
  --projects-dir projects --placeholders --max-seconds 4
```

`smoke` is a 2 second `video_compose` compose. It does not call either API.
A 120–180 second spec does not render until `--max-seconds` or
`--full-timeline` is passed. `--full-timeline` still uses placeholder plates
unless real footage has replaced `assets/video/plate-*.mp4`.
