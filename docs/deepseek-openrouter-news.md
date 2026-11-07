# DeepSeek 导演与 OpenRouter Despina 配音

这是一条可选流程。OpenMontage 默认仍由编码助手阅读 pipeline 技能并调用工具。下面的命令让 DeepSeek 官方接口排列证据并写出可渲染的时间轴，再让 OpenRouter 的 Despina 声线按同一时间轴配音。

本文档依据 2026-11-07 的公开文档编写。本仓库的测试全部使用假响应，没有用真实密钥调用过这两个接口。

## 密钥

只放在环境变量里，不要写进 brief、checkpoint、日志或命令行参数。

```bash
export DEEPSEEK_API_KEY="你的密钥"
export OPENROUTER_API_KEY="你的密钥"
```

`dub` 只读 `OPENROUTER_API_KEY`。仅有这把密钥时，`tts_selector` 不会把 Despina 排进自动配音，已有流水线的默认选择保持不变。若要让选择器看见 `openrouter_tts`，再设置：

```bash
export OPENMONTAGE_OPENROUTER_TTS=1
```

可选：

```bash
export DEEPSEEK_MODEL="deepseek-flash"   # 或 deepseek-v4-pro
export DEEPSEEK_API_BASE="https://api.deepseek.com"
```

`DEEPSEEK_API_BASE` 只能是 `https://api.deepseek.com`。其他主机（包括 OpenRouter）会被拒绝。DeepSeek 不会改走 OpenRouter，Despina 也不会改走 DeepSeek。

`config.yaml` 里的 `llm.provider` 保持原样。那个字段不会发起 DeepSeek 请求。

## 证据文件

每条成片一份 JSON。新闻片时长必须是 120–180 秒。旁白和事实只来自这个文件，模型只能调整证据顺序和每段起止时间。

模板：`examples/ai-news-evidence.template.json`。

两条片就准备两份文件，例如 `briefs/news-a.json` 和 `briefs/news-b.json`，各编译到自己的 `project-id`。模板里的句子是占位，不是新闻成片。

## 命令

在 OpenMontage 仓库根目录、已激活的虚拟环境中执行。

离线编译（不调用 DeepSeek，不配音，不渲染）：

```bash
python -m lib.deepseek_director compile --offline \
  --brief briefs/news-a.json \
  --project-id ai-news-a \
  --projects-dir projects

python -m lib.deepseek_director compile --offline \
  --brief briefs/news-b.json \
  --project-id ai-news-b \
  --projects-dir projects
```

让 DeepSeek 排列时间轴（会消耗 DeepSeek 额度，仍不配音、不渲染）：

```bash
python -m lib.deepseek_director plan \
  --brief briefs/news-a.json \
  --project-id ai-news-a \
  --projects-dir projects
```

模型只能返回这些操作：`use_pipeline`、`lock_runtime`（只能是 `ffmpeg`）、`order_evidence`、`set_timeline`。shell、路径和旁白改写都会被整份拒绝，不会落盘。

直接 API：

- `POST https://api.deepseek.com/chat/completions`
- 模型：`deepseek-flash`（默认）或 `deepseek-v4-pro`
- `response_format` 为 `{"type":"json_object"}`

写出的项目包含：

- `checkpoint_idea.json`，状态是 `awaiting_human`
- `artifacts/brief.json`
- `artifacts/script.json`
- `artifacts/scene_plan.json`
- `artifacts/asset_manifest.json`
- `artifacts/edit_decisions.json`，`render_runtime` 为 `ffmpeg`

画面路径 `assets/video/plate-*.mp4` 在渲染前是占位色板，不是成片素材。

## Despina 配音

```bash
python -m lib.deepseek_director dub \
  --project-id ai-news-a \
  --projects-dir projects
```

这条命令会消耗 OpenRouter 额度。请求为：

```bash
POST https://openrouter.ai/api/v1/audio/speech
```

```json
{
  "model": "google/gemini-3.8-flash-tts",
  "input": "该段旁白原文",
  "voice": "Despina",
  "response_format": "pcm"
}
```

公开文档里的约定：

- OpenRouter create-speech 把 `audio/pcm` 定义为 16-bit little-endian 原始音频，响应不是 JSON。该参考页没有单独的采样率字段。
- `google/gemini-3.8-flash-tts` 的模型说明列出 Despina，以及 `pcm`（默认）和 `mp3`。
- OpenRouter 的 TTS 指南写明 Gemini TTS 在这个端点返回 PCM。
- Google Gemini 3.8 TTS 文档写明线性 PCM 为 24 kHz、单声道、16-bit signed little-endian。

因此适配器始终请求 `pcm`，并按 24000 Hz、单声道、16-bit 封装成 WAV。若响应 `Content-Type` 带有 `rate=`，WAV 头改用该采样率。MP3 不会被当成 WAV 保存。

对齐规则：

- 实录音短于时间槽：尾部补静音，时间轴不缩短。
- 实录音长于时间槽：记为 `overflow`，不裁切语句，也不写出混合音轨。
- 混合结果在 `assets/audio/narration-mix.wav`。
- 报告在 `artifacts/narration_alignment.json`。

费用没有内置单价。工具返回 `cost_status: unquoted`。

## 渲染

短烟雾测试，不调用任何付费接口，大约 2 秒：

```bash
python -m lib.deepseek_director smoke --projects-dir /tmp/openmontage-smoke
```

它会通过 OpenMontage 的 `video_compose`，操作是 `compose`，运行时是 `ffmpeg`。需要本机有 `ffmpeg` 和 `ffprobe`。输出在 `/tmp/openmontage-smoke/despina-render-smoke/renders/timeline.mp4`。这是时间轴通路检查，不是新闻片。

已编译的 120–180 秒规格默认不会开渲染。要看前 4 秒占位画面：

```bash
python -m lib.deepseek_director render \
  --project-id ai-news-a \
  --projects-dir projects \
  --placeholders \
  --max-seconds 4
```

占位色板整段编码（仍然不是成片，也不会自动换成真实素材）：

```bash
python -m lib.deepseek_director render \
  --project-id ai-news-a \
  --projects-dir projects \
  --placeholders \
  --full-timeline
```

配音完成且每段都没有 `overflow` 时，可以去掉 `--placeholders`。此时 `assets/video/plate-*.mp4` 必须已经换成真实画面，否则渲染会停在缺失素材。

## 装到现有目录

父机已有 `/workspace/apps/OpenMontage`，提交为 `d21f5d4eaa08422b09c7091757a05affcbabb35b`，并已有 `.venv` 与 `node_modules`。不要重新创建仓库。在该目录：

```bash
cd /workspace/apps/OpenMontage
git status
git rev-parse HEAD
git apply --check /path/to/openmontage-deepseek-despina.patch
git apply /path/to/openmontage-deepseek-despina.patch
source .venv/bin/activate
python -m pytest tests/lib/test_pcm_wav.py tests/lib/test_deepseek_director.py tests/tools/test_openrouter_tts.py -q
python -m lib.deepseek_director smoke --projects-dir /tmp/openmontage-smoke
```

`pytest` 不需要密钥，也不应访问外网。`smoke` 只使用本地 ffmpeg。

确认密钥已在该机器的环境里之后，再对两份真实证据文件执行 `compile --offline` 或 `plan`，然后按需 `dub`。不要把密钥写进补丁。

## 限制

- 没有在本环境用真实密钥验证过 DeepSeek 或 OpenRouter。接口字段以 2026-11-07 的公开文档为准。
- 占位色板不是新闻画面。纪录片流水线后续的素材检索仍要单独做。
- idea 检查点停在 `awaiting_human`。后面的阶段不会被标成完成。
- 渲染走 `video_compose` 的 `compose`，不是 Remotion / HyperFrames 成片语法。混合旁白是 24000 Hz 单声道 WAV，合成进成片后变成 AAC，采样率仍是 24000 Hz。
- 旁白长于时间槽时不会自动改时间轴，也不会切掉语音。
- 若 OpenRouter 返回的 PCM 采样率不是 24000，且 `Content-Type` 没有 `rate=`，WAV 头会按 Gemini 3.8 的 24000 Hz 契约写入。这一点未经实响应核对。
- `compile` 必须带 `--offline`。它不会在缺少 DeepSeek 时偷偷改走别的模型。
