# ASR Migration onto Latest Main

## Goal

Rebuild `feat/zengzhitao` from `origin/main` commit `a2a0ce7`, preserving the
latest upstream ASR fixes while adding the benchmark's official offline
Paraformer mode without committing model binaries.

## Migration Boundary

Keep from latest `main`:

- ASR model selection and asynchronous loading state.
- Broken pipe handling in the VAD subprocess.
- VAD pre-buffering that prevents first-word truncation.
- COS-backed Silero VAD download.
- Current TTS, VOP, model downloader, KWS test, and agent-core changes.

Port from the old feature branch:

- `offline`, `online`, `streaming`, and `segmented` modes under the existing
  `asr` tool contract, with `offline` as the default.
- Official Paraformer offline recognition through
  `sherpa_onnx.OfflineRecognizer.from_paraformer`.
- KWS disabled by default for benchmark input.
- Environment overrides for MCP and WebSocket ports.
- Focused contract tests and the five sub-1 MB WAV fixtures.
- Build-time download of the official Paraformer model from the stable internal
  model server.

Do not port the old branch's TTS, VOP, agent-core, or generic model-downloader
versions. Do not register a separate `asr_offline` tool because the evaluator
calls the default `asr` plugin.

## Model Distribution

`perception/Dockerfile.jetson` downloads `config.json`, `model.int8.onnx`, and
`tokens.txt` from:

```text
http://172.28.4.81:34567/zengzhitao/embodied-ai/official_paraformer/
```

The build validates known SHA256 values and fails on a network or checksum
error. Silero VAD remains a runtime COS download managed by latest `main`.

## History Reconstruction

Implementation occurs on `feat/zengzhitao-clean`, created directly from the
latest `origin/main`. Only reviewed source, tests, docs, and small audio fixtures
are committed. After verification and explicit approval, that clean commit is
force-pushed with lease to `origin/feat/zengzhitao`. `main` is never pushed or
rewritten.
