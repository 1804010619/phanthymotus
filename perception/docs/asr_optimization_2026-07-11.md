# ASR Runtime Optimization Record

Date: 2026-07-11
Branch: `feat/zengzhitao`

## Scope

This change improves the ASR runtime while preserving the existing ROS topic
contract and offline Paraformer model. The Jetson model download URL,
downloader, and Docker build steps are explicitly out of scope and remain
unchanged.

## Design

The runtime keeps the existing pipeline:

```text
AudioChunk -> VAD -> complete utterance -> ASR adapter -> JSON result
```

The implementation changes are:

1. Treat `offline` and `streaming` as the two canonical recognizer modes.
   Keep `segmented` as an alias of `offline` and `online` as an alias of
   `streaming` for compatibility.
2. Move PCM conversion and live VAD sessions behind a shared runtime module.
   ROS and WebSocket paths use the same VAD interface and honor the configured
   backend. `silero` remains an alias for the sherpa-onnx Silero model.
3. Build VAD pre-roll from timestamped PCM history before the reported speech
   start. Do not prepend the most recently received tail frames after a segment
   has already completed.
4. Serialize access to each cached sherpa-onnx offline recognizer. Cache
   creation and inference use separate locks.
5. Load the initial ASR model asynchronously, with explicit `loading` and
   `error` states. Model changes continue to stop active nodes before loading.
6. Report received chunks, dropped chunks, completed utterances, transcription
   errors, queue depth, and last activity timestamps in node status.
7. Use NumPy for PCM conversion when available, with a standard-library
   fallback for minimal environments.

## Compatibility

- The MCP tool name and ROS input/output formats do not change.
- Existing `online` and `segmented` values remain accepted as aliases.
- Existing `sherpa_onnx`, `silero`, `webrtc`, and `energy` VAD names remain
  accepted.
- The default remains offline Paraformer with sherpa-onnx VAD.
- Model download and packaging behavior do not change.

## Verification Plan

- Add behavior tests for canonical mode resolution.
- Add behavior tests for energy VAD pre-roll and invalid backends.
- Add a concurrency test proving one cached recognizer is not decoded by two
  threads simultaneously.
- Add a plugin test proving constructor startup does not block on model load.
- Add a node callback test proving queue drops are counted and exposed.
- Run all tests under `perception/tests`.

## Implementation Result

Implemented changes:

- Added `plugins/asr_runtime.py` as the shared PCM and VAD runtime used by ROS
  and WebSocket ASR.
- Replaced the completed-segment tail pre-buffer with timestamped PCM history.
  sherpa segments now use the original PCM covering the configured pre-roll
  and segment sample interval.
- Preserved per-chunk timestamps across gaps, including gaps caused by input
  queue drops.
- Added energy, WebRTC, and sherpa-onnx backend selection with explicit
  validation. `silero` maps to the sherpa-onnx Silero implementation.
- Canonicalized recognizer modes to `offline` and `streaming` while accepting
  `segmented` and `online` as compatibility aliases.
- Serialized offline Paraformer, streaming Paraformer, and Zipformer decode
  calls because plugin instances can be shared by multiple ASR nodes.
- Changed initial plugin model loading to a generation-protected background
  load, so bundle startup is not blocked by model initialization.
- Added a lifecycle lock around plugin dispatch operations to prevent duplicate
  nodes and start/config races.
- Added node metrics for received and dropped chunks, completed utterances,
  errors, queue depth, and last activity timestamps. Per-instance `info`
  responses expose these metrics.
- Propagated VAD backend, threshold, silence, pre-roll, and model directory
  settings consistently to both ROS and WebSocket paths.
- Added NumPy PCM conversion with a standard-library fallback.

Verification performed:

- TDD red/green cycles were observed for mode aliases, asynchronous loading,
  queue metrics, recognizer concurrency, VAD pre-roll, partial-frame flush,
  timestamp gaps, VAD model directory propagation, and concurrent starts.
- `python -m unittest discover -s perception/tests -v`: 29 tests passed.
- `python -m compileall -q` passed for the changed Python runtime files.
- Python 3.8 grammar parsing passed for the changed Python runtime files.
- `git diff --check` reported no whitespace errors.
- `perception/Dockerfile.jetson` and
  `perception/utils/asr_model_downloader.py` have no changes.

Environment limitations:

- A Docker image build was not run because Docker is not installed on this
  workstation.
- Tests use deterministic fake sherpa-onnx objects for VAD and concurrency
  behavior. Real Jetson model inference and ROS audio integration still require
  validation in the target image.
