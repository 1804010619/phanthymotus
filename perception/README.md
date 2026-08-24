# Perception Stack

Modular ASR/TTS perception plugins running as an MCP HTTP server. Connects to Agent Core via MCP tool calls and exchanges audio/text over ROS2 DDS topics.

## Audio Requirements for ASR

The ASR plugin (VAD + speech recognition) has strict requirements on the audio stream it receives. Any mic driver that does not meet these requirements will produce no output.

### ROS2 Message Type

```
audio_msgs/AudioChunk
  std_msgs/Header header
  string format          # must be "audio/pcm-16k"
  uint8[] data           # raw PCM bytes (little-endian signed 16-bit)
```

### PCM Format

| Parameter | Required value |
|-----------|---------------|
| Encoding | 16-bit signed integer, little-endian (PCM_S16_LE) |
| Sample rate | **16 000 Hz** |
| Channels | **Mono (1 channel)** |
| `format` field | `"audio/pcm-16k"` |

### Chunk Size

| Parameter | Constraint |
|-----------|-----------|
| Minimum | **1 024 bytes** (512 samples, ~32 ms) |
| Recommended | 1 024 – 4 096 bytes (32 – 128 ms per chunk) |
| Maximum | No hard limit, but very large chunks increase latency |

Chunks smaller than 1 024 bytes are **silently discarded** by the VAD. This is the most common cause of "ASR receives audio but never outputs anything."

> **Why 512 samples?** The Silero VAD model requires at least one 512-sample window to compute a speech probability. WebRTC VAD requires 480-sample (30 ms) frames. Both backends use 512 samples as the minimum chunk size.

### Common Pitfalls

#### External USB mic (ALSA, 48 kHz native rate)

Most USB audio interfaces run at 48 000 Hz. After downsampling to 16 000 Hz, a 512-frame ALSA period becomes only **170 samples (340 bytes)** — below the VAD minimum.

**Fix (already applied in `phanthymotus-driver`):** Buffer resampled output until 512 samples are accumulated before publishing each `AudioChunk`.

If you are writing a custom mic driver, apply the same buffering pattern:

```python
TARGET = 1024  # bytes (512 int16 samples)
_buf = bytearray()

# Inside your capture loop, after resampling:
_buf += resampled_bytes
while len(_buf) >= TARGET:
    chunk, _buf = bytes(_buf[:TARGET]), _buf[TARGET:]
    publish(chunk)
```

#### Native G1 robot mic (UDP multicast)

Publishes raw 16 kHz PCM at 1 024 bytes per chunk. No resampling or buffering needed.

---

## VAD Tuning

The VAD parameters can be adjusted per ASR canvas card via the instance config (⚙ button):

| Parameter | Default | Notes |
|-----------|---------|-------|
| `vad_threshold` | `0.5` | Speech probability threshold (0–1). Raise to `0.7`–`0.85` in noisy environments (e.g. robot motor noise). |
| `vad_silence_ms` | `400` | Silence duration (ms) required before an utterance is considered complete. |
| `vad_pre_roll_ms` | `500` | Audio retained from *before* the VAD tripped. Recovers clipped word onsets — without it the first syllable is often missing, which costs wake-word recall. |

---

## sherpa-onnx Execution Provider

`hw_provider` (under `plugins.asr` and `plugins.tts` in `config.yaml`) selects the
ONNX Runtime provider for everything that goes through sherpa-onnx. `auto` is the
default and is resolved by `utils/onnx_provider.py`:

| Value | Behaviour |
|-------|-----------|
| `auto` | `cuda` only if the installed wheel bundles `libonnxruntime_providers_cuda.so` **and** the model's weight files are fp32. Otherwise `cpu`. |
| `cuda` | Passed through unchanged. Either a deliberate override or a misconfigured deployment — both should stay visible. |
| `cpu` | Passed through unchanged. |

### Why the fp32 condition exists

ONNX Runtime's CUDA execution provider has no kernels for the quantised ops in an
int8 model. It partitions the graph and falls back to CPU node by node, inserting
a host↔device copy at every boundary — so an int8 model runs *slower* on the GPU
than on the CPU. Measured on orin5 (Orin NX, JetPack 5.11, CUDA 11.4, 6 cores),
sherpa-onnx 1.13.6+cuda, same audio, steady-state median. Both providers were
swept across 1/2/4 `num_threads`, because the ratio moves a lot with thread count
and a single-thread-count comparison overstates the effect in both directions:

| model | dtype | CPU t=2 | CUDA t=2 | ratio | best CPU | best CUDA | ratio |
|-------|-------|---------|----------|-------|----------|-----------|-------|
| streaming paraformer `encoder.int8.onnx` (30.7 s audio) | int8 | 3383 ms | 11125 ms | **0.30x** | 3097 ms (t=4) | 10524 ms (t=4) | **0.29x** |
| X-ASR offline transducer, beam search (28.7 s audio) | int8 encoder+joiner, fp32 decoder | 2832 ms | 5707 ms | **0.50x** | 2503 ms (t=4) | 5707 ms (t=2) | **0.44x** |
| offline SenseVoice `model.int8.onnx` (28.7 s audio) | int8 | 2086 ms | 2332 ms | **0.89x** | 1404 ms (t=4) | 1797 ms (t=4) | **0.78x** |
| offline SenseVoice `model.onnx` (28.7 s audio) | **fp32** | 4905 ms | **426 ms** | **11.52x** | 2842 ms (t=4) | 426 ms (t=2) | **6.68x** |
| Matcha TTS `model-steps-3.onnx` + vocos (13.3 s audio) | fp32 | 1784 ms | 416 ms | **4.28x** | 1175 ms (t=4) | 416 ms (t=1) | **2.83x** |

`num_threads: 2` is what `config.yaml` deploys, so the `t=2` columns are the
operationally relevant ones; the best-of-each columns are there because the CPU
scales further with threads while CUDA does not (Matcha at t=4 regresses to
538 ms, X-ASR to 6238 ms). Model load also gets slower on CUDA for int8 — X-ASR
went from ~8.6 s to ~11 s.

X-ASR is the useful case for the rule as written: its bundle is *mixed* precision
(int8 encoder and joiner, fp32 decoder), `is_quantised()` treats any int8 file as
disqualifying, and the measurement confirms that is the right call.

The two SenseVoice rows are the same model in both dtypes, which is what makes
this a dtype result rather than a per-model coincidence: int8 loses on CUDA, fp32
wins by 11.5x. Mechanism confirmed two ways — fp32-CUDA is completely insensitive
to thread count (433 / 426 / 431 ms at 1/2/4) with GR3D pinned at 97%, i.e. the
graph really is on the GPU; int8-CUDA instead scales with **CPU** threads
(3314 → 2332 → 1797 ms), i.e. much of it is not.

### int8 on CUDA also changes the output

Beyond throughput, the per-node fallback is numerically noisy: requantising at
every int8↔fp32 partition boundary perturbs results. Same model, same audio,
`num_threads=2`:

- SenseVoice **int8**: CPU vs CUDA differed on **3 of 4** clips — not only
  punctuation but a real word substitution (`不然` → `主然`).
- SenseVoice **fp32**: CPU vs CUDA differed on **0 of 4** clips.
- X-ASR (int8 encoder/joiner, beam search): differed on 1 of 4, by one character
  (「嗯」vs「呃」, both hesitation particles). Beam search reorders near-tied
  hypotheses under provider-level numeric differences.
- streaming paraformer int8 with greedy search: byte-identical.

So keeping int8 models on CPU buys output stability as well as speed. Conversely,
fp32 is provider-consistent, so moving an fp32 model to the GPU does not change
what it transcribes.

### Putting ASR on the GPU

It is possible, and the code already supports it — but it needs a **model change,
not a config change**. Ship the fp32 bundle instead of the int8 one and
`hw_provider: auto` routes it to CUDA on its own. For offline SenseVoice that is
426 ms versus the 2086 ms the deployed int8-on-CPU takes, a **4.9x** end-to-end
win on the same audio.

What that costs, and what is still unknown:

- **Size.** SenseVoice fp32 `model.onnx` is 894 MB against 228 MB for int8 — on
  the COS download and on disk, per robot.
- **Accuracy is different, not obviously better.** On these four clips the two
  dtypes disagree in both directions (int8 produced `礼拜2` where fp32 gave
  `礼2`; fp32 emitted a spurious leading `，`). Four clips cannot settle it —
  switching dtype needs a labelled set.
- **Streaming is untested at fp32.** These fp32 numbers are the *offline*
  recogniser. At int8 the streaming path was 3x worse than offline relative to CPU
  (0.30x vs 0.89x), so sherpa's per-chunk decode carries real GPU overhead and a
  streaming fp32 model may not see anything like 11x. The deployed default
  (`paraformer-zh-en`) is streaming.
- **fp16 is probably the better target than fp32** and has not been measured:
  Orin's fp16 throughput is far above fp32 and ONNX Runtime's CUDA provider has
  full fp16 kernel coverage (unlike int8), so it should keep the GPU win at
  roughly half the memory and bandwidth. It needs an fp32→fp16 ONNX conversion
  and a check that sherpa-onnx feeds it correctly.

GPU contention is not an alternative explanation for any of the above: the idle
`GR3D_FREQ` baseline was 0%, and GR3D sat at 88–97% during the CUDA runs, so the
work does reach the GPU — int8 is simply slower there. The same thread-sensitivity
signature shows up on the streaming int8 model too (17861 → 11125 → 10524 ms at
1/2/4 threads on CUDA).

So `auto` keeps every int8 ASR/KWS bundle on CPU and gives fp32 models — Matcha
TTS today — the GPU. `is_quantised()` decides from the filename (`*.int8.onnx`),
the same convention `asr.py`'s `_find(prefer_int8=...)` already relies on.

One more caveat on the rule's reach: `int16` is not a middle ground worth trying.
ONNX Runtime's int16 quantisation (`QInt16`/`QUInt16`, opset 21) is newer than
int8, the CUDA provider has no kernels for it either, and the CPU side lacks the
dot-product paths that make int8 fast — it would likely lose on both providers.

**What follows `hw_provider`:** ASR (all sherpa adapters, including X-ASR), KWS,
and the `sherpa_onnx` TTS engine.

**What does not:** the VAD, in both `_vad_worker` and the `ws_asr` path, is pinned
to `provider="cpu"`. silero VAD infers one 512-sample window at a time, so a CUDA
session would pay a kernel launch plus a H2D/D2H copy per 32 ms of audio for a
model that finishes on a single core — and in `_vad_worker` it would hold a second
CUDA context in a child process. The `vits2_trt` TTS engine also ignores it: that
is a TensorRT engine and never touches ONNX Runtime.

### The jp5.11 CUDA wheel

PyPI ships CPU-only `sherpa-onnx`, so `Dockerfile.jetson` downloads a wheel built
in-house for JetPack 5.11 from COS
(`public/sherpa-onnx/sherpa_onnx-<ver>+cuda-cp38-cp38-linux_aarch64.whl`) and
falls back to the PyPI CPU wheel for every other `JP_VERSION`.

To rebuild it, build **inside a container started from the perception image** for
the target JetPack — that image already carries cmake, g++, the matching CPython
headers and CUDA, so the pybind extension lands on the right CPython ABI and
glibc. Building on the host instead is what produces an unusable wheel: the
extension is tagged `cp<major><minor>` and will not import under a different
Python.

```bash
# on the build host (must match the target JetPack: jp5.11 → L4T R35, jp6.1 → R36)
docker run -d --name sherpa-build \
  -v /path/to/k2-fsa/sherpa-onnx:/src:ro -v /path/to/outdir:/out \
  --entrypoint bash <perception-image-for-that-jp> /out/build.sh

# inside, against a *writable* copy of the tree (setup.py appends __version__ to it):
export SHERPA_ONNX_ENABLE_GPU=ON
export SHERPA_ONNX_LINUX_ARM64_GPU_ONNXRUNTIME_VERSION=1.16.0   # jp5.11 / CUDA 11.4
export SHERPA_ONNX_MAKE_ARGS="-j2"                              # Orin has 6 cores but ~4 GB free
SHERPA_ONNX_CMAKE_ARGS="-DCMAKE_BUILD_TYPE=Release \
  -DSHERPA_ONNX_ENABLE_GPU=ON \
  -DSHERPA_ONNX_LINUX_ARM64_GPU_ONNXRUNTIME_VERSION=1.16.0 \
  -DPYTHON_EXECUTABLE=/usr/bin/python3.8 \
  -DPython_EXECUTABLE=/usr/bin/python3.8" \
  python3.8 setup.py bdist_wheel
```

Notes:

- `setup.py bdist_wheel` builds its **own** cmake tree at
  `build/temp.linux-aarch64-cpython-<ver>/` with `SHERPA_ONNX_ENABLE_PYTHON=ON`. A
  tree left behind by `build-aarch64-linux-gnu.sh` has `ENABLE_PYTHON=OFF` and is
  not reusable — expect a full compile.
- To avoid re-downloading onnxruntime, drop
  `onnxruntime-linux-aarch64-gpu-<ver>.tar.bz2` in `/tmp/`;
  `cmake/onnxruntime-linux-aarch64-gpu.cmake` checks there before GitHub.
- Upload the result under the `public/` COS prefix (anonymous read, same prefix
  `utils/model_downloader.py` uses) and bump `SHERPA_GPU_WHEEL` in
  `Dockerfile.jetson`.

**TODO — jp6.1.** The jp5.11 wheel links `libcudart.so.11.0` and cannot run on
jp6.1 (CUDA 12.6). That target needs its own build with
`SHERPA_ONNX_LINUX_ARM64_GPU_ONNXRUNTIME_VERSION=1.18.1` and
`PYTHON_EXECUTABLE=python3.10`, run on a JetPack 6 host inside
`jetson-base:jp61-torch`. Until then jp6.1 stays on the CPU wheel and `auto`
resolves to `cpu`.

---

## Plugin Concurrency

**`dispatch()` is not single-threaded.** `main.py` serves MCP over
`ThreadingHTTPServer`, so every `tools/call` runs on its own thread. `start`,
`stop`, `config`, and `speak` on the *same* plugin can genuinely run at once —
the canvas does exactly this (config → start, then stop, then config → start).

This has already caused a production incident, so the rules below are not
theoretical.

### The failure mode

Any plugin that keeps per-instance state in a dict is exposed to this shape:

```python
# ❌ WRONG — check-then-act with no lock
node_key = instance_id or input_topic
if node_key not in self._nodes:          # ← two threads both pass here
    node = _ASRNode(...)
    self._executor.add_node(node)
    self._nodes[node_key] = node         # ← only the last one survives
return self._nodes[node_key].start()
```

Both threads build a node with the *same* ROS node name, both add it to the
executor, and the dict keeps only the second. The first is now an **orphan**: its
subscription, its VAD subprocess, and its transcription thread are all still
running and still publishing to the same output topic, but it is not in
`self._nodes`, so `stop` can never reach it. It survives until the process exits.

Observable symptoms: every utterance recognised and published twice, duplicate
files in `/models/vad_segments` with byte-identical content, an extra
`vad_worker` child process that `stop` does not reap, and this from rclpy:

```
Publisher already registered for provided node name. If this is due to multiple
nodes with the same name then all logs for that logger name will go out over the
existing publisher.
```

### The rules

**1. Make the dict access atomic.** One `threading.RLock` per plugin, guarding
every read-modify-write of the state dict:

```python
# ✅ CORRECT — atomic get-or-create
with self._nodes_lock:
    node = self._nodes.get(node_key)
    if node is None:
        node = _ASRNode(...)
        try:
            self._executor.add_node(node)
        except Exception:
            node.destroy_node()          # don't leak a half-registered node
            raise
        self._nodes[node_key] = node
    else:
        self._sync_cfg(node)
```

**2. Never hold that lock across `node.start()`, `node.stop()`, or a model
load.** `_ASRNode.start()` blocks for up to 15 s waiting for the first audio
chunk. If `stop` is queued behind the lock for those 15 s, it cannot set the
cancellation flag in time, `start` sails through to `running`, and you are left
with a pipeline nobody asked for. Register the node inside the lock, then release
it and call `start()` outside.

**3. Register the node *before* starting it.** That is what lets a concurrent
`stop` find it and cancel the in-flight start. Loading a model or otherwise
blocking *before* the node is in the dict means `stop` finds nothing, returns
`{"state": "idle"}`, and silently no-ops — while the start it was meant to cancel
completes anyway.

**4. `stop` signals first, locks second.** Give the node a non-blocking
`request_stop()` that sets its cancellation events, call that before taking any
lock, and only then tear down:

```python
def stop(self) -> dict:
    self.request_stop()                  # non-blocking; unblocks an in-flight start
    with self._lifecycle_lock:
        self._teardown()
        self.state = "idle"
        return {"state": "idle"}
```

**5. Guard the node object too, and treat "starting" as taken.** A per-node
`RLock` plus `if self.state in ("running", "starting")` — otherwise two threads
that resolve to the *same* node object can both enter `_start_inner()` and build
two subscriptions and two subprocesses on one node.

**6. `destroy_node()`, not just `remove_node()`.** `remove_node` detaches the
node from the executor; it does not release the rclpy node, its publishers, or
its node name. Skip it and every start/stop cycle leaks a topic endpoint, and a
later start on the same key collides with the still-registered ghost:

```python
def _dispose_node(self, node, key=""):
    node.stop()
    self._executor.remove_node(node)
    node.destroy_node()                  # ← required
```

**7. Snapshot before iterating.** `info` is a heartbeat probe called constantly.
Iterating the live dict can raise `RuntimeError: dictionary changed size during
iteration` in the middle of a start. Copy under the lock, then iterate the copy.

### Where this applies

Every MCP server in the project uses `ThreadingHTTPServer` — `perception/main.py`
and each robot driver's `main.py`. Any plugin holding a `self._nodes` /
`self._instances` / `self._streams` dict needs the treatment above.

---

## Topic Naming

| Direction | Topic pattern | Format |
|-----------|--------------|--------|
| Input (mic) | `/{namespace}/mic/audio` or `/{namespace}/ext_mic/{id}/audio` | `audio/pcm-16k` |
| Output (ASR result) | `{input_topic}/asr` | `data/json` |

ASR result JSON:
```json
{
  "text": "recognized speech text",
  "audio_start_ts": 1234567890.123,
  "audio_end_ts":   1234567891.456,
  "asr_complete_ts": 1234567891.789
}
```

## Sensitive Config Fields

Perception plugins hold real credentials (ASR/TTS API keys). Canvas configuration
gets packaged into shareable **Solutions** and uploaded to the Resource Center,
so every credential field must declare itself sensitive in its `configSchema` —
packaging blanks declared fields only, there is no field-name blocklist:

```python
"configSchema": {
    "type": "object",
    "properties": {
        "api_key": {"type": "string", "format": "password"},   # masked input + never packaged
        "app_key": {"type": "string", "x-sensitive": True},    # visible input + never packaged
        "model":   {"type": "string"},                         # packaged as-is
    },
}
```

An unmarked credential is uploaded in clear text and readable by anyone who
downloads the solution. Full spec: `phanthymotus-driver/README_dev.md`
§ "Marking sensitive fields".
