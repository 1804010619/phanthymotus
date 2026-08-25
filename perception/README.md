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

## sherpa-onnx Device Selection

`device: cpu | gpu` (under `plugins.asr` and `plugins.tts` in `config.yaml`, and on
the dashboard's config form) selects where sherpa-onnx runs. It defaults to `cpu`.

**The model follows the device, not the other way round.** `ASR_MODELS` in
`plugins/asr.py` maps each (model, device) pair to the weights that pair loads,
because the best weights differ per device: quantised weights are right on the CPU
and wrong on the GPU. `device: gpu` therefore downloads a different bundle, not
just a different provider string.

| `asr_model` | `device: cpu` | `device: gpu` | gpu speed-up |
|-------------|---------------|---------------|--------------|
| `sensevoice-small` (default) | int8, 228 MB | **fp16, 448 MB** | **3.4x** per utterance |
| `paraformer-zh-en` (streaming) | int8, 226 MB | **fp32, 825 MB** | **1.77x** |
| `x-asr-zh-en` | int8 + fp32 | — not offered | 0.80x, i.e. slower |
| `paraformer-offline` | int8 | — not offered | unmeasured |
| `zipformer-en` | int8 | — not offered | unmeasured |

**gpu costs about 2 GB of RAM, and ~1.4 GB of that is unreturnable.** Measured with
only ASR resident: the cpu adapter adds 542 MB and drops back to 129 MB when
released; the gpu adapter adds 1968 MB and still holds 1516 MB after release,
because a process that has touched CUDA does not give its context and memory pool
back. On a 7.4 GB Orin already running vop (YOLO), OCR (TensorRT) and TTS, turning
on gpu ASR was enough to exhaust memory: perception was restarted in a loop, Agent
Core could not reach port 15720, and the dashboard rolled the project back and the
cards vanished. Budget for it before enabling.

TTS is simpler: Matcha is fp32 only, so both devices load the same files and
`device` only picks the provider (gpu measured ~4.3x). The `vits2_trt` engine
ignores `device` entirely — it is a TensorRT engine and never touches ONNX Runtime.

`device: gpu` also needs the CUDA sherpa-onnx wheel, which only the jp5.11 image
installs (see § The jp5.11 CUDA wheel). On jp6.1 and x86 dev hosts it falls back to
`cpu` with a warning rather than failing to start.

### Latency per utterance, which is what an operator feels

Same box, SenseVoice, real VAD segments (1–4 s of speech, the length a wake-word
turn actually produces), with the rest of perception left running:

| | cpu (int8) | gpu (fp16) | |
|---|---|---|---|
| first call, no warmup | 165 ms | **1659 ms** | gpu 10x slower |
| first call, warmed | 162 ms | **82 ms** | gpu 2.0x faster |
| sustained p50 (40 calls) | 195 ms | **58 ms** | gpu 3.4x faster |
| sustained p90 | 319 ms | 70 ms | |
| sustained p99 | 326 ms | **78 ms** | gpu's worst case beats cpu's median |
| after 60 s idle | 159 ms | 73 ms | 1.3x its own p50 — no downclock cliff |
| adapter build | 4.6 s | 5.3 s | |

Three things worth knowing:

- **Warmup does its job.** 1659 ms cold versus 82 ms warmed, so
  `_warmup_adapter()` (a second of silence right after building) is what keeps that
  cost out of the operator's first utterance. `warmup: false` opts out.
- **No drift and no idle cliff.** Across 40 back-to-back calls the first half and
  second half both sat at a 57.7 ms p50, and a call after 60 s of silence took
  73 ms — an idle Orin GPU downclocking was a plausible worry and did not
  materialise.
- **The GPU is also steadier.** cpu p99 is 1.7x its own p50 (326 vs 195 ms) because
  vop and OCR compete for cores; gpu p99 is 1.35x (78 vs 58 ms). The GPU's worst
  case is faster than the CPU's median.

The batch figures in § Measurements are larger (up to 23x) because they decode the
model's own `test_wavs`, which are 7 s each — longer audio gives the GPU more to
amortise its fixed cost against. Those numbers are the right way to compare dtypes
with each other; the table above is the latency a user experiences.

### Switching engine or device blocks for the bounded part

`action=config` on the `tts` tool waits up to `ENGINE_SWITCH_WAIT_S` (20 s) for the
new engine before answering `loading`. Only some of a build is open-ended — a cold
model download — while constructing the session afterwards took ~2 s on cpu and ~5 s
on gpu. Reporting `loading` for that made the dashboard send its `start` into a
facade with no engine, and the start was dropped: the engine then came up idle, and
Agent Core's loading watcher (`api/config.py` `_settle_loading_item`) reports "启动已
取消" the moment it sees `idle`.

Waiting is safe on that path — `mcp_call_tool` sets no client timeout and the
watcher polls for up to 900 s. (The 60 s often quoted in these plugins belongs to
the *LLM* tool path in `agent-core/src/mcp_client.py`, which does not send `config`.)
A build that outlives the bound still goes async, and `TTSPlugin` records any
`start` that arrives mid-build and replays it once the engine is resident, so the
card reaches `running` instead of being cancelled. A `stop` cancels a pending
replay. The bound is a time limit rather than a "does it need to download?" check
because the download is not the only slow phase: the gpu paraformer encoder is
636 MB and reading it cold takes seconds by itself.

### Measurements

orin5 (Orin NX, JetPack 5.11, CUDA 11.4, 6 cores), sherpa-onnx 1.13.6+cuda,
steady-state median, **on an idle box** — `embodied-perception` and `agent-core`
stopped, idle `GR3D_FREQ` verified 0% before each run. Both providers swept across
1/2/4 `num_threads`, because the ratio moves a lot with thread count:

| model | dtype | CPU t=2 | CUDA t=2 | ratio |
|-------|-------|---------|----------|-------|
| streaming paraformer (30.7 s audio) | int8 | 3295 ms | 8394 ms | **0.39x** |
| streaming paraformer | fp32 | 8702 ms | **1859 ms** | **4.68x** |
| streaming paraformer | fp16 | 42890 ms | 2077 ms | 20.65x ⚠️ broken, see below |
| X-ASR, beam search (28.7 s audio) | int8 | 2645 ms | 3294 ms | **0.80x** |
| offline SenseVoice (28.7 s audio) | int8 | 1996 ms | 2753 ms | **0.73x** |
| offline SenseVoice | fp32 | 4792 ms | 416 ms | **11.52x** |
| offline SenseVoice | fp16 | 8117 ms | **344 ms** | **23.60x** |
| Matcha TTS (13.3 s audio) | fp32 | 1784 ms | 416 ms | **4.28x** |

`num_threads: 2` is what `config.yaml` deploys. Threads matter on the CPU (4
threads buys roughly 1.2–1.5x) and not at all on the GPU for non-quantised weights.

### Why int8 loses on the GPU

ONNX Runtime's CUDA execution provider has no kernels for the quantised ops in an
int8 model. It partitions the graph and falls back to CPU node by node, inserting a
host↔device copy at every boundary.

Two independent confirmations, not just the int8/fp32 correlation:

- fp32 and fp16 on CUDA are **completely insensitive to CPU thread count**
  (417/416/415 and 343/344/346 ms at 1/2/4) with GR3D pinned at 95–97% — the graph
  really is on the GPU. int8 on CUDA instead **scales with CPU threads**
  (3793 → 2753 → 1905 ms for SenseVoice, 17861 → 11125 → 10524 for streaming
  paraformer), which only makes sense if much of it is executing on the CPU.
- Requantising at every partition boundary perturbs the output. Same model, same
  audio, `num_threads=2`, CPU vs CUDA: SenseVoice int8 differed on **3 of 4** clips,
  including `不然` → `主然` — a real word error, not punctuation. SenseVoice fp32 and
  fp16 differed on **0 of 4**.

GPU contention is not an alternative explanation: the idle `GR3D_FREQ` baseline was
0% for every run above. That is not a hypothetical — an earlier X-ASR run taken
while another container was building TensorRT engines read 0.50x instead of 0.80x.

`int16` is not a middle ground worth trying: ONNX Runtime's int16 quantisation
(`QInt16`/`QUInt16`, opset 21) is newer than int8, the CUDA provider has no kernels
for it either, and the CPU side lacks the dot-product paths that make int8 fast.

### Admitting a new (model, device) pair

The `gpu` column above is short because each entry had to earn its place. **A pair
is only added to `ASR_MODELS` after decoding real audio on the target device and
reading the text.**

That rule exists because of one result. Streaming paraformer fp16 on CUDA:

- created an ONNX Runtime session without complaint,
- ran in 2077 ms, 20x faster than the same file on CPU,
- produced byte-identical output across all three thread counts,
- and emitted nothing but `</s> </s> </s> …`.

The same fp16 file on CPU transcribed correctly, so the conversion was fine and the
CUDA+fp16+streaming *combination* is not. Session creation, speed, and
self-consistency were all green. Only reading the text caught it. (fp16 is also
slower than fp32 for that model, so there was nothing to gain by debugging it —
`paraformer-zh-en`'s gpu entry is fp32.)

Checklist:

1. Benchmark both devices across 1/2/4 threads on an idle box, and verify the idle
   `GR3D_FREQ` is 0% first.
2. Decode real audio on the target device and read the transcripts. Compare against
   the same weights on the other provider, and against the cpu entry.
3. Add the entry to `ASR_MODELS` with its `dtype`, and add the pinned bundle to
   `SHERPA_GPU_BUNDLES` in `utils/model_downloader.py`.
4. Add the model to the `device` field's `x-show-when` list in the configSchema.
   `tests/test_asr_device_registry.py` asserts that list matches the registry, that
   no gpu entry is int8, and that no cpu entry is fp16.

### Producing fp16 weights

`tools/convert_onnx_fp16.py` converts an fp32 sherpa-onnx model. Two flags are
load-bearing:

- `keep_io_types=True` — sherpa hands the session fp32 features, so only the graph
  interior may be fp16.
- shape inference must stay **on**. Ops that cannot take fp16 (`Range` above all)
  are already in onnxconverter-common's default `op_block_list` and get fenced with
  Cast nodes, but placing those Casts needs shape inference. Disabling it produces
  a file that saves fine and then fails at session creation with
  `Type 'tensor(float16)' of input parameter (…) of operator (Range) … is invalid`.

fp16 is a **GPU-only** choice: ONNX Runtime has no fp16 CPU kernels and casts
everything, which is why the streaming fp16 CPU row above is 42890 ms against
int8's 3295 ms. `provider_for_device()` logs an error if a cpu entry ever points at
fp16 weights, and the registry test rejects it.

### What does not follow `device`

- **KWS** is pinned to CPU. Its zipformer bundle ships int8 only, and int8 on CUDA
  lost on every model measured, so there is nothing to gain.
- **VAD** is pinned to CPU in both code paths (`_vad_worker` and
  `_vad_segment_sync`). silero infers one 512-sample window at a time — too little
  work to amortise a kernel launch plus two copies per 32 ms of audio — and in
  `_vad_worker` it would hold a second CUDA context in a child process.
- **`vits2_trt` TTS**, as above: TensorRT, not ONNX Runtime.

### GPU bundle distribution

`device: gpu` weights come from `SHERPA_GPU_BUNDLES` in `utils/model_downloader.py`
and are fetched with `ensure_verified_bundle`, which pins every file's size and
SHA256. The cpu bundles use `ensure_model`, whose only integrity check is "does
`check_file` exist in the archive" — acceptable for a 230 MB archive, not for a
780 MB one, where a truncated transfer would pass and then fail confusingly at
session creation.

Provenance: the fp32 weights come from `pengzhendong`'s ModelScope mirrors of the
k2-fsa model zoo, accepted only after that mirror's int8 files were confirmed
**byte-identical** (SHA256) to the copies we already deploy from COS. The fp16 files
are converted from those with `tools/convert_onnx_fp16.py`.

---

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
`jetson-base:jp61-torch`. Until then jp6.1 stays on the CPU wheel and
`device: gpu` falls back to `cpu` there.

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
