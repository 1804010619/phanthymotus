# ASR Main Migration Implementation Plan

> **For AI agent workers:** Required sub-skill: use superpowers:executing-plans to implement this plan task by task. Track progress with the checkboxes below.

**Goal:** Rebuild the ASR benchmark feature on the latest `main`, keeping upstream fixes and downloading the official offline Paraformer model during the Jetson image build.

**Architecture:** The existing `plugins/asr.py` remains the public `asr` tool and owns VAD/KWS/ROS behavior. A focused `plugins/asr_offline.py` supplies the official Paraformer adapter. A standard-library downloader installs and verifies model files at Docker build time.

**Technical stack:** Python 3 standard library, `unittest`, sherpa-onnx, ROS2, Dockerfile.jetson, Git worktrees.

---

### Task 1: Add the verified model downloader

**Files:**
- Create: `perception/tests/test_asr_model_downloader.py`
- Create: `perception/utils/asr_model_downloader.py`

- [ ] **Step 1: Write failing download and checksum tests**

Test a `download_model(base_url, output_dir, manifest)` function with a
temporary `file://` source. Assert valid files are copied and a bad digest
raises `ValueError` without leaving the invalid destination file.

- [ ] **Step 2: Verify the tests fail because the module is missing**

Run: `python3 -m unittest perception.tests.test_asr_model_downloader -v`

Expected: `ImportError` for `utils.asr_model_downloader`.

- [ ] **Step 3: Implement the downloader**

Implement:

```python
def download_model(base_url: str, output_dir: str, manifest: dict[str, str]) -> None:
    for filename, expected_sha256 in manifest.items():
        destination = Path(output_dir) / filename
        urlretrieve(f"{base_url.rstrip('/')}/{filename}", temporary_path)
        if sha256_file(temporary_path) != expected_sha256:
            raise ValueError(f"SHA256 mismatch for {filename}")
        temporary_path.replace(destination)
```

Expose a CLI with the internal base URL as the default and the three verified
digests from the design specification.

- [ ] **Step 4: Run downloader tests**

Run: `python3 -m unittest perception.tests.test_asr_model_downloader -v`

Expected: two tests pass.

- [ ] **Step 5: Commit**

```bash
git add perception/tests/test_asr_model_downloader.py perception/utils/asr_model_downloader.py
git commit -m "feat: download verified offline asr model"
```

### Task 2: Package models without Git binaries

**Files:**
- Modify: `.gitignore`
- Modify: `perception/Dockerfile.jetson`
- Modify: `perception/config.yaml`
- Create: `perception/tests/test_asr_packaging.py`

- [ ] **Step 1: Write failing packaging tests**

Read the Dockerfile and config as text. Assert the Dockerfile calls
`asr_model_downloader.py`, contains the internal base URL, and has no
`COPY perception/models/`. Assert config enables ASR, defaults to
`mode: offline`, points to `/models/sherpa-onnx/asr-offline`, and disables KWS.

- [ ] **Step 2: Verify packaging tests fail**

Run: `python3 -m unittest perception.tests.test_asr_packaging -v`

Expected: failures for missing downloader invocation and offline config.

- [ ] **Step 3: Modify Dockerfile and config**

Add:

```dockerfile
ARG ASR_OFFLINE_MODEL_BASE_URL=http://172.28.4.81:34567/zengzhitao/embodied-ai/official_paraformer
COPY perception/utils/asr_model_downloader.py /tmp/asr_model_downloader.py
RUN python3 /tmp/asr_model_downloader.py \
    --base-url "${ASR_OFFLINE_MODEL_BASE_URL}" \
    --output-dir /models/sherpa-onnx/asr-offline
```

Configure the existing `asr` plugin for offline mode and disable unrelated
plugins in the benchmark default config. Add `perception/models/` and `*.onnx`
to `.gitignore`.

- [ ] **Step 4: Run packaging and downloader tests**

Run: `python3 -m unittest perception.tests.test_asr_packaging perception.tests.test_asr_model_downloader -v`

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add .gitignore perception/Dockerfile.jetson perception/config.yaml perception/tests/test_asr_packaging.py
git commit -m "build: fetch offline asr model for jetson"
```

### Task 3: Add the official offline Paraformer adapter

**Files:**
- Create: `perception/plugins/asr_offline.py`
- Create: `perception/tests/test_asr_offline_sherpa_compat.py`

- [ ] **Step 1: Port the minimal sherpa compatibility test**

Stub a sherpa module exposing only
`OfflineRecognizer.from_paraformer(**kwargs)`. Assert the adapter passes the
resolved ONNX path, token path, CPU provider, and thread count.

- [ ] **Step 2: Verify the compatibility test fails**

Run: `python3 -m unittest perception.tests.test_asr_offline_sherpa_compat -v`

Expected: import failure because `plugins.asr_offline` is absent.

- [ ] **Step 3: Implement a focused adapter**

Implement `OfflineASRAdapter` with `get_instance`, WAV decoding, and
`OfflineRecognizer.from_paraformer`. Read `config.json` when present, resolve
`model.int8.onnx` and `tokens.txt`, and support older sherpa wheels that reject
optional ITN arguments.

- [ ] **Step 4: Run compatibility tests**

Run: `python3 -m unittest perception.tests.test_asr_offline_sherpa_compat -v`

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add perception/plugins/asr_offline.py perception/tests/test_asr_offline_sherpa_compat.py
git commit -m "feat: add official paraformer offline adapter"
```

### Task 4: Integrate four modes into latest main ASR

**Files:**
- Modify: `perception/plugins/asr.py`
- Create: `perception/tests/test_asr_contract.py`

- [ ] **Step 1: Write contract tests before modifying ASR**

Using ROS stubs, assert:

```python
self.assertEqual(asr._resolve_asr_mode({}), "offline")
for mode in ("offline", "online", "streaming", "segmented"):
    self.assertEqual(asr._resolve_asr_mode({"mode": mode}), mode)
self.assertEqual(asr._asr_output_topic("/robot/mic/audio"), "/robot/mic/audio/asr")
self.assertFalse(asr._is_kws_enabled({}))
```

Also assert the public tool remains named `asr` with the existing input and
output formats.

- [ ] **Step 2: Verify contract tests fail on latest main**

Run: `python3 -m unittest perception.tests.test_asr_contract -v`

Expected: failures for missing mode helpers.

- [ ] **Step 3: Add mode routing without replacing latest main fixes**

Add mode fields to `TOOLS`, route `offline` and `segmented` to
`OfflineASRAdapter`, and retain latest main's adapters for `online` and
`streaming`. Preserve asynchronous model loading, VAD subprocess death
handling, and pre-buffer logic. Make KWS active only when both `enabled: true`
and `trigger_mode: kws` are explicit.

- [ ] **Step 4: Run ASR tests**

Run:

```bash
python3 -m unittest \
  perception.tests.test_asr_contract \
  perception.tests.test_asr_offline_sherpa_compat -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add perception/plugins/asr.py perception/tests/test_asr_contract.py
git commit -m "feat: add offline mode to default asr plugin"
```

### Task 5: Add benchmark ports and small test assets

**Files:**
- Modify: `perception/main.py`
- Create: `perception/tests/test_benchmark_contract.py`
- Create: `perception/tests/fixtures/audio/demo_01_short.wav`
- Create: `perception/tests/fixtures/audio/demo_02_short.wav`
- Create: `perception/tests/fixtures/audio/demo_03_normal.wav`
- Create: `perception/tests/fixtures/audio/demo_04_normal.wav`
- Create: `perception/tests/fixtures/audio/demo_05_normal.wav`
- Create: `perception/tools/test_asr_models.py`
- Create: `perception/docs/asr_cer_results.md`

- [ ] **Step 1: Write a failing benchmark contract test**

Read `main.py` and assert the exact `MCP_PORT` and `WS_PORT` environment
fallback expressions are present. Assert every committed WAV fixture is less
than 1 MB.

- [ ] **Step 2: Verify the benchmark contract test fails**

Run: `python3 -m unittest perception.tests.test_benchmark_contract -v`

Expected: failures for missing environment port expressions and fixtures.

- [ ] **Step 3: Apply the port change and port only small ASR assets**

Use:

```python
mcp_port = int(os.environ.get("MCP_PORT") or cfg.get("mcp_port", 15720))
ws_port = int(os.environ.get("WS_PORT") or cfg.get("ws_port", 15721))
```

Restore the five WAV fixtures, ASR model comparison tool, and CER notes from
the old feature branch. Do not restore TTS or plugin dispatch files.

- [ ] **Step 4: Run benchmark contract tests**

Run: `python3 -m unittest perception.tests.test_benchmark_contract -v`

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add perception/main.py perception/tests/test_benchmark_contract.py perception/tests/fixtures/audio perception/tools/test_asr_models.py perception/docs/asr_cer_results.md
git commit -m "fix: align asr benchmark runtime contract"
```

### Task 6: Verify the rebuilt branch

**Files:**
- Verify all files changed since `origin/main`.

- [ ] **Step 1: Run the complete local test suite**

Run: `python3 -m unittest discover -s perception/tests -p 'test_*.py' -v`

Expected: all tests pass.

- [ ] **Step 2: Compile Python sources**

Run: `python3 -m compileall -q perception`

Expected: exit code 0.

- [ ] **Step 3: Check patch quality and forbidden Docker copies**

Run:

```bash
git diff --check origin/main...HEAD
rg -n "COPY perception/models" perception/Dockerfile.jetson
```

Expected: diff check exits 0 and `rg` returns no matches.

- [ ] **Step 4: Check file sizes and branch history**

List every feature-added tracked file larger than 1 MB and every reachable blob
larger than 1 MB. Expected: no new feature file or model blob is reported.

- [ ] **Step 5: Record Jetson-only verification**

On the Jetson host, build `perception/Dockerfile.jetson`, confirm the three
official Paraformer files download from the internal URL, start the container,
and transcribe at least one WAV fixture. This cannot be claimed locally because
the internal URL and aarch64 base image are unavailable here.

- [ ] **Step 6: Request approval for remote replacement**

Show the new commit ID, comparison against `origin/main`, test results, and
large-file scan. Only after explicit approval run:

```bash
git push --force-with-lease origin feat/zengzhitao-clean:feat/zengzhitao
```
