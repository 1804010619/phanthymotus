# Review rules — Perception (`phanthymotus/perception`)

Authoritative reference: **`perception/README.md`**. It is short and almost
entirely about the ASR audio contract, which is the thing PRs break.

The same contract is restated from the driver side in
`phanthymotus-driver/README.md` §Audio Requirements for ASR Compatibility. If a
PR changes the contract, both documents need updating — check whether it did.

## The audio contract

ASR consumes `audio_msgs/AudioChunk`: **PCM 16 kHz mono S16LE**, chunks of at
least 1024 bytes.

Known failure modes to check for:

- A USB microphone delivering 48 kHz. It must be resampled to 16 kHz, not passed
  through — the symptom is recognition that silently produces nonsense rather
  than an error.
- Chunks smaller than 1024 bytes. These need buffering before publish; a driver
  that publishes per-callback without accumulating will trip this.
- Any change to sample rate, channel count or sample format is a **contract
  change**, not an implementation detail. It affects every driver that feeds ASR.

## Structure

- `main.py` — MCP server entry
- `plugins/` — `asr`, `tts`, `vop`, `htmsg`, `kws`. Read a sibling plugin to
  learn the local convention before judging a new one.
- `config.yaml` — per-plugin enable/disable
- `utils/model_downloader.py` — the model manifest. **Models belong here, fetched
  from COS at runtime, not committed.** Eleven models are already listed; a new
  one should be added to this manifest in the same shape.

Perception ships `deploy/service.yml`, so it deploys the same way drivers do:
Agent Core extracts the fragment from the image and merges it into the host
compose file.

## Two Dockerfiles, different bases

`Dockerfile` builds `FROM ros-base` for CPU; `Dockerfile.jetson` builds from a
prebuilt Jetson torch image and downloads CLIP weights at build time. A change to
one usually needs the same change in the other — flag it when only one moved.

Note `Dockerfile.jetson` hardcodes its registry rather than taking an `ARG`,
unlike every other Dockerfile in the project. Worth mentioning if a PR touches
that line anyway, not worth raising on its own.
