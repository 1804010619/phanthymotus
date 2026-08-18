# VITS2 TensorRT TTS

Optional Chinese-English VITS2 speech synthesis for JetPack 6.1. The plugin is disabled by default and does not change the existing `tts` plugin.

## Build

Build on an aarch64 JetPack 6.1 host:

```bash
docker build \
  --build-arg JP_VERSION=61 \
  --build-arg ENABLE_VITS2_TRT=1 \
  -f perception/Dockerfile.jetson \
  -t phanthymotus-perception:vits2-jp61 .
```

The optional builder stage compiles OpenFST 1.8.3 and Pynini 2.1.6. Model files are not embedded in the image.

## Configure

Enable `plugins.vits2_tts` in `perception/config.yaml`. The MCP tool name is `vits2_tts`; the original `tts` tool remains available when its plugin is enabled.

The first `start` or `speak` call downloads the pinned ModelScope release into `model_dir`, verifies file sizes and SHA256 checksums, and loads the TensorRT engines. `info` and tool discovery do not access the network. A complete verified model directory is reused without network access.

The initial release supports TensorRT 10 on Jetson Orin. TensorRT plans are not portable across incompatible TensorRT versions or GPU architectures.
