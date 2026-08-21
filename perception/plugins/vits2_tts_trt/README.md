# VITS2 TensorRT TTS

Chinese-English VITS2 speech synthesis for Jetson Orin. It is the default
implementation of the standard `tts` plugin, not a second MCP plugin.

## Build

The Dockerfile keeps the upstream JetPack 5.1.1 default. Build for JetPack 6.1
explicitly when using the currently published TensorRT 10 model release:

```bash
docker build \
  --build-arg JP_VERSION=61 \
  -f perception/Dockerfile.jetson \
  -t phanthymotus-perception:vits2-jp61 .
```

The image installs the WeText/Kaldifst runtime. Chinese text normalization
executes checksum-verified FST files from the model release and does not
compile OpenFST or Pynini on the device. Model files are not embedded in the
image.

## Configure

`plugins.tts.engine` defaults to `vits2_trt`. The MCP tool name remains `tts`;
there is no separate `vits2_tts` tool. Set `engine: sherpa_onnx` only when an
existing Sherpa model deployment is intentionally selected.

The first `start` or `speak` call downloads the pinned ModelScope release into `model_dir`, verifies file sizes and SHA256 checksums, and loads the TensorRT engines. `info` and tool discovery do not access the network. A complete verified model directory is reused without network access.

The published release provides TensorRT 8 engines for JetPack 5.1.1 and
TensorRT 10 engines for JetPack 6.1. TensorRT plans are not portable across
incompatible TensorRT versions or GPU architectures.
