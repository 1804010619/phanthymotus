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

The default builder stage compiles OpenFST 1.8.3 and Pynini 2.1.6 for the
selected JetPack image. Model files are not embedded in the image.

## Configure

`plugins.tts.engine` defaults to `vits2_trt`. The MCP tool name remains `tts`;
there is no separate `vits2_tts` tool. Set `engine: sherpa_onnx` only when an
existing Sherpa model deployment is intentionally selected.

The first `start` or `speak` call downloads the pinned ModelScope release into `model_dir`, verifies file sizes and SHA256 checksums, and loads the TensorRT engines. `info` and tool discovery do not access the network. A complete verified model directory is reused without network access.

The published release currently supports TensorRT 10 on Jetson Orin. The image
can be built on JetPack 5.1.1, but a JP511 engine release must be published
before JP511 can synthesize. TensorRT plans are not portable across incompatible
TensorRT versions or GPU architectures.
