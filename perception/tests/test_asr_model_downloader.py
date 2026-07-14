import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class ASRModelDownloaderTest(unittest.TestCase):
    def test_downloads_files_without_checksums(self):
        from utils.asr_model_downloader import download_model

        with tempfile.TemporaryDirectory() as source_tmp, tempfile.TemporaryDirectory() as output_tmp:
            source = Path(source_tmp)
            payloads = {
                "config.json": b'{"model": "paraformer"}',
                "model.int8.onnx": b"test-onnx-payload",
                "tokens.txt": b"0 <blank>\n",
            }
            for filename, payload in payloads.items():
                (source / filename).write_bytes(payload)

            download_model(source.as_uri(), output_tmp, tuple(payloads))

            for filename, payload in payloads.items():
                self.assertEqual((Path(output_tmp) / filename).read_bytes(), payload)

    def test_download_failure_does_not_leave_partial_file(self):
        from utils.asr_model_downloader import download_model

        with tempfile.TemporaryDirectory() as output_tmp:
            def fail_after_partial_download(_url, temporary_path):
                Path(temporary_path).write_bytes(b"partial")
                raise OSError("download interrupted")

            with mock.patch(
                "utils.asr_model_downloader.urlretrieve",
                side_effect=fail_after_partial_download,
            ):
                with self.assertRaisesRegex(OSError, "download interrupted"):
                    download_model(
                        "https://models.example.test",
                        output_tmp,
                        ("model.int8.onnx",),
                    )

            self.assertFalse((Path(output_tmp) / "model.int8.onnx").exists())
            self.assertEqual(list(Path(output_tmp).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
