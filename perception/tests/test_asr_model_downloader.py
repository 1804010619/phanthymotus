import hashlib
import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class ASRModelDownloaderTest(unittest.TestCase):
    def test_downloads_files_and_verifies_sha256(self):
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

            manifest = {
                filename: hashlib.sha256(payload).hexdigest()
                for filename, payload in payloads.items()
            }
            download_model(source.as_uri(), output_tmp, manifest)

            for filename, payload in payloads.items():
                self.assertEqual((Path(output_tmp) / filename).read_bytes(), payload)

    def test_checksum_mismatch_does_not_leave_invalid_file(self):
        from utils.asr_model_downloader import download_model

        with tempfile.TemporaryDirectory() as source_tmp, tempfile.TemporaryDirectory() as output_tmp:
            source = Path(source_tmp)
            (source / "model.int8.onnx").write_bytes(b"corrupt")

            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                download_model(
                    source.as_uri(),
                    output_tmp,
                    {"model.int8.onnx": hashlib.sha256(b"expected").hexdigest()},
                )

            self.assertFalse((Path(output_tmp) / "model.int8.onnx").exists())


if __name__ == "__main__":
    unittest.main()
