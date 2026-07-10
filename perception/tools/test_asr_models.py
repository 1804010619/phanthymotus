#!/usr/bin/env python3
"""ASR model CER benchmark on Chinese test data (optimized)."""

import argparse
import heapq
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import sherpa_onnx

try:
    from rapidfuzz import distance
except ImportError:
    distance = None


def compute_cer_numpy(ref, hyp):
    """Levenshtein-based CER using rapidfuzz for speed."""
    if not ref and not hyp:
        return 0.0
    if not ref:
        return 1.0
    if not hyp:
        return 1.0

    if distance is not None:
        dist = distance.Levenshtein.distance(ref, hyp)
    else:
        dist = _levenshtein(ref, hyp)
    return dist / len(ref)


def _levenshtein(ref: str, hyp: str) -> int:
    prev = list(range(len(hyp) + 1))
    for i, rch in enumerate(ref, 1):
        cur = [i]
        for j, hch in enumerate(hyp, 1):
            cur.append(min(
                prev[j] + 1,
                cur[j - 1] + 1,
                prev[j - 1] + (rch != hch),
            ))
        prev = cur
    return prev[-1]


def compute_cer(ref, hyp):
    """Character Error Rate."""
    return compute_cer_numpy(ref, hyp)


def load_wav_float32(path, target_sr=16000):
    """Load WAV as float32 numpy array normalized to [-1, 1], resampled to target_sr.
    Handles PCM16, PCM8, and IEEE float (format 3) WAV files."""
    with open(path, 'rb') as f:
        f.read(12)  # RIFF header

        sample_rate = 16000
        bit_depth = 16
        format_code = 1  # default: PCM
        data_start = 0
        data_size = 0

        offset = 12
        while True:
            header = f.read(8)
            if len(header) < 8:
                break
            chunk_id = header[:4]
            chunk_size = int.from_bytes(header[4:8], 'little')

            if chunk_id == b'fmt ':
                format_code = int.from_bytes(f.read(2), 'little')
                f.read(2)   # channels
                sample_rate = int.from_bytes(f.read(4), 'little')
                f.read(4)   # byte rate
                f.read(2)   # block align
                bit_depth = int.from_bytes(f.read(2), 'little')
                # Skip extra fmt bytes
                if chunk_size > 16:
                    f.read(chunk_size - 16)
            elif chunk_id == b'data':
                data_start = f.tell()
                data_size = chunk_size
                break
            else:
                # Skip any other chunk (fact, info, etc.)
                f.read(chunk_size + (chunk_size % 2))

            offset = f.tell()

        # Validate: data chunk must exist and use supported format
        if data_size == 0:
            raise ValueError(f"No data chunk found in {path}")
        if format_code not in (1, 3):
            raise ValueError(f"Unsupported WAV format_code={format_code} in {path} (only PCM=1 and IEEE float=3 supported)")

        f.seek(data_start)
        raw_data = f.read(data_size)

        if bit_depth == 32:
            samples = np.frombuffer(raw_data, dtype=np.float32)
        elif bit_depth == 16:
            samples = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32)
            samples = samples / 32768.0
        elif bit_depth == 8:
            samples = np.frombuffer(raw_data, dtype=np.uint8).astype(np.float32)
            samples = (samples - 128.0) / 128.0
        else:
            samples = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32)
            samples = samples / 32768.0

    if sample_rate != target_sr:
        ratio = target_sr / sample_rate
        new_len = int(len(samples) * ratio)
        indices = np.linspace(0, len(samples) - 1, new_len)
        samples = np.interp(indices, np.arange(len(samples)), samples)
        sample_rate = target_sr

    return samples, sample_rate


def load_manifest(manifest_path, max_samples=None):
    """Load manifest and return list of (audio_path, reference_text).
    Filters for Chinese-heavy text (>80% CJK characters)."""
    items = []
    base_dir = os.path.dirname(manifest_path)
    with open(manifest_path, encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"  Warning: skipping malformed JSON at line {line_num}: {e}")
                continue
            audio_rel = obj.get("audio", "")
            text = obj.get("text", "").strip()
            if not audio_rel or not text:
                continue
            # Filter: >80% Chinese/CJK characters
            zh_count = len(re.findall(r'[\u4e00-\u9fff]', text))
            total = len(text)
            if total == 0 or zh_count / total <= 0.8:
                continue
            if os.path.isabs(audio_rel):
                full_path = audio_rel
            else:
                full_path = os.path.join(base_dir, audio_rel)
            if os.path.exists(full_path):
                items.append((full_path, text))
            if max_samples and len(items) >= max_samples:
                break
    return items


def load_fleurs_tsv(tsv_path, max_samples=None):
    """Load Fleurs TSV: col1=domain, col2=wav_name, col3=text_zh, ...
    Returns list of (full_path, reference_text)."""
    items = []
    audio_base = os.path.join(os.path.dirname(tsv_path), "audio", "test")
    with open(tsv_path, encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            parts = line.strip().split('\t')
            if len(parts) < 3:
                print(f"  Warning: skipping malformed TSV at line {line_num}: expected 3+ columns")
                continue
            wav_name = parts[1]
            ref_text = parts[2].strip()
            if not ref_text:
                continue
            # Filter: >80% Chinese/CJK
            zh_count = len(re.findall(r'[\u4e00-\u9fff]', ref_text))
            total = len(ref_text)
            if total == 0 or zh_count / total <= 0.8:
                continue
            full_path = os.path.join(audio_base, wav_name)
            if os.path.exists(full_path):
                items.append((full_path, ref_text))
            if max_samples and len(items) >= max_samples:
                break
    return items


def test_model(model_name, model_dir, items, num_threads=2):
    """Test a single ASR model and return CER results."""
    print(f"\n{'='*60}")
    print(f"Testing: {model_name}")
    print(f"{'='*60}")

    recognizer = None

    try:
        if "paraformer" in model_dir:
            recognizer = sherpa_onnx.OfflineRecognizer.from_paraformer(
                paraformer=os.path.join(model_dir, "model.int8.onnx"),
                tokens=os.path.join(model_dir, "tokens.txt"),
                num_threads=num_threads,
                sample_rate=16000,
                decoding_method="greedy_search",
                debug=False,
                provider="cpu",
            )
        elif "zipformer" in model_dir or ("ctc" in model_dir and "nemo" not in model_dir):
            recognizer = sherpa_onnx.OfflineRecognizer.from_zipformer_ctc(
                model=os.path.join(model_dir, "model.int8.onnx"),
                tokens=os.path.join(model_dir, "tokens.txt"),
                num_threads=num_threads,
                sample_rate=16000,
                decoding_method="greedy_search",
                debug=False,
                provider="cpu",
            )
        elif "nemo" in model_dir:
            recognizer = sherpa_onnx.OfflineRecognizer.from_nemo_ctc(
                model=os.path.join(model_dir, "model.onnx"),
                tokens=os.path.join(model_dir, "tokens.txt"),
                num_threads=num_threads,
                sample_rate=16000,
                decoding_method="greedy_search",
                debug=False,
                provider="cpu",
            )
        elif "moonshine" in model_dir:
            recognizer = sherpa_onnx.OfflineRecognizer.from_moonshine(
                preprocessor=os.path.join(model_dir, "preprocess.ort"),
                encoder=os.path.join(model_dir, "encode.int8.ort"),
                uncached_decoder=os.path.join(model_dir, "uncached_decode.int8.ort"),
                cached_decoder=os.path.join(model_dir, "cached_decode.int8.ort"),
                tokens=os.path.join(model_dir, "tokens.txt"),
                num_threads=num_threads,
                decoding_method="greedy_search",
                debug=False,
                provider="cpu",
            )
        else:
            print(f"SKIP: Unknown model type for {model_dir}")
            return None

        print(f"Model loaded (threads={num_threads})")

    except Exception as e:
        print(f"FAIL: {e}")
        return None

    total_cer = 0.0
    count = 0
    errors = 0
    samples_detail = []

    for i, (audio_path, ref_text) in enumerate(items):
        try:
            samples, sr = load_wav_float32(audio_path, target_sr=16000)
            stream = recognizer.create_stream()
            stream.accept_waveform(sr, samples.tolist())
            recognizer.decode_stream(stream)
            hyp_text = str(stream.result.text).strip()
            cer = compute_cer(ref_text, hyp_text)
            total_cer += cer
            count += 1
            samples_detail.append((ref_text, hyp_text, cer))

            if (i + 1) % 100 == 0:
                print(f"  Processed {i+1}/{len(items)}, avg CER so far: {total_cer/count:.4f}")

        except Exception as e:
            errors += 1
            if errors <= 3:
                print(f"  Error on {os.path.basename(audio_path)}: {e}")

    avg_cer = total_cer / max(count, 1)
    print(f"Done: {count} samples, {errors} errors, avg CER = {avg_cer:.4f}")

    # Show 10 worst errors
    worst = heapq.nlargest(10, samples_detail, key=lambda x: x[2])
    print(f"\nWorst CER samples:")
    for ref, hyp, cer in worst:
        print(f"  CER={cer:.3f}  REF: {ref}")
        print(f"           HYP: {hyp}")
        print()

    return {"model": model_name, "samples": count, "errors": errors, "avg_cer": avg_cer}


def main():
    parser = argparse.ArgumentParser(description="ASR model CER benchmark on Chinese test data")
    parser.add_argument("--dataset", choices=["fleurs", "aireport"], default="fleurs",
                        help="Dataset to test (default: fleurs)")
    parser.add_argument("--fleurs-tsv", default="",
                        help="Path to Fleurs TSV file")
    parser.add_argument("--aireport-manifest", default="",
                        help="Path to aireport manifest.jsonl")
    parser.add_argument("--max-samples", type=int, default=945,
                        help="Max samples to test (default: 945 for Fleurs)")
    parser.add_argument("--num-threads", type=int, default=2, help="Number of threads")
    parser.add_argument("--model-dir", nargs=2, action="append",
                        help="Custom model: --model-dir <name> <path> (can repeat)")
    args = parser.parse_args()

    if args.dataset == "fleurs":
        if not args.fleurs_tsv:
            parser.error("--fleurs-tsv is required when --dataset=fleurs")
        items = load_fleurs_tsv(args.fleurs_tsv, max_samples=args.max_samples)
        print(f"Loaded {len(items)} test items from Fleurs zh")
    elif args.dataset == "aireport":
        if not args.aireport_manifest:
            parser.error("--aireport-manifest is required when --dataset=aireport")
        items = load_manifest(args.aireport_manifest, max_samples=args.max_samples)
        print(f"Loaded {len(items)} test items from aireport")

    if not items:
        print("ERROR: No test items found.")
        sys.exit(1)

    if not args.model_dir:
        parser.error("--model-dir <name> <path> is required and can be repeated")
    models = args.model_dir

    all_results = []
    for name, path in models:
        r = test_model(name, path, items, args.num_threads)
        if r:
            all_results.append(r)

    print(f"\n{'='*60}")
    print(f"SUMMARY ({len(items)} samples)")
    print(f"{'='*60}")
    print(f"{'Model':<45} {'Samples':>8} {'Errors':>6} {'Avg CER':>10}")
    print(f"{'-'*60}")
    for r in all_results:
        name_short = r['model'][:44]
        print(f"{name_short:<45} {r['samples']:>8} {r['errors']:>6} {r['avg_cer']:>10.4f}")


if __name__ == "__main__":
    main()
