#!/usr/bin/env python3
"""Generate an empirical WHAM/WHAMR train speech-to-noise SNR distribution."""

from __future__ import annotations

import argparse
import csv
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf


def compute_snr(item: tuple[Path, Path]) -> tuple[str, float]:
    speech_path, noise_path = item
    speech, speech_sr = sf.read(speech_path, dtype="float32")
    noise, noise_sr = sf.read(noise_path, dtype="float32")
    if speech_sr != noise_sr:
        raise ValueError(f"Sample-rate mismatch for {speech_path.name}")

    length = min(len(speech), len(noise))
    if length == 0:
        raise ValueError(f"Empty waveform for {speech_path.name}")
    speech = speech[:length].astype(np.float64, copy=False)
    noise = noise[:length].astype(np.float64, copy=False)
    speech_power = np.mean(np.square(speech))
    noise_power = np.mean(np.square(noise))
    snr_db = 10.0 * np.log10((speech_power + 1.0e-12) / (noise_power + 1.0e-12))
    if not np.isfinite(snr_db):
        raise ValueError(f"Invalid SNR for {speech_path.name}")
    return speech_path.name, float(snr_db)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--speech-subdir", default="mix_clean_reverb")
    parser.add_argument("--dataset-name", default="WHAMR")
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    args = parser.parse_args()

    train_root = args.dataset_root / "tr"
    speech_dir = train_root / args.speech_subdir
    noise_dir = train_root / "noise"
    if not speech_dir.is_dir() or not noise_dir.is_dir():
        raise FileNotFoundError(
            f"Expected {args.dataset_name} directories: {speech_dir} and {noise_dir}"
        )

    speech_files = sorted(speech_dir.glob("*.wav"))
    if not speech_files:
        raise FileNotFoundError(f"No wav files found in {speech_dir}")
    pairs = [(path, noise_dir / path.name) for path in speech_files]
    missing = [str(noise_path) for _, noise_path in pairs if not noise_path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} noise files; first: {missing[0]}")

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        rows = list(executor.map(compute_snr, pairs))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["utterance_id", "snr_db"])
        writer.writerows((key, f"{snr_db:.8f}") for key, snr_db in rows)
    temporary.replace(args.output)

    values = np.asarray([snr_db for _, snr_db in rows])
    print(
        f"[DONE] {len(rows)} {args.dataset_name} SNR values -> {args.output} "
        f"(min={values.min():.3f}, max={values.max():.3f}, mean={values.mean():.3f})"
    )


if __name__ == "__main__":
    main()
