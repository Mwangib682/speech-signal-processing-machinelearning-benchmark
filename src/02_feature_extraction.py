"""
Stage 2: Acoustic Feature Extraction

Extract 95 numerical acoustic features per preprocessed recording.

Run from the repository root:
    python src/02_feature_extraction.py

Input:  data/processed/{0,...,9}/*.wav
Output: data/voice_features.csv

The feature definitions and parameters match the original experiment.
"""
from pathlib import Path

import librosa
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_FOLDER = PROJECT_ROOT / "data" / "processed"
OUTPUT_CSV = PROJECT_ROOT / "data" / "voice_features.csv"
ERROR_CSV = PROJECT_ROOT / "data" / "feature_extraction_errors.csv"

SR = 16000
N_FFT = 512
HOP = 160
N_MFCC = 13
CLASS_LABELS = range(10)


def summarize(values, name, features):
    """Store finite-value mean and standard deviation for a feature."""
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    features[f"{name}_mean"] = float(np.mean(values)) if values.size else 0.0
    features[f"{name}_std"] = float(np.std(values)) if values.size else 0.0


def extract_features(path):
    """Extract 95 acoustic features from one 16 kHz mono WAV recording."""
    y, sr = librosa.load(path, sr=None, mono=True)
    if sr != SR:
        raise ValueError(f"Expected {SR} Hz, got {sr} Hz")
    if len(y) == 0 or not np.all(np.isfinite(y)):
        raise ValueError("Empty or invalid recording")
    if len(y) < N_FFT:
        y = np.pad(y, (0, N_FFT - len(y)))

    features = {}
    stft = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP))

    mfcc = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP
    )
    if mfcc.shape[1] >= 3:
        width = min(
            9, mfcc.shape[1] if mfcc.shape[1] % 2 else mfcc.shape[1] - 1
        )
        delta = librosa.feature.delta(mfcc, width=width)
        delta2 = librosa.feature.delta(mfcc, width=width, order=2)
    else:
        delta = np.zeros_like(mfcc)
        delta2 = np.zeros_like(mfcc)

    for prefix, matrix in (
        ("mfcc", mfcc),
        ("delta", delta),
        ("delta2", delta2),
    ):
        for i in range(N_MFCC):
            summarize(matrix[i], f"{prefix}_{i + 1}", features)

    scalar_features = {
        "rms": librosa.feature.rms(
            y=y, frame_length=N_FFT, hop_length=HOP
        ),
        "zcr": librosa.feature.zero_crossing_rate(
            y, frame_length=N_FFT, hop_length=HOP
        ),
        "centroid": librosa.feature.spectral_centroid(S=stft, sr=sr),
        "bandwidth": librosa.feature.spectral_bandwidth(S=stft, sr=sr),
        "rolloff": librosa.feature.spectral_rolloff(S=stft, sr=sr),
        "flatness": librosa.feature.spectral_flatness(S=stft),
    }
    for name, values in scalar_features.items():
        summarize(values, name, features)

    normalized = stft / np.maximum(stft.sum(axis=0, keepdims=True), 1e-10)
    flux = np.sqrt(np.sum(np.diff(normalized, axis=1) ** 2, axis=0))
    summarize(flux, "flux", features)

    f0, voiced_flag, _ = librosa.pyin(
        y, sr=sr, fmin=65, fmax=600,
        frame_length=1024, hop_length=HOP
    )
    valid_f0 = f0[np.isfinite(f0)]
    summarize(valid_f0, "f0", features)
    features["voiced_fraction"] = (
        float(np.mean(voiced_flag)) if len(voiced_flag) else 0.0
    )

    if len(features) != 95:
        raise RuntimeError(
            f"Expected 95 acoustic features; extracted {len(features)}"
        )
    return features


def main():
    if not INPUT_FOLDER.is_dir():
        raise FileNotFoundError(
            f"Processed audio folder not found: {INPUT_FOLDER}\n"
            "Run Stage 1 first, with input recordings in data/raw/0 ... data/raw/9."
        )

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    errors = []

    print("STAGE 2: ACOUSTIC FEATURE EXTRACTION")
    print(f"Input: {INPUT_FOLDER}")

    for class_id in CLASS_LABELS:
        folder = INPUT_FOLDER / str(class_id)
        if not folder.is_dir():
            print(f"Missing class folder: {folder}")
            continue

        for path in sorted(folder.glob("*.wav")):
            try:
                row = {
                    "filename": path.name,
                    "class_label": class_id,
                }
                row.update(extract_features(path))
                rows.append(row)
                print(f"OK: {class_id}/{path.name}")
            except Exception as exc:
                errors.append((str(path), str(exc)))
                print(f"FAILED: {path} -> {exc}")

    if not rows:
        raise RuntimeError("No recordings were successfully processed")

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False)
    print(
        f"Saved {len(df)} recordings with {len(df.columns) - 2} "
        f"features to {OUTPUT_CSV}"
    )
    print(f"Failed files: {len(errors)}")
    print(df.head().to_string(index=False, max_cols=8))

    if errors:
        pd.DataFrame(errors, columns=["path", "error"]).to_csv(
            ERROR_CSV, index=False
        )
        print(f"Error report: {ERROR_CSV}")


if __name__ == "__main__":
    main()
