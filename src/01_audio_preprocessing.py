
"""
Stage 1: Audio Preprocessing
Project: Comparative Evaluation of Feature Selection and Machine Learning
         Algorithms for Multiclass Speech Classification

Description:
    Preprocesses audio recordings organized into 10 class directories (0-9).

    Operations:
        1. Load audio and convert to mono
        2. Resample to 16,000 Hz
        3. Reduce background noise
        4. Normalize peak amplitude to 0.95
        5. Save as 16-bit PCM WAV

Dependencies:
    numpy, librosa, soundfile, noisereduce

Usage:
    python src/01_audio_preprocessing.py

Expected input:
    data/raw/0/ ... data/raw/9/

Output:
    data/processed/0/ ... data/processed/9/
"""

from pathlib import Path

import numpy as np
import librosa
import soundfile as sf
import noisereduce as nr


# ====================================
# CONFIGURATION
# ====================================

# Automatically locate the project root.
# This script is expected to be inside the "src" directory.

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FOLDER = PROJECT_ROOT / "data" / "raw"
OUTPUT_FOLDER = PROJECT_ROOT / "data" / "processed"

TARGET_SR = 16000
TARGET_PEAK = 0.95

EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}

CLASS_LABELS = [str(i) for i in range(10)]


# ====================================
# PREPROCESSING FUNCTIONS
# ====================================

def resample_audio(audio, original_sr):
    """
    Resample an audio signal to the target sampling rate.

    Args:
        audio: Input audio signal.
        original_sr: Original sampling rate.

    Returns:
        Resampled audio signal.
    """

    if original_sr == TARGET_SR:
        return audio

    return librosa.resample(
        audio,
        orig_sr=original_sr,
        target_sr=TARGET_SR
    )


def normalize_audio(audio):
    """
    Normalize audio to the configured peak amplitude.

    Silent signals are returned unchanged.
    """

    peak = np.max(np.abs(audio))

    if peak < 1e-9:
        return audio

    return audio * (TARGET_PEAK / peak)


def reduce_background_noise(audio):
    """
    Apply nonstationary background noise reduction.

    Uses the same parameters as the original experiment.
    """

    return nr.reduce_noise(
        y=audio,
        sr=TARGET_SR,
        stationary=False,
        prop_decrease=0.7
    )


# ====================================
# PROCESS DATASET
# ====================================

def main():
    """
    Process recordings from the ten class directories.

    Processed recordings are saved to corresponding
    output directories while preserving class labels.
    """

    if not INPUT_FOLDER.is_dir():
        raise FileNotFoundError(
            f"Dataset folder not found: {INPUT_FOLDER}\n"
            "Place your recordings inside data/raw/0 through data/raw/9."
        )

    total = 0
    failed = 0

    print("=" * 60)
    print("STAGE 1: AUDIO PREPROCESSING")
    print("=" * 60)

    print(f"Input directory: {INPUT_FOLDER}")
    print(f"Output directory: {OUTPUT_FOLDER}")
    print(f"Target sampling rate: {TARGET_SR} Hz")
    print(f"Target peak amplitude: {TARGET_PEAK}")

    for class_name in CLASS_LABELS:

        source_folder = INPUT_FOLDER / class_name
        target_folder = OUTPUT_FOLDER / class_name

        if not source_folder.is_dir():
            print(f"Skipping missing folder: {class_name}")
            continue

        target_folder.mkdir(
            parents=True,
            exist_ok=True
        )

        for file_path in sorted(source_folder.iterdir()):

            if file_path.suffix.lower() not in EXTENSIONS:
                continue

            try:
                # Load recording and convert to mono.
                audio, sr = librosa.load(
                    file_path,
                    sr=None,
                    mono=True
                )

                if len(audio) == 0:
                    raise ValueError("Empty audio file")

                # 1. Resample to 16 kHz.
                audio = resample_audio(audio, sr)

                # 2. Reduce background noise.
                audio = reduce_background_noise(audio)

                # 3. Normalize peak amplitude.
                audio = normalize_audio(audio)

                # 4. Save processed audio as WAV.
                output_path = (
                    target_folder / (file_path.stem + ".wav")
                )

                sf.write(
                    output_path,
                    audio,
                    TARGET_SR,
                    subtype="PCM_16"
                )

                total += 1

                print(
                    f"Processed: {class_name}/{file_path.name}"
                )

            except Exception as error:

                failed += 1

                print(
                    f"FAILED: {class_name}/{file_path.name}: {error}"
                )

    # ====================================
    # FINAL SUMMARY
    # ====================================

    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETED")
    print("=" * 60)

    print(f"Successful recordings: {total}")
    print(f"Failed recordings: {failed}")
    print(f"Output folder: {OUTPUT_FOLDER}")


if __name__ == "__main__":
    main()
