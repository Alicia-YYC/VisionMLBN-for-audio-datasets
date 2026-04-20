import argparse
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Convert ESC-50 WAV files to log-Mel spectrogram PNG dataset")
    parser.add_argument(
        "--esc50_root",
        type=str,
        default="/shared/data/ESC-50-master",
        help="Root directory of ESC-50 dataset",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="/shared/data/processed_esc50/",
        help="Root directory of output spectrogram dataset",
    )
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--duration", type=float, default=5.0, help="Clip duration in seconds")
    parser.add_argument("--n_fft", type=int, default=1024)
    parser.add_argument("--hop_length", type=int, default=256)
    parser.add_argument("--n_mels", type=int, default=128)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument(
        "--val_fold",
        type=int,
        default=1,
        help="Which ESC-50 fold to use as validation set (1~5)",
    )
    return parser.parse_args()


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def load_audio_fixed_length(audio_path: Path, sample_rate: int, duration: float):
    target_length = int(sample_rate * duration)
    y, sr = librosa.load(audio_path, sr=sample_rate, mono=True)

    if len(y) < target_length:
        y = np.pad(y, (0, target_length - len(y)), mode="constant")
    else:
        y = y[:target_length]

    return y, sr


def audio_to_image(audio_path: Path, args):
    y, sr = load_audio_fixed_length(audio_path, args.sample_rate, args.duration)

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        n_mels=args.n_mels,
        power=2.0,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)

    x_min, x_max = log_mel.min(), log_mel.max()
    if x_max - x_min < 1e-8:
        arr = np.zeros_like(log_mel, dtype=np.uint8)
    else:
        arr = ((log_mel - x_min) / (x_max - x_min) * 255).astype(np.uint8)

    img = Image.fromarray(arr).convert("L")
    img = img.resize((args.img_size, args.img_size), resample=Image.Resampling.BICUBIC)
    img = img.convert("RGB")
    return img


def sanitize_class_name(name: str) -> str:
    return name.strip().replace(" ", "_").replace("/", "_")


def process_row(row, audio_dir: Path, output_root: Path, args):
    filename = row["filename"]
    category = sanitize_class_name(row["category"])
    fold = int(row["fold"])

    split_name = "val" if fold == args.val_fold else "train"
    save_dir = output_root / split_name / category
    ensure_dir(save_dir)

    audio_path = audio_dir / filename
    if not audio_path.exists():
        raise FileNotFoundError(f"Missing audio file: {audio_path}")

    img = audio_to_image(audio_path, args)
    out_path = save_dir / f"{audio_path.stem}.png"
    img.save(out_path)


def main():
    args = parse_args()
    esc50_root = Path(args.esc50_root)
    output_root = Path(args.output_root)
    ensure_dir(output_root)

    csv_path = esc50_root / "meta" / "esc50.csv"
    audio_dir = esc50_root / "audio"

    if not csv_path.exists():
        raise FileNotFoundError(f"Missing CSV: {csv_path}")
    if not audio_dir.exists():
        raise FileNotFoundError(f"Missing audio directory: {audio_dir}")

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} samples from {csv_path}")
    print(f"Number of classes: {df['category'].nunique()}")
    print(f"Validation fold: {args.val_fold}")

    n_train = (df["fold"] != args.val_fold).sum()
    n_val = (df["fold"] == args.val_fold).sum()
    print(f"train={n_train}, val={n_val}")

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing ESC-50"):
        try:
            process_row(row, audio_dir, output_root, args)
        except Exception as e:
            print(f"[WARNING] Failed on {row['filename']}: {e}")

    print(f"\nDone. Output saved to: {output_root}")


if __name__ == "__main__":
    main()