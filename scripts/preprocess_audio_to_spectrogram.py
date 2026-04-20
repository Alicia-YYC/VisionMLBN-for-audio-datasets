import argparse
import random
from pathlib import Path

import librosa
import numpy as np
from PIL import Image
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Convert WAV files to log-Mel spectrogram PNG dataset")
    parser.add_argument(
        "--input_root",
        type=str,
        default="/shared/data/rawaudio/",
        help="Root directory of raw audio dataset",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="/shared/data/processed/",
        help="Root directory of output spectrogram dataset",
    )
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--duration", type=float, default=5.0, help="Clip duration in seconds")
    parser.add_argument("--n_fft", type=int, default=1024)
    parser.add_argument("--hop_length", type=int, default=256)
    parser.add_argument("--n_mels", type=int, default=128)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
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


def split_files(files, val_ratio, seed):
    files = list(files)
    rng = random.Random(seed)
    rng.shuffle(files)
    n_val = int(len(files) * val_ratio)
    return files[n_val:], files[:n_val]


def process_class(files, class_name, split_name, output_root, args):
    save_dir = output_root / split_name / class_name
    ensure_dir(save_dir)

    for audio_path in tqdm(files, desc=f"{split_name}/{class_name}"):
        try:
            img = audio_to_image(audio_path, args)
            out_path = save_dir / f"{audio_path.stem}.png"
            img.save(out_path)
        except Exception as e:
            print(f"[WARNING] Failed on {audio_path}: {e}")


def main():
    args = parse_args()
    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    ensure_dir(output_root)

    class_dirs = {
        "bird": input_root / "parsed_bird_clips",
        "not_bird": input_root / "parsed_not_bird_clips",
    }

    for class_name, class_dir in class_dirs.items():
        if not class_dir.exists():
            raise FileNotFoundError(f"Missing directory: {class_dir}")

        files = sorted(class_dir.glob("*.wav"))
        print(f"{class_name}: {len(files)} files")

        train_files, val_files = split_files(files, args.val_ratio, args.seed)
        print(f"  train={len(train_files)}, val={len(val_files)}")

        process_class(train_files, class_name, "train", output_root, args)
        process_class(val_files, class_name, "val", output_root, args)

    print(f"\nDone. Output saved to: {output_root}")


if __name__ == "__main__":
    main()