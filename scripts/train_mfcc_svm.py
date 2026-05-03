import os
import argparse
import numpy as np
import pandas as pd
import librosa

from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report


def extract_mfcc_feature(file_path, sr=16000, duration=5.0, n_mfcc=20):
    y, _ = librosa.load(file_path, sr=sr, mono=True)

    target_len = int(sr * duration)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

    feat = np.concatenate([
        np.mean(mfcc, axis=1),
        np.std(mfcc, axis=1)
    ])

    return feat.astype(np.float32)


def build_dataset(df, audio_dir, sr=16000, duration=5.0, n_mfcc=20):
    X, y = [], []

    class_names = sorted(df["category"].unique().tolist())
    class_to_idx = {c: i for i, c in enumerate(class_names)}

    for _, row in df.iterrows():
        audio_path = os.path.join(audio_dir, row["filename"])
        if not os.path.exists(audio_path):
            print(f"[WARN] File not found: {audio_path}")
            continue

        feat = extract_mfcc_feature(
            audio_path,
            sr=sr,
            duration=duration,
            n_mfcc=n_mfcc
        )
        X.append(feat)
        y.append(class_to_idx[row["category"]])

    if len(X) == 0:
        raise ValueError("No audio files loaded. Please check audio_dir and csv_path.")

    X = np.array(X)
    y = np.array(y)
    return X, y, class_names


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--audio_dir", type=str, required=True)
    parser.add_argument("--train_folds", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--val_fold", type=int, default=5)
    parser.add_argument("--esc10_only", action="store_true")
    parser.add_argument("--sr", type=int, default=16000)
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--n_mfcc", type=int, default=20)
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--kernel", type=str, default="rbf")
    parser.add_argument("--categories", type=str, nargs="+", default=None)
    args = parser.parse_args()

    df = pd.read_csv(args.csv_path)

    if args.esc10_only:
      df = df[df["esc10"].astype(str).str.lower() == "true"].copy()

    if args.categories is not None:
      df = df[df["category"].isin(args.categories)].copy()

    train_df = df[df["fold"].isin(args.train_folds)].copy()
    val_df = df[df["fold"] == args.val_fold].copy()
    if len(train_df) == 0 or len(val_df) == 0:
      raise ValueError("Filtered train/val set is empty. Check folds and category names.")

    print("Selected categories:", sorted(df["category"].unique().tolist()))

    print("Train samples:", len(train_df))
    print("Val samples:", len(val_df))
    print("Train categories:", sorted(train_df["category"].unique().tolist()))
    print("Val categories:", sorted(val_df["category"].unique().tolist()))

    X_train, y_train, class_names = build_dataset(
        train_df,
        args.audio_dir,
        sr=args.sr,
        duration=args.duration,
        n_mfcc=args.n_mfcc
    )

    X_val, y_val, _ = build_dataset(
        val_df,
        args.audio_dir,
        sr=args.sr,
        duration=args.duration,
        n_mfcc=args.n_mfcc
    )

    print("X_train shape:", X_train.shape)
    print("X_val shape:", X_val.shape)
    print("Classes:", class_names)

    model = make_pipeline(
        StandardScaler(),
        SVC(C=args.C, kernel=args.kernel)
    )

    print("Training SVM...")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    acc = accuracy_score(y_val, y_pred)

    print(f"Validation Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred, target_names=class_names, digits=4))


if __name__ == "__main__":
    main()
