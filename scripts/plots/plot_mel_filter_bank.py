from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import librosa

sr = 16000
n_fft = 1024
n_mels = 20

mel_filters = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels)

freqs = np.linspace(0, sr / 2, mel_filters.shape[1])
output_dir = Path("results/figures")
output_dir.mkdir(parents=True, exist_ok=True)

plt.figure(figsize=(8, 5))
for i in range(n_mels):
    plt.plot(freqs, mel_filters[i], linewidth=1.2)

plt.xlabel("Frequency (Hz)")
plt.ylabel("Filter response")
plt.title("Mel filter bank")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(output_dir / "mel_filter_bank.png", dpi=300, bbox_inches="tight")
plt.show()
