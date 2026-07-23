from pathlib import Path

import matplotlib.pyplot as plt

epochs_esc5 = list(range(1, 17))
train_loss_esc5 = [1.5054, 1.2876, 1.2642, 1.1943, 1.1177, 1.0825,
                   0.9006, 0.7822, 0.7574, 0.6939, 0.6452, 0.6024,
                   0.6439, 0.4971, 0.5671, 0.5104]
val_loss_esc5 = [1.1888, 1.0037, 0.9211, 0.8044, 0.8479, 0.6363,
                 0.5730, 0.4500, 0.4239, 0.4727, 0.3865, 0.4072,
                 0.3415, 0.4309, 0.4979, 0.3300]
output_dir = Path("results/figures")
output_dir.mkdir(parents=True, exist_ok=True)

plt.figure(figsize=(8, 5))
plt.plot(epochs_esc5, train_loss_esc5, marker="o", linewidth=2, label="Training loss")
plt.plot(epochs_esc5, val_loss_esc5, marker="s", linewidth=2, label="Validation loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("ESC-5 loss")
plt.xticks(epochs_esc5)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig(output_dir / "esc5_loss_curve.png", dpi=300, bbox_inches="tight")
plt.show()
