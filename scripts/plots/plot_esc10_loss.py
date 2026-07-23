from pathlib import Path

import matplotlib.pyplot as plt

epochs_esc10 = list(range(1, 21))
train_loss_esc10 = [2.0451, 1.8015, 1.7590, 1.6334, 1.4873, 1.3831,
                    1.2068, 1.0851, 0.9697, 0.9211, 0.9187, 0.8351,
                    0.8266, 0.8596, 0.7784, 0.6821, 0.6777, 0.7107,
                    0.7194, 0.6888]
val_loss_esc10 = [1.5528, 1.6488, 1.4463, 1.2215, 1.2356, 1.1171,
                  0.9561, 1.0191, 0.8774, 0.8932, 0.8810, 0.8325,
                  0.9343, 0.8222, 0.8232, 0.8468, 0.8348, 0.9566,
                  0.9439, 0.9497]
output_dir = Path("results/figures")
output_dir.mkdir(parents=True, exist_ok=True)

plt.figure(figsize=(8, 5))
plt.plot(epochs_esc10, train_loss_esc10, marker="o", linewidth=2, label="Training loss")
plt.plot(epochs_esc10, val_loss_esc10, marker="s", linewidth=2, label="Validation loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("ESC-10 loss")
plt.xticks(epochs_esc10)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig(output_dir / "esc10_loss_curve.png", dpi=300, bbox_inches="tight")
plt.show()
