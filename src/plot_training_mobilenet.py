import matplotlib.pyplot as plt

# Replace these with the exact lists from your training output:
train_loss = [0.1884, 0.0272, 0.0117, 0.0053, 0.0020, 0.0019, 0.0092, 0.0043, 0.0046, 0.0045]
val_loss   = [0.0195, 0.0161, 0.0111, 0.0105, 0.0070, 0.0114, 0.0165, 0.0205, 0.0216, 0.0146]

train_acc = [0.9553, 0.9932, 0.9972, 0.9990, 1.0000, 0.9998, 0.9980, 0.9988, 0.9982, 0.9988]
val_acc   = [0.9950, 0.9940, 0.9960, 0.9950, 0.9970, 0.9940, 0.9940, 0.9900, 0.9940, 0.9940]

epochs = range(1, len(train_loss) + 1)

# Accuracy plot
plt.figure()
plt.plot(epochs, train_acc, marker="o", label="Train Acc")
plt.plot(epochs, val_acc, marker="o", label="Val Acc")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("MobileNetV2 Training and Validation Accuracy")
plt.grid(True)
plt.legend()
plt.savefig("../plots/mobilenetv2_accuracy.png", bbox_inches="tight")

# Loss plot
plt.figure()
plt.plot(epochs, train_loss, marker="o", label="Train Loss")
plt.plot(epochs, val_loss, marker="o", label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("MobileNetV2 Training and Validation Loss")
plt.grid(True)
plt.legend()
plt.savefig("../plots/mobilenetv2_loss.png", bbox_inches="tight")
