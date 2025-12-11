import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm.auto import tqdm
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import sys
import importlib

TEST_DIR = "test"


def make_test_loader(IMAGE_SIZE, BATCH_SIZE):
    test_transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    test_dataset = datasets.ImageFolder(TEST_DIR, transform=test_transform)

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    return test_loader, test_dataset


def evaluate(module_name="train1"):
    train_module = importlib.import_module(module_name)
    SimpleCNN = train_module.SimpleCNN
    IMAGE_SIZE = train_module.IMAGE_SIZE
    BATCH_SIZE = train_module.BATCH_SIZE
    DEVICE = train_module.DEVICE
    CHECKPOINT_PATH = train_module.CHECKPOINT_PATH

    test_loader, test_dataset = make_test_loader(IMAGE_SIZE, BATCH_SIZE)
    num_classes = len(test_dataset.classes)

    # use the model from train py
    model = SimpleCNN(num_classes=num_classes).to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    model.eval()

    correct = 0
    total = 0

    all_preds = []  
    all_labels = []
    mis_images = []
    mis_true = []
    mis_pred = []
    mis_prob = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="test", leave=False):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            max_probs, preds = probs.max(dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_preds.append(preds)
            all_labels.append(labels)

            for img, label, pred, prob in zip(images.cpu(), labels.cpu(), preds.cpu(), max_probs.cpu()):
                if label.item() != pred.item():
                    mis_images.append(img)
                    mis_true.append(label.item())
                    mis_pred.append(pred.item())
                    mis_prob.append(prob.item())

    test_accuracy = 100.0 * correct / total

    # compute F1
    all_preds = torch.cat(all_preds).cpu().numpy()
    all_labels = torch.cat(all_labels).cpu().numpy()

    cm = confusion_matrix(all_labels, all_preds)
    print("Confusion matrix:\n", cm)

    print("\nPer-class report:")
    print(classification_report(
        all_labels,
        all_preds,
        target_names=test_dataset.classes
    ))

    test_f1 = f1_score(all_labels, all_preds, average="macro")

    print(f"test_acc {test_accuracy:.2f}%  test_f1 {test_f1:.4f}")

    # visualization of misclassified examples
    if mis_images:
        n = min(8, len(mis_images))
        fig, axes = plt.subplots(1, n, figsize=(3 * n, 3))
        if n == 1:
            axes = [axes]
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        for i in range(n):
            img = mis_images[i]
            img = img.permute(1, 2, 0).numpy()
            img = std * img + mean
            img = np.clip(img, 0, 1)
            ax = axes[i]
            ax.imshow(img)
            true_label = test_dataset.classes[mis_true[i]]
            pred_label = test_dataset.classes[mis_pred[i]]
            prob = mis_prob[i]
            ax.set_title(f"true: {true_label}\npred: {pred_label}\nconf: {prob:.2f}")
            ax.axis("off")
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    module_name = sys.argv[1] if len(sys.argv) > 1 else "train1"
    evaluate(module_name)