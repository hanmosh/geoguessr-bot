import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm.auto import tqdm
from sklearn.metrics import f1_score

from train import SimpleCNN, IMAGE_SIZE, BATCH_SIZE, DEVICE, CHECKPOINT_PATH

TEST_DIR = "test"


def make_test_loader():
    # same resize as train but no data augmentation
    test_transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
        ]
    )

    test_dataset = datasets.ImageFolder(TEST_DIR, transform=test_transform)

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    return test_loader, test_dataset


def evaluate():
    test_loader, test_dataset = make_test_loader()
    num_classes = len(test_dataset.classes)

    # use the model from train py
    model = SimpleCNN(num_classes=num_classes).to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    model.eval()

    correct = 0
    total = 0

    all_preds = []  
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="test", leave=False):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            preds = outputs.argmax(dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_preds.append(preds)
            all_labels.append(labels)

    test_accuracy = 100.0 * correct / total

    # compute F1
    all_preds = torch.cat(all_preds).cpu().numpy()
    all_labels = torch.cat(all_labels).cpu().numpy()
    test_f1 = f1_score(all_labels, all_preds, average="macro")

    print(f"test_acc {test_accuracy:.2f}%  test_f1 {test_f1:.4f}")


if __name__ == "__main__":
    evaluate()