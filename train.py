import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm.auto import tqdm 
from sklearn.metrics import f1_score 

# expects train and val folders created by 01_split_dataset py

SEED = 42
TRAIN_DIR = "train"
VAL_DIR = "val"
IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4
PATIENCE = 5  
CHECKPOINT_PATH = "best_model.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)


class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        # five conv layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()

        # we resize images to 224x224
        # after three pool layers the size is 28x28
        # final channels are 256 so flattened size is 256 * 28 * 28
        self.fc1 = nn.Linear(256 * 28 * 28, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))  # 3 -> 32  224 -> 112
        x = self.pool(self.relu(self.conv2(x)))  # 32 -> 64 112 -> 56
        x = self.pool(self.relu(self.conv3(x)))  # 64 -> 128 56 -> 28
        x = self.relu(self.conv4(x))             # 128 -> 256 28 stays 28
        x = self.relu(self.conv5(x))             # 256 -> 256 28 stays 28
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def make_dataloaders():
    train_transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
        ]
    )

    train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
    val_dataset = datasets.ImageFolder(VAL_DIR, transform=val_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    return train_loader, val_loader, train_dataset


def train():
    set_seed(SEED)

    train_loader, val_loader, train_dataset = make_dataloaders()
    num_classes = len(train_dataset.classes)

    model = SimpleCNN(num_classes=num_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(NUM_EPOCHS):
        print(f"epoch {epoch + 1}/{NUM_EPOCHS}")

        # training loop
        model.train()
        total_train_loss = 0.0

        for images, labels in tqdm(train_loader, desc="train", leave=False):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item() * labels.size(0)

        avg_train_loss = total_train_loss / len(train_loader.dataset)

        # validation loop
        model.eval()
        total_val_loss = 0.0
        correct = 0
        total = 0

        all_preds = []  
        all_labels = []

        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="val", leave=False):
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)

                outputs = model(images)
                loss = criterion(outputs, labels)

                total_val_loss += loss.item() * labels.size(0)
                preds = outputs.argmax(dim=1)

                correct += (preds == labels).sum().item()
                total += labels.size(0)

                all_preds.append(preds)
                all_labels.append(labels)

        avg_val_loss = total_val_loss / len(val_loader.dataset)
        val_accuracy = 100.0 * correct / total

        # compute F1
        all_preds = torch.cat(all_preds).cpu().numpy()
        all_labels = torch.cat(all_labels).cpu().numpy()
        val_f1 = f1_score(all_labels, all_preds, average="macro")

        print(
            f"train_loss {avg_train_loss:.4f}  "
            f"val_loss {avg_val_loss:.4f}  "
            f"val_acc {val_accuracy:.2f}%  "
            f"val_f1 {val_f1:.4f}"
        )


        # early stopping and checkpoint
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), CHECKPOINT_PATH)
            print("saved new best model")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= PATIENCE:
                print("early stopping")
                break

    print("training finished")


if __name__ == "__main__":
    train()