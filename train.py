import os
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

import warnings
warnings.filterwarnings("ignore")

ImageFile.LOAD_TRUNCATED_IMAGES = True


#============================
# Device Setup
#============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on: {device}")


# ============================
# Dataset Download
# ============================
import kagglehub

DATA_DIR = Path("data/bone-fracture-xray")
train_dir = DATA_DIR / "Bone_Fracture_Dataset/train"
val_dir = DATA_DIR / "Bone_Fracture_Dataset/val"


def download_dataset_if_needed():
    if DATA_DIR.exists():
        print("Dataset already exists at:", DATA_DIR.resolve())
        return

    print("Dataset not found. Downloading...")
    cache_path = Path(kagglehub.dataset_download("usman44m/bone-fracture-x-ray-dataset"))
    DATA_DIR.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(cache_path, DATA_DIR)
    print("Dataset ready at:", DATA_DIR.resolve())


# ============================
# Exclusions
# =============================
VALID_EXTS = (".png", ".jpg", ".jpeg")
IGNORE = {
    'IMG0004347.jpg',
    'IMG0004148.jpg',
    'IMG0004134.jpg',
    'IMG0004149.jpg',
    'IMG0004143.jpg',
    'IMG0004308.jpg'
}

class XRayDataset(Dataset):
    def __init__(self, data_dir, transform=None, ignore_files=None):
        self.data_dir = data_dir
        self.transform = transform
        self.ignore_files = set(ignore_files) if ignore_files else set()

        self.image_paths = []
        self.labels = []

        self.classes = sorted(os.listdir(data_dir))
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        self.ignored_files = []

        for label_name in self.classes:
            label_dir = os.path.join(data_dir, label_name)
            for img_name in os.listdir(label_dir):
                if img_name in self.ignore_files:
                    continue
                if not img_name.lower().endswith(VALID_EXTS):
                    self.ignored_files.append(img_name)
                    continue

                img_path = os.path.join(label_dir, img_name)
                if not os.path.isfile(img_path):
                    continue

                self.image_paths.append(img_path)
                self.labels.append(self.class_to_idx[label_name])

        if self.ignored_files:
            print(f"Ignored {len(self.ignored_files)} files in {data_dir}:")
            print(self.ignored_files)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = float(self.labels[idx])

        if self.transform:
            image = self.transform(image)

        return image, label


# ============================
# Transforms & Dataloaders
# ============================

def get_transforms(input_size=224):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    base = [
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ]

    return transforms.Compose(base), transforms.Compose(base)

def get_dataloaders(batch_size=32, num_workers=4, pin_memory=True):
    train_transforms, val_transforms = get_transforms()

    train_dataset = XRayDataset(
        data_dir=train_dir,
        transform=train_transforms,
        ignore_files=IGNORE
    )

    val_dataset = XRayDataset(
        data_dir=val_dir,
        transform=val_transforms,
        ignore_files=IGNORE
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory)

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin_memory)
    return train_loader, val_loader


# ============================
# Defining the Model
# =============================

class FractureClassifierEfficientNet(nn.Module):
    def __init__(self, hidden_dim, droprate):
        super().__init__()
        self.base_model = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1
        )

        for param in self.base_model.parameters():
            param.requires_grad = False

        in_features = self.base_model.classifier[1].in_features

        self.base_model.classifier = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(droprate),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.base_model(x)


def make_model(hidden_dim, lr, droprate):
    model = FractureClassifierEfficientNet(hidden_dim=hidden_dim, droprate=droprate)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    return model, optimizer


# ============================
# Metrics
# ============================

def loss_and_accuracy(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.float().to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels.unsqueeze(1))

            running_loss += loss.item()
            preds = (torch.sigmoid(outputs) > 0.5).float()

            total += labels.size(0)
            correct += (preds.squeeze() == labels).sum().item()

    return running_loss / len(dataloader), correct / total


# ============================
# Training Loop
# ============================

def train_and_evaluate(model, optimizer, train_loader, val_loader, criterion,
                       num_epochs, device, patience=8):

    best_val_acc = 0.0
    best_val_loss = float("inf")
    patience_counter = 0

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": []
    }

    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.float().to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.unsqueeze(1))
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            preds = (torch.sigmoid(outputs) > 0.5).float()

            total += labels.size(0)
            correct += (preds.squeeze() == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = correct / total

        val_loss, val_acc = loss_and_accuracy(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # ---------------------------
        # Checkpointing
        # ---------------------------
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint_path = f"checkpoints/fracture_classifier_epoch{epoch+1:02d}_valacc{val_acc:.3f}.pth"
            torch.save(model.state_dict(), checkpoint_path)
            best_checkpoint = checkpoint_path
            print(f"Checkpoint saved: {checkpoint_path}")

        # ---------------------------
        # Early Stopping
        # ---------------------------
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

    return history, best_checkpoint

  
# ===========================
# Plotting
# ==========================
def plot_training_curves(history):
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["train_acc"], label="Train Accuracy")
    plt.plot(epochs, history["val_acc"], label="Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Curve")
    plt.legend()

    plt.tight_layout()
    plt.savefig("training_curves.png")
    print("Loss and accuracy curves saved as training_curves.png")


# ============================
# Hyperparameters
# ===========================
lr = 0.001
hidden_dim = 256
droprate = 0.3
num_epochs = 50
patience = 8
batch_size = 32
num_workers = 4


# ============================
# Running Everything
# ===========================

download_dataset_if_needed()
train_loader, val_loader = get_dataloaders(batch_size=batch_size, num_workers=num_workers)

model, optimizer = make_model(hidden_dim=hidden_dim, lr=lr, droprate=droprate)

history, best_checkpoint = train_and_evaluate(
    model=model,
    optimizer=optimizer,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=nn.BCEWithLogitsLoss(),
    num_epochs=num_epochs,
    device=device,
    patience=patience
)

plot_training_curves(history)
shutil.copy(best_checkpoint, "model.pth")
print("Best model saved as model.pth")