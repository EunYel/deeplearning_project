import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader, random_split
from torchsummary import summary

# Hyperparameters
batch_size = 64
learning_rate = 0.001
epochs = 30
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Data transformations
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.RandomCrop(224, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# Load CIFAR-10 dataset
def load_data():
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

# Freeze feature extractor
def freeze_feature_extractor(model):
    for param in model.features.parameters():
        param.requires_grad = False

# AlexNet with Skip Connection
class AlexNetWithSkip(nn.Module):
    def __init__(self):
        super(AlexNetWithSkip, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.match_channels = nn.Conv2d(64, 192, kernel_size=1, stride=1, padding=0)
        self.nonlinear = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(256 * 6 * 6, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )
        self.linear_regressor = nn.Linear(128, 10)

    def forward(self, x):
        x1 = self.features[0:3](x)
        x2 = self.features[3:6](x1)
        x1_matched = self.match_channels(x1)
        x1_resized = nn.functional.interpolate(x1_matched, size=x2.shape[2:], mode='bilinear', align_corners=False)
        x2 = x2 + x1_resized
        x3 = self.features[6:](x2)
        x = x3.view(x3.size(0), 256 * 6 * 6)
        x = self.nonlinear(x)
        x = self.linear_regressor(x)
        return x

# AlexNet without Skip Connection
class AlexNetWithNoSkip(nn.Module):
    def __init__(self, output_dim=10):
        super(AlexNetWithNoSkip, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.nonlinear = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 6 * 6, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )
        self.linear_regressor = nn.Linear(128, output_dim)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.nonlinear(x)
        x = self.linear_regressor(x)
        return x

def train_model(model, train_loader, val_loader, criterion, optimizer, epochs, model_name):
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            labels_one_hot = F.one_hot(labels, num_classes=10).float().to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels_one_hot)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, preds = outputs.max(1)
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)

        train_loss = running_loss / len(train_loader)
        train_accuracy = correct_train / total_train * 100
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        # Validation phase
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                labels_one_hot = F.one_hot(labels, num_classes=10).float().to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels_one_hot)
                val_loss += loss.item()
                _, preds = outputs.max(1)
                correct_val += (preds == labels).sum().item()
                total_val += labels.size(0)

        val_loss /= len(val_loader)
        val_accuracy = correct_val / total_val * 100
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        print(f"Epoch {epoch + 1}/{epochs}, "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")

    # Save training and validation loss plot
    fig, ax = plt.subplots()
    ax.plot(range(1, epochs + 1), train_losses, label="Train Loss")
    ax.plot(range(1, epochs + 1), val_losses, label="Validation Loss")
    ax.set_title(f"Loss - {model_name}")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid()
    plt.savefig(f"{model_name}_loss.png")
    plt.close(fig)

    # Save training and validation accuracy plot
    fig, ax = plt.subplots()
    ax.plot(range(1, epochs + 1), train_accuracies, label="Train Accuracy")
    ax.plot(range(1, epochs + 1), val_accuracies, label="Validation Accuracy")
    ax.set_title(f"Accuracy - {model_name}")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Accuracy (%)")
    ax.legend()
    ax.grid()
    plt.savefig(f"{model_name}_accuracy.png")
    plt.close(fig)

    print(f"Plots saved for {model_name}")
    return train_losses, val_losses, train_accuracies, val_accuracies

def test_model(model, test_loader, criterion, model_name):
    model.eval()
    test_loss = 0.0
    correct_top1 = 0
    correct_top3 = 0
    total = 0
    all_labels = []
    all_preds = []
    all_outputs = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            labels_one_hot = F.one_hot(labels, num_classes=10).float().to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels_one_hot)
            test_loss += loss.item()

            # Top-1 and Top-3 accuracy
            _, top1_preds = outputs.topk(1, dim=1)
            _, top3_preds = outputs.topk(3, dim=1)

            correct_top1 += (top1_preds.squeeze(1) == labels).sum().item()
            correct_top3 += (labels.unsqueeze(1) == top3_preds).sum().item()
            total += labels.size(0)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(top1_preds.squeeze(1).cpu().numpy())
            all_outputs.append(outputs.cpu().numpy())

    test_loss /= len(test_loader)
    top1_accuracy = correct_top1 / total * 100
    top3_accuracy = correct_top3 / total * 100

    print(f"Test Results for {model_name}:")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Top-1 Accuracy: {top1_accuracy:.2f}%")
    print(f"Top-3 Accuracy: {top3_accuracy:.2f}%")

    # Save confusion matrix
    conf_matrix = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=np.arange(10))
    disp.plot(cmap='Blues', xticks_rotation='vertical')
    plt.title(f"Confusion Matrix: {model_name}")
    plt.savefig(f"{model_name}_confusion_matrix.png")
    plt.close()

    return all_outputs, all_labels
    
    # PCA implementation
def pca_manual(data, n_components=2):
    mean_data = np.mean(data, axis=0)
    centered_data = data - mean_data
    covariance_matrix = np.cov(centered_data, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, sorted_indices]
    eigenvalues = eigenvalues[sorted_indices]
    selected_eigenvectors = eigenvectors[:, :n_components]
    transformed_data = np.dot(centered_data, selected_eigenvectors)
    return transformed_data, eigenvalues, selected_eigenvectors

if __name__ == "__main__":
    # Load data
    train_loader, val_loader, test_loader = load_data()

    # Train and Test AlexNet with Skip Connection
    model_skip = AlexNetWithSkip().to(device)
    freeze_feature_extractor(model_skip)
    optimizer_skip = optim.Adam(model_skip.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    print("Training AlexNetWithSkip...")
    train_model(model_skip, train_loader, val_loader, criterion, optimizer_skip, epochs, "AlexNetWithSkip")
    print("Testing AlexNetWithSkip...")
    outputs_skip, labels_skip = test_model(model_skip, test_loader, criterion, "AlexNetWithSkip")

    # Perform PCA on AlexNetWithSkip outputs
    print("Performing PCA for AlexNetWithSkip...")
    all_outputs_skip = np.concatenate(outputs_skip, axis=0)
    pca_outputs_skip, eigenvalues_skip, _ = pca_manual(all_outputs_skip, n_components=2)

    # Save PCA scatter plot for AlexNetWithSkip
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(pca_outputs_skip[:, 0], pca_outputs_skip[:, 1], c=labels_skip, cmap='tab10', alpha=0.7)
    plt.colorbar(scatter, label="Class Label")
    ax.set_title("2D PCA Embedding of CIFAR-10 Outputs (AlexNetWithSkip)")
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.grid()
    plt.savefig("AlexNetWithSkip_PCA.png")
    plt.close(fig)
    print("PCA plot saved for AlexNetWithSkip.")

    # Train and Test AlexNet without Skip Connection
    model_no_skip = AlexNetWithNoSkip().to(device)
    freeze_feature_extractor(model_no_skip)
    optimizer_no_skip = optim.Adam(model_no_skip.parameters(), lr=learning_rate)

    print("Training AlexNetWithNoSkip...")
    train_model(model_no_skip, train_loader, val_loader, criterion, optimizer_no_skip, epochs, "AlexNetWithNoSkip")
    print("Testing AlexNetWithNoSkip...")
    outputs_no_skip, labels_no_skip = test_model(model_no_skip, test_loader, criterion, "AlexNetWithNoSkip")

    # Perform PCA on AlexNetWithNoSkip outputs
    print("Performing PCA for AlexNetWithNoSkip...")
    all_outputs_no_skip = np.concatenate(outputs_no_skip, axis=0)
    pca_outputs_no_skip, eigenvalues_no_skip, _ = pca_manual(all_outputs_no_skip, n_components=2)

    # Save PCA scatter plot for AlexNetWithNoSkip
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(pca_outputs_no_skip[:, 0], pca_outputs_no_skip[:, 1], c=labels_no_skip, cmap='tab10', alpha=0.7)
    plt.colorbar(scatter, label="Class Label")
    ax.set_title("2D PCA Embedding of CIFAR-10 Outputs (AlexNetWithNoSkip)")
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.grid()
    plt.savefig("AlexNetWithNoSkip_PCA.png")
    plt.close(fig)
    print("PCA plot saved for AlexNetWithNoSkip.")
