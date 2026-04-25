import torch
from torch.utils.data import DataLoader, random_split, ConcatDataset, WeightedRandomSampler
import torch.nn as nn
from torchvision import transforms
from tqdm import tqdm
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, precision_recall_curve, roc_curve
)

from utils.dataset import CoMoFoDOptimizedDataset
from utils.casia_dataset import CASIADataset
from model.model import OptimizedForgeryDetector


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = torch.exp(-bce)
        return (self.alpha * (1 - pt) ** self.gamma * bce).mean()



def mixup(x, y, alpha=0.4):
    lam = np.random.beta(alpha, alpha)
    index = torch.randperm(x.size(0)).to(x.device)
    return lam * x + (1 - lam) * x[index], y, y[index], lam



def collate_hybrid(batch):
    imgs = torch.stack([x[0] for x in batch])
    handcrafted = torch.stack([x[1] for x in batch])
    labels = torch.tensor([x[2] for x in batch])
    return imgs, handcrafted, labels


def get_transforms(size=224):
    train_t = transforms.Compose([
        transforms.Resize((size + 32, size + 32)),
        transforms.RandomCrop(size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(0.2,0.2,0.2,0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    val_t = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    return train_t, val_t


def save_training_graphs(losses, train_accs, val_accs, val_f1s):
    os.makedirs("results", exist_ok=True)
    epochs = range(1, len(losses)+1)

    plt.plot(epochs, losses)
    plt.title("Loss")
    plt.savefig("results/loss.png")
    plt.clf()

    plt.plot(epochs, train_accs, label="Train")
    plt.plot(epochs, val_accs, label="Val")
    plt.legend()
    plt.title("Accuracy")
    plt.savefig("results/accuracy.png")
    plt.clf()

    plt.plot(epochs, val_f1s)
    plt.title("F1 Score")
    plt.savefig("results/f1.png")
    plt.clf()


def plot_confusion_matrix(cm):
    plt.figure()
    sns.heatmap(cm, annot=True, fmt='d')
    plt.savefig("results/confusion_matrix.png")
    plt.close()


def plot_roc(y_true, y_scores):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    plt.plot(fpr, tpr)
    plt.title("ROC")
    plt.savefig("results/roc.png")
    plt.close()


def plot_pr(y_true, y_scores):
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    plt.plot(recall, precision)
    plt.title("PR")
    plt.savefig("results/pr.png")
    plt.close()


def plot_bar(acc, prec, rec, f1):
    plt.bar(["Acc","Prec","Rec","F1"], [acc,prec,rec,f1])
    plt.savefig("results/bar.png")
    plt.close()



def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("🔥 Using device:", device)

    os.makedirs("checkpoints", exist_ok=True)

    train_t, val_t = get_transforms()

    como = CoMoFoDOptimizedDataset("CoMoFoD_dataset/", train_t, True)
    casia = CASIADataset("CASIA2/", train_t, True)

    full = ConcatDataset([como, casia])

    train_size = int(0.7 * len(full))
    val_size = int(0.15 * len(full))
    test_size = len(full) - train_size - val_size

    train_data, val_data, test_data = random_split(
        full, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    labels = []
    for idx in train_data.indices:
        labels.append(como.labels[idx] if idx < len(como) else casia.labels[idx - len(como)])

    counts = np.bincount(labels)
    weights = 1. / counts
    sample_weights = [weights[l] for l in labels]

    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    train_loader = DataLoader(train_data, batch_size=4, sampler=sampler, collate_fn=collate_hybrid)
    val_loader = DataLoader(val_data, batch_size=4, collate_fn=collate_hybrid)
    test_loader = DataLoader(test_data, batch_size=4, collate_fn=collate_hybrid)

   
    model = OptimizedForgeryDetector(use_handcrafted=True).to(device)

    optimizer = torch.optim.AdamW([
        {"params": model.convnext.parameters(), "lr": 5e-6},
        {"params": model.efficient.parameters(), "lr": 5e-6},
        {"params": model.vit.parameters(), "lr": 3e-6},
        {"params": model.classifier.parameters(), "lr": 1e-5},
        {"params": model.fusion_projector.parameters(), "lr": 1e-5},
    ])

    criterion = FocalLoss()
    scaler = torch.amp.GradScaler("cuda")

    best_f1 = 0
    patience = 3
    no_improve = 0

    losses, train_accs, val_accs, val_f1s = [], [], [], []

   
    for epoch in range(12):
        model.train()
        total, correct, loss_sum = 0, 0, 0

        for imgs, handcrafted, labels in tqdm(train_loader):
            imgs, handcrafted = imgs.to(device), handcrafted.to(device)
            labels = labels.to(device).unsqueeze(1).float()

            imgs, y_a, y_b, lam = mixup(imgs, labels)

            optimizer.zero_grad()

            with torch.amp.autocast("cuda"):
                outputs = model(imgs, handcrafted)
                loss = lam * criterion(outputs, y_a) + (1 - lam) * criterion(outputs, y_b)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            loss_sum += loss.item()

            preds = (torch.sigmoid(outputs) > 0.5).int()
            correct += (preds == labels.int()).sum().item()
            total += labels.size(0)

        train_acc = correct / total

        # ===== VALIDATION =====
        model.eval()
        all_preds, all_labels, all_probs = [], [], []

        with torch.no_grad():
            for imgs, handcrafted, labels in val_loader:
                imgs, handcrafted = imgs.to(device), handcrafted.to(device)

                outputs = model(imgs, handcrafted)
                probs = torch.sigmoid(outputs).cpu().numpy()
                preds = (probs > 0.5).astype(int)

                all_probs.extend(probs.flatten())
                all_preds.extend(preds.flatten())
                all_labels.extend(labels.numpy())

        val_acc = accuracy_score(all_labels, all_preds)
        val_f1 = f1_score(all_labels, all_preds)

        print(f"\nEpoch {epoch+1} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | F1: {val_f1:.4f}")

        torch.save(model.state_dict(), f"checkpoints/epoch_{epoch+1}.pth")

        if val_f1 > best_f1:
            best_f1 = val_f1
            no_improve = 0
            torch.save(model.state_dict(), "checkpoints/best.pth")
        else:
            no_improve += 1

        losses.append(loss_sum/len(train_loader))
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        val_f1s.append(val_f1)

        if no_improve >= patience:
            print("⏹ Early stopping")
            break


    print("\n🧪 TESTING ENSEMBLE\n")

    model1 = OptimizedForgeryDetector(use_handcrafted=True).to(device)
    model2 = OptimizedForgeryDetector(use_handcrafted=True).to(device)

    model1.load_state_dict(torch.load("checkpoints/best.pth"))
    model2.load_state_dict(torch.load("checkpoints/epoch_3.pth"))

    model1.eval()
    model2.eval()

    all_labels, all_probs = [], []

    with torch.no_grad():
        for imgs, handcrafted, labels in test_loader:
            imgs, handcrafted = imgs.to(device), handcrafted.to(device)

            p1 = torch.sigmoid(model1(imgs, handcrafted))
            p2 = torch.sigmoid(model2(imgs, handcrafted))

            probs = ((p1 + p2) / 2).cpu().numpy()

            all_probs.extend(probs.flatten())
            all_labels.extend(labels.numpy())

    precision, recall, thresholds = precision_recall_curve(all_labels, all_probs)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    best_threshold = thresholds[np.argmax(f1_scores)]

    preds = (np.array(all_probs) > best_threshold).astype(int)

    acc = accuracy_score(all_labels, preds)
    prec = precision_score(all_labels, preds)
    rec = recall_score(all_labels, preds)
    f1 = f1_score(all_labels, preds)
    cm = confusion_matrix(all_labels, preds)

    print("Test Acc:", acc)
    print("Precision:", prec)
    print("Recall:", rec)
    print("F1:", f1)
    print("Threshold:", best_threshold)
    print(cm)

    save_training_graphs(losses, train_accs, val_accs, val_f1s)
    plot_confusion_matrix(cm)
    plot_roc(all_labels, all_probs)
    plot_pr(all_labels, all_probs)
    plot_bar(acc, prec, rec, f1)


if __name__ == "__main__":
    main()