import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import shutil
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from utils import EmotionAudioDataset, MSTRModel
from colorama import Fore

def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            out = model(X)
            _, pred = torch.max(out, 1)
            total += y.size(0)
            correct += (pred == y).sum().item()
    return 100 * correct / total

def plot_confusion_matrix(model, loader, label_map, device):
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            out = model(X)
            _, pred = torch.max(out, 1)
            y_true.extend(y.cpu().numpy())
            y_pred.extend(pred.cpu().numpy())
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_map.keys(), yticklabels=label_map.keys())
    plt.xlabel('predicted')
    plt.ylabel('true')
    plt.title('confusion matrix')
    plt.savefig('confusion_matrix.png')
    print("confusion matrix saved as confusion_matrix.png")
    print("\nclassification report:")
    print(classification_report(y_true, y_pred, target_names=label_map.keys()))

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = EmotionAudioDataset('./Tess', max_len=100)
    if len(dataset) == 0:
        print("no data found. exiting.")
        exit()
    
    train_idx, test_idx = train_test_split(
        range(len(dataset)),
        test_size=0.2,
        random_state=42,
        stratify=[dataset[i][1] for i in range(len(dataset))]
    )
    train_loader = DataLoader(
        dataset,
        batch_size=16,
        sampler=torch.utils.data.SubsetRandomSampler(train_idx),
        num_workers=4,
        pin_memory=True if device.type == 'cuda' else False
    )
    test_loader = DataLoader(
        dataset,
        batch_size=16,
        sampler=torch.utils.data.SubsetRandomSampler(test_idx),
        num_workers=4,
        pin_memory=True if device.type == 'cuda' else False
    )

    print(f"dataset length: {len(dataset)}")
    print(f"train samples: {len(train_idx)}, test samples: {len(test_idx)}")

    model = MSTRModel(
        input_dim=40,
        num_classes=len(dataset.label_map),
        num_scales=3,
        window_size=4,
        num_heads=4
    ).to(device)
    
    checkpoint_path = './models/best_model.pth'
    backup_path = './models/backup_best_model.pth'
    if os.path.exists(checkpoint_path):
        try:
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
            print(f"loaded pre-trained model from {checkpoint_path}")
            shutil.copy(checkpoint_path, backup_path)
            print(f"backed up {checkpoint_path} to {backup_path}")
        except RuntimeError as e:
            print(f"error loading model: {e}. starting from scratch.")
    else:
        print(f"no checkpoint found at {checkpoint_path}. starting from scratch.")

    initial_test_acc = evaluate(model, test_loader, device)
    print(f"initial test acc: {initial_test_acc:.2f}%")

    class_counts = [dataset.label_counts[i] for i in range(len(dataset.label_map))]
    weights = 1. / torch.tensor(class_counts, dtype=torch.float)
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    best_test_acc = initial_test_acc
    patience = 25
    counter = 0
    os.makedirs('./models', exist_ok=True)

    for epoch in range(150):
        model.train()
        total_loss = 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(X)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        train_acc = evaluate(model, train_loader, device)
        test_acc = evaluate(model, test_loader, device)
        print(f"epoch {epoch+1}: loss = {avg_loss:.4f}, train acc = {train_acc:.2f}%, test acc = {test_acc:.2f}%")
        
        epoch_checkpoint = f'./models/checkpoint_epoch_{epoch+1}.pth'
        torch.save(model.state_dict(), epoch_checkpoint)
        print(f"saved checkpoint at {epoch_checkpoint}")
        
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            counter = 0
            torch.save(model.state_dict(), checkpoint_path)
            print(Fore.GREEN + f"saved best model with test acc: {best_test_acc:.2f}%")
        else:
            counter += 1
            if counter >= patience:
                print("early stopping!")
                break
        
        scheduler.step(avg_loss)

    plot_confusion_matrix(model, test_loader, dataset.label_map, device)