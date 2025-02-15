import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import json
import random
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import os

class AttentionLayer(nn.Module):
    def __init__(self, d):
        super(AttentionLayer, self).__init__()
        self.query = nn.Linear(d, d)
        self.key = nn.Linear(d, d)
        self.value = nn.Linear(d, d)
        self.softmax = nn.Softmax(dim=-1)
        self.scale = np.sqrt(d)
    
    def forward(self, x_i, x_j):
        """
        x_i: Tensor of shape (batch_size, d)
        x_j: Tensor of shape (batch_size, d)
        """
        Q = self.query(x_i)  # (batch_size, d)
        K = self.key(x_j)    # (batch_size, d)
        V = self.value(x_j)  # (batch_size, d)
        
        attn_scores = torch.sum(Q * K, dim=-1, keepdim=True) / self.scale  # (batch_size, 1)
        attn_weights = self.softmax(attn_scores)  # (batch_size, 1)
        
        attended = attn_weights * V  # (batch_size, d)
        
        return attended

class LinearProbeWithAttention(nn.Module):
    def __init__(self, d):
        super(LinearProbeWithAttention, self).__init__()
        self.attention = AttentionLayer(d)
        self.linear = nn.Linear(2 * d, 2)  # Binary classification (2 classes)
    
    def forward(self, x_i, x_j):
        attended_i = self.attention(x_i, x_j)  # (batch_size, d)
        attended_j = self.attention(x_j, x_i)  # (batch_size, d)
        
        z = torch.cat([attended_i, attended_j], dim=-1)  # (batch_size, 2d)
        logits = self.linear(z)  # (batch_size, 2)
        
        return logits

class LinearProbe(nn.Module):
    def __init__(self, d):
        super(LinearProbe, self).__init__()
        self.linear = nn.Linear(2 * d, 2)  # Binary classification (2 classes)

    def forward(self, x_i, x_j):
        z = torch.cat([x_i, x_j], dim=-1)  # (batch_size, 2d)
        return self.linear(z)  # (batch_size, 2)

def save_checkpoint(model, optimizer, epoch, best_val_accuracy, file_path):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_accuracy': best_val_accuracy
    }
    torch.save(checkpoint, file_path)

def load_model(model_class, d, file_path, load_entire_model=False):
    if load_entire_model:
        model = torch.load(file_path)
    else:
        model = model_class(d)
        model.load_state_dict(torch.load(file_path))
    model.eval()
    return model

class PairDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.N = X.size(0)
        self.pairs = []
        self.labels = []
        self._prepare_pairs()
    
    def _prepare_pairs(self):
        self.pairs = []
        self.labels = []
        for i in range(self.N):
            for j in range(i + 1, self.N):
                self.pairs.append((i, j))
                self.labels.append(1 if self.Y[i] == self.Y[j] else 0)
        self.pairs = torch.tensor(self.pairs, dtype=torch.long)
        self.labels = torch.tensor(self.labels, dtype=torch.long)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        i, j = self.pairs[idx]
        return self.X[i], self.X[j], self.labels[idx]

def train_random_model(train_loader, val_loader, model, device, epochs=10, lr=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in tqdm(range(epochs), desc="Sanity Check Epochs"):
        F = 0  # Cumulative loss for tracking
        for batch in train_loader:
            x_i, x_j, _ = batch
            x_i = x_i.to(device)
            x_j = x_j.to(device)

            label = torch.randint(0, 2, (x_i.size(0),)).to(device)
            outputs = model(x_i, x_j)
            loss = criterion(outputs, label)
            
            F += loss.item() * x_i.size(0)
                
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        avg_loss = F / len(train_loader.dataset)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

        val_accuracy, val_loss = eval_model(val_loader, model, device, criterion)
    print("Training finished")

def train_model(train_loader, val_loader, model, device, save_name, epochs=10, lr=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses, train_accuracies = [], []
    val_losses, val_accuracies = [], []

    best_val_accuracy = 0.0  # Initialize best validation accuracy
    best_model_path = f"best_model_{save_name}.pth"  # Define checkpoint path

    for epoch in tqdm(range(epochs), desc="Training Epochs"):
        model.train()
        F = 0  # Cumulative loss
        correct = 0
        total = 0

        for batch in train_loader:
            x_i, x_j, labels = batch
            x_i = x_i.to(device)
            x_j = x_j.to(device)
            labels = labels.to(device)

            outputs = model(x_i, x_j)
            loss = criterion(outputs, labels)

            F += loss.item() * x_i.size(0)

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        avg_loss = F / len(train_loader.dataset)
        accuracy = correct / total

        train_losses.append(avg_loss)
        train_accuracies.append(accuracy)

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")

        val_accuracy, val_loss = eval_model(val_loader, model, device, criterion)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        if val_accuracy >= best_val_accuracy:
            best_val_accuracy = val_accuracy
            save_checkpoint(model, optimizer, epoch, best_val_accuracy, best_model_path)
            print(f"New best model saved with validation accuracy: {best_val_accuracy:.4f}")

    plot_and_save_metrics(
        train_losses, val_losses, train_accuracies, val_accuracies, save_path=save_name
    )
    print("Training finished")
    print(f"Best Validation Accuracy: {best_val_accuracy:.4f}")
    print(f"Best model saved at: {best_model_path}")

def eval_model(dataloader, model, device, criterion):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0

    with torch.no_grad():
        for batch in dataloader:
            x_i, x_j, labels = batch
            x_i = x_i.to(device)
            x_j = x_j.to(device)
            labels = labels.to(device)

            outputs = model(x_i, x_j)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * x_i.size(0)

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    avg_loss = total_loss / total

    print(
        f"Evaluation Results - Accuracy = {accuracy:.4f}, Avg Loss = {avg_loss:.4f}"
    )

    return accuracy, avg_loss

def plot_and_save_metrics(train_losses, val_losses, train_accuracies, val_accuracies, save_path):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label="Training Loss", color="tab:blue")
    plt.plot(epochs, val_losses, label="Validation Loss", color="tab:orange")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss over Epochs")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label="Training Accuracy", color="tab:green")
    plt.plot(epochs, val_accuracies, label="Validation Accuracy", color="tab:red")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Accuracy over Epochs")
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"{save_path}_metrics.png")
    plt.close()

def main(args):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")
    
    random.seed(args.seed)
    np.random.seed(args.seed) 
    torch.manual_seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(args.seed)

    with open(args.data, "r") as f:
        data = json.load(f)
    
    key = "hidden_state" if "hidden_state" in data[0] else "last_hidden_state"
    X = [torch.tensor(item[key], dtype=torch.float32) for item in data]
    
    if "acc" in data[0]:
        Y = [int(item["acc"]) for item in data]
    elif "accuracy" in data[0]:
        Y = [int(item["accuracy"]) for item in data]
    else:
        Y = [1 if int(item["gt_label"]) == int(item["prediction"]) else 0 for item in data]
    
    X = torch.stack(X)
    Y = torch.tensor(Y, dtype=torch.long)

    if Y.dim() > 1:
        Y = Y.view(-1)
        print(f"Reshaped Y shape: {Y.shape}")

    if X.dim() == 3:
        X = X.reshape(X.size(0), -1)
    
    d = X.size(-1)
    permutation = torch.randperm(X.size(0))
    X = X[permutation]
    Y = Y[permutation]
    
    dataset = PairDataset(X, Y)
    split_idx = int(0.9 * len(dataset))
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [split_idx, len(dataset) - split_idx])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    print(f"Training size: {len(train_dataset)}")
    print(f"Validation size: {len(val_dataset)}")
    
    if args.lrs:
        learning_rates = args.lrs
    else:
        learning_rates = [args.lr]
    
    overall_best_val_accuracy = 0.0
    overall_best_lr = None
    overall_best_model_path = None
    os.makedirs("lr_search_models", exist_ok=True)

    for lr in learning_rates:
        print(f"\nTraining with Learning Rate: {lr}")

        model = (LinearProbeWithAttention(d) if args.use_attention else LinearProbe(d)).to(device)
        save_name = f"{args.seed}_lr_{lr}_epochs_{args.epochs}"
        
        train_model(train_loader, val_loader, model, device, save_name, epochs=args.epochs, lr=lr)
        best_model_path = f"best_model_{save_name}.pth"
        
        checkpoint = torch.load(best_model_path)
        best_val_accuracy = checkpoint['best_val_accuracy']
        print(f"Validation Accuracy for LR {lr}: {best_val_accuracy:.4f}")

        if best_val_accuracy > overall_best_val_accuracy:
            overall_best_val_accuracy = best_val_accuracy
            overall_best_lr = lr
            overall_best_model_path = best_model_path

    print("\nSanity Check:")
    train_random_model(train_loader, val_loader, model, device, epochs=args.epochs, lr=overall_best_lr)
    print("\nSanity Check completed.")

    print("\nLearning Rate Search Completed")
    if overall_best_lr is not None:
        print(f"Best Learning Rate: {overall_best_lr} with Validation Accuracy: {overall_best_val_accuracy:.4f}")
        print(f"Best model saved at: {overall_best_model_path}")
    else:
        print("No valid learning rate found.")

def parse_arguments():
    parser = argparse.ArgumentParser(description="Train or load a linear probe model with attention")

    parser.add_argument('--seed', type=int, required=True, help="Random seed for reproducibility")
    parser.add_argument('--data', type=str, required=True, help="Path to the JSON data file")
    
    parser.add_argument('--lrs', type=float, nargs='+', default=[0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001], help="List of learning rates to try")
    parser.add_argument('--lr', type=float, default=0.001, help="Single learning rate for training if --lrs is not provided")
    
    parser.add_argument('--epochs', type=int, default=10, help="Number of epochs for training")
    parser.add_argument('--batch_size', type=int, default=512, help="Batch size for training and evaluation")
    
    parser.add_argument('--use_attention', action='store_true', help="Use attention mechanism in the model")

    parser.add_argument('--save_path', type=str, default="linear_probe.pth", 
                        help="Path to save the model")
    parser.add_argument('--load_path', type=str, default="linear_probe.pth", 
                        help="Path to load the model")
    parser.add_argument('--save_entire_model', action='store_true', 
                        help="Flag to save the entire model (architecture + weights)")
    parser.add_argument('--load_entire_model', action='store_true', 
                        help="Flag to load the entire model (architecture + weights)")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    main(args)