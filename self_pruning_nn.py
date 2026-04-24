import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================================================
# Part 1: Custom Prunable Linear Layer [cite: 69]
# =========================================================
class PrunableLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        # Standard weight and bias [cite: 71]
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        
        # gate_scores parameter for learnable pruning [cite: 72, 73]
        self.gate_scores = nn.Parameter(torch.Tensor(out_features, in_features))
        
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        nn.init.constant_(self.bias, 0)
        # Initialize gates near 0.5 (Sigmoid(0) = 0.5)
        nn.init.constant_(self.gate_scores, 0.0)

    def forward(self, x):
        # Apply Sigmoid to gate scores [cite: 77]
        gates = torch.sigmoid(self.gate_scores)
        # Element-wise multiplication for pruned weights [cite: 79]
        pruned_weights = self.weight * gates
        return F.linear(x, pruned_weights, self.bias)

# =========================
# Part 2: Neural Network [cite: 111]
# =========================
class PrunableNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = PrunableLinear(32 * 32 * 3, 512)
        self.fc2 = PrunableLinear(512, 256)
        self.fc3 = PrunableLinear(256, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# =========================
# Part 3: Dataset (CIFAR-10) [cite: 107]
# =========================
# Added normalization for better convergence on CIFAR-10
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

trainset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

# =========================================================
# Part 4: Sparsity & Evaluation Functions [cite: 101, 103]
# =========================================================
def compute_sparsity(model, threshold=1e-2):
    total, pruned = 0, 0
    for layer in model.modules():
        if isinstance(layer, PrunableLinear):
            gates = torch.sigmoid(layer.gate_scores)
            total += gates.numel()
            pruned += (gates < threshold).sum().item()
    return (pruned / total) * 100

def evaluate(model):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in testloader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            _, preds = torch.max(outputs, 1)
            total += y.size(0)
            correct += (preds == y).sum().item()
    return 100 * correct / total

# =========================================================
# Part 5: Training Function [cite: 98]
# =========================================================
def train_model(lambda_val, epochs=10):
    model = PrunableNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        loop = tqdm(trainloader, leave=False)
        for x, y in loop:
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            outputs = model(x)
            
            # Classification Loss [cite: 84]
            class_loss = criterion(outputs, y)
            
            # Sparsity Regularization (L1 Norm of Sigmoid Gates) [cite: 87, 89]
            sparsity_loss = 0
            for layer in model.modules():
                if isinstance(layer, PrunableLinear):
                    sparsity_loss += torch.sigmoid(layer.gate_scores).sum()
            
            # Total Loss formulation [cite: 87]
            total_loss = class_loss + lambda_val * sparsity_loss
            
            total_loss.backward()
            optimizer.step()
            
            loop.set_description(f"λ={lambda_val} Epoch {epoch+1}")
            loop.set_postfix(loss=total_loss.item())

    accuracy = evaluate(model)
    sparsity = compute_sparsity(model)
    return model, accuracy, sparsity

# =========================
# Main Execution 
# =========================
if __name__ == "__main__":
    # Range of lambda values to show trade-off [cite: 104]
    lambda_values = [1e-6, 1e-5, 5e-5] 
    results = []
    best_model = None

    print(f"{'Lambda':<10} | {'Accuracy (%)':<15} | {'Sparsity (%)':<15}")
    print("-" * 45)

    for lam in lambda_values:
        model, acc, sp = train_model(lam)
        results.append((lam, acc, sp))
        print(f"{lam:<10} | {acc:<15.2f} | {sp:<15.2f}")
        best_model = model # Using the last trained model for visualization

    # Histogram Visualization [cite: 117, 118]
    all_gates = []
    for layer in best_model.modules():
        if isinstance(layer, PrunableLinear):
            gates = torch.sigmoid(layer.gate_scores).detach().cpu().numpy()
            all_gates.extend(gates.flatten())

    plt.figure(figsize=(10, 6))
    plt.hist(all_gates, bins=50, color='teal', alpha=0.7)
    plt.title("Distribution of Final Gate Values")
    plt.xlabel("Gate Value (Sigmoid Output)")
    plt.ylabel("Frequency")
    plt.grid(axis='y', alpha=0.3)
    plt.savefig("gate_distribution.png")
    plt.show()
