import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# Part 1: The "Prunable" Linear Layer [cite: 69]
class PrunableLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(PrunableLinear, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        # Initialize gate_scores to 0.5 so gates start at sigmoid(0.5) ≈ 0.62
        self.gate_scores = nn.Parameter(torch.Tensor(out_features, in_features))
        
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        nn.init.constant_(self.bias, 0)
        nn.init.constant_(self.gate_scores, 0.5) 

    def forward(self, x):
        # Apply Sigmoid to turn scores into [0, 1] gates [cite: 77]
        gates = torch.sigmoid(self.gate_scores)
        # pruned_weights = weight * gates [cite: 79]
        pruned_weights = self.weight * gates
        return nn.functional.linear(x, pruned_weights, self.bias)

# Part 2: Prunable Network Definition [cite: 111]
class PrunableNet(nn.Module):
    def __init__(self):
        super(PrunableNet, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = PrunableLinear(28*28, 256)
        self.fc2 = PrunableLinear(256, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        return self.fc2(x)

    def get_sparsity_loss(self):
        # L1 norm of all gate values [cite: 89, 91]
        return torch.sum(torch.sigmoid(self.fc1.gate_scores)) + \
               torch.sum(torch.sigmoid(self.fc2.gate_scores))

# Part 3: Training and Evaluation Loop [cite: 112]
def run_experiment(lambda_val):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False)

    model = PrunableNet().to(device)
    # Balanced Learning Rates
    optimizer = optim.Adam([
        {'params': [model.fc1.weight, model.fc2.weight, model.fc1.bias, model.fc2.bias], 'lr': 1e-3},
        {'params': [model.fc1.gate_scores, model.fc2.gate_scores], 'lr': 1e-2} 
    ])
    criterion = nn.CrossEntropyLoss()

    for epoch in range(5):
        model.train()
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            # Total Loss = ClassificationLoss + Lambda * SparsityLoss [cite: 87]
            loss = criterion(model(images), labels) + lambda_val * model.get_sparsity_loss()
            loss.backward()
            optimizer.step()

    model.eval()
    correct, total, p_count, t_count = 0, 0, 0, 0
    all_gates = []
    with torch.no_grad():
        # Evaluate Accuracy [cite: 103]
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        # Calculate Sparsity (threshold 1e-2) 
        for layer in [model.fc1, model.fc2]:
            g = torch.sigmoid(layer.gate_scores)
            all_gates.append(g.cpu().numpy().flatten())
            p_count += (g < 0.01).sum().item()
            t_count += g.numel()
            
    accuracy = 100 * correct / total
    sparsity = (p_count / t_count) * 100
    return accuracy, sparsity, np.concatenate(all_gates)

# Run for 3 Lambda values [cite: 104, 116]
lambdas = [1e-5, 2e-4, 1e-3]
results = []

print(f"{'Lambda':<10} | {'Test Accuracy (%)':<20} | {'Sparsity (%)':<15}")
print("-" * 55)

for l in lambdas:
    acc, sp, dist = run_experiment(l)
    results.append((l, acc, sp))
    print(f"{l:<10} | {acc:<20.2f} | {sp:<15.2f}")
    if l == 2e-4: # Histogram for the balanced model [cite: 117]
        best_dist = dist

# Plotting the gate distribution [cite: 117]
plt.figure(figsize=(8, 5))
plt.hist(best_dist, bins=50, color='royalblue', edgecolor='black')
plt.title("Final Gate Value Distribution (Middle Lambda)")
plt.xlabel("Gate Value (Sigmoid Output)")
plt.ylabel("Frequency")
plt.show()
