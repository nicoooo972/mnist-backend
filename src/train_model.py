#!/usr/bin/env python3

import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
import os

# Import du modèle
from .models.convnet import ConvNet

def train_model(model, train_loader, device, perm, n_epoch=10):
    perm = perm.to(device)
    model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters())

    for epoch in range(n_epoch):
        print(f"--- Epoch {epoch+1}/{n_epoch} ---")
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            batch_size = data.shape[0]
            data_flattened = data.view(batch_size, -1)
            data_permuted = data_flattened[:, perm]
            data_reshaped = data_permuted.view(batch_size, 1, 28, 28)
            
            optimizer.zero_grad()
            logits = model(data_reshaped)
            loss = F.cross_entropy(logits, target)
            loss.backward()
            optimizer.step()

            if batch_idx % 100 == 0:
                print(f"  Batch: {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")

def test_model(model, test_loader, device, perm):
    perm = perm.to(device)
    model.to(device)
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            batch_size = data.shape[0]
            data_flattened = data.view(batch_size, -1)
            data_permuted = data_flattened[:, perm]
            data_reshaped = data_permuted.view(batch_size, 1, 28, 28)
            
            logits = model(data_reshaped)
            test_loss += F.cross_entropy(logits, target, reduction='sum').item()
            pred = torch.argmax(logits, dim=1)
            correct += pred.eq(target.view_as(pred)).sum().item()
        
        test_loss /= len(test_loader.dataset)
        accuracy = 100. * correct / len(test_loader.dataset)
        print(f'Test: Loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)')
        return test_loss, accuracy

def main():
    # Configuration du device
    device = torch.device("cpu")  # Forcer CPU pour éviter les erreurs CUDA
    print(f"Using device: {device}")

    # Chargement des données
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST("data/raw", download=True, train=True, transform=tf),
        batch_size=64, shuffle=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST("data/raw", download=True, train=False, transform=tf),
        batch_size=64, shuffle=True
    )

    # Paramètres du modèle
    n_kernels = 6
    input_size = 1
    output_size = 10

    # Création d'une permutation fixe
    perm = torch.randperm(784)
    print(f"Permutation created: {perm[:10]}...")

    # Création et entraînement du modèle
    convnet = ConvNet(input_size, n_kernels, output_size)
    convnet.to(device)
    print(f"Parameters: {sum(p.numel() for p in convnet.parameters())/1e3:.1f}K")
    
    # Entraînement
    train_model(convnet, train_loader, device, perm)
    
    # Test
    test_model(convnet, test_loader, device, perm)

    # Sauvegarde du modèle avec la permutation
    os.makedirs("models", exist_ok=True)
    
    model_data = {
        'model_state_dict': convnet.state_dict(),
        'permutation': perm,
        'n_kernels': n_kernels,
        'input_size': input_size,
        'output_size': output_size
    }
    torch.save(model_data, "models/convnet.pt")
    print("✅ Modèle sauvegardé avec succès dans models/convnet.pt")

if __name__ == "__main__":
    main()