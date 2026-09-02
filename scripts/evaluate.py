"""Évaluation top-1 sur le jeu de test (utilisée avec les poids EMA)."""
import torch


@torch.no_grad()
def evaluate(model, test_loader, device):
    model.eval()
    correct, total = 0, 0
    for imgs, labels in test_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    model.train()
    return correct / total
