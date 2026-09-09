import torch
from sklearn.metrics import accuracy_score, f1_score


@torch.no_grad()
def evaluate_f1_and_accuracy(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x)
        preds = logits.argmax(dim=1).cpu()
        all_preds.append(preds)
        all_labels.append(y)
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    f1 = f1_score(all_labels, all_preds, average="macro")
    acc = accuracy_score(all_labels, all_preds)
    return f1, acc
