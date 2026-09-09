"""Exponential Moving Average des poids du modèle, utilisée pour l'évaluation."""
import torch


class EMA:
    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        state = model.state_dict()
        for k, v in self.shadow.items():
            if v.dtype.is_floating_point:
                v.mul_(self.decay).add_(state[k].detach(), alpha=1 - self.decay)
            else:
                v.copy_(state[k].detach())  # ex. num_batches_tracked : compteur entier, pas de moyenne

    def copy_to(self, model):
        model.load_state_dict(self.shadow, strict=True)
