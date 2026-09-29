"""Exponential moving average of the model weights, which is what gets evaluated."""
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
                v.copy_(state[k].detach())  # e.g. num_batches_tracked: an integer counter, not something to average

    def copy_to(self, model):
        model.load_state_dict(self.shadow, strict=True)
