"""Exponential Moving Average des poids du modèle, utilisée pour l'évaluation."""
import torch


class EMA:
    def __init__(self, model, decay):
        self.decay = decay
        state = model.state_dict()
        # Séparés une seule fois (les clés d'un state_dict sont stables pendant tout l'entraînement) :
        # les tenseurs flottants reçoivent l'EMA, les autres (ex. num_batches_tracked, un compteur
        # entier) sont simplement recopiés tels quels, comme dans la version précédente.
        self._float_keys = [k for k, v in state.items() if v.dtype.is_floating_point]
        self._other_keys = [k for k, v in state.items() if not v.dtype.is_floating_point]
        self.shadow = {k: v.detach().clone() for k, v in state.items()}
        self._shadow_float = [self.shadow[k] for k in self._float_keys]

    @torch.no_grad()
    def update(self, model):
        """Équivalent numérique de la version précédente, mais via torch._foreach_* (même technique
        que torch.optim en interne) au lieu d'une boucle Python + un appel .mul_()/.add_() par
        tenseur : un WideResNet-28-2 a 100+ tenseurs dans son state_dict (poids de conv +
        running_mean/running_var/num_batches_tracked par BatchNorm), donc 100+ lancements de kernel
        séparés par itération sans ce regroupement -- un coût fixe qui ne profite pas de
        torch.compile (qui ne couvre que le forward du modèle) et qui pèse proportionnellement plus
        depuis que le reste de l'itération est accéléré.
        """
        state = model.state_dict()
        model_float = [state[k].detach() for k in self._float_keys]
        torch._foreach_mul_(self._shadow_float, self.decay)
        torch._foreach_add_(self._shadow_float, model_float, alpha=1 - self.decay)
        for k in self._other_keys:
            self.shadow[k].copy_(state[k].detach())

    def copy_to(self, model):
        model.load_state_dict(self.shadow, strict=True)
