import torch
from copy import deepcopy


class ModelEma(torch.nn.Module):
    def __init__(self, model, decay=0.9999, device=None):
        super().__init__()
        # Create a disconnected copy of the model
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.device = device

        if self.device is not None:
            self.module.to(device=device)

    def _update(self, model, update_fn):
        with torch.no_grad():
            # Zip iterates over both models' parameters simultaneously
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                # Apply the update function (decay logic)
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        # Standard EMA formula: New_Shadow = Decay * Old_Shadow + (1-Decay) * Current_Weights
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def set(self, model):
        # Force copy weights (useful for initialization)
        self._update(model, update_fn=lambda e, m: m)