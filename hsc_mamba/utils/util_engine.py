import copy
import torch

class ModelEMA:
    def __init__(self, model, decay=0.999, device=None):
        self.ema = copy.deepcopy(model)
        self.ema.eval()
        self.decay = decay
        self.device = device
        for param in self.ema.parameters():
            param.requires_grad = False
        if self.device:
            self.ema.to(device=device)

    def update(self, model):
        with torch.no_grad():
            msd = model.state_dict()
            esd = self.ema.state_dict()
            for name, param in msd.items():
                if name in esd:
                    if param.dtype in [torch.float32, torch.float16, torch.bfloat16, torch.float64]:
                        esd[name].mul_(self.decay).add_(param, alpha=1. - self.decay)
                    else:
                        esd[name].copy_(param)

    def swap_weights(self, model):
        msd = model.state_dict()
        esd = self.ema.state_dict()
        with torch.no_grad():
            for name, param in msd.items():
                if name in esd:
                    tmp = param.data.clone()
                    param.data.copy_(esd[name].data)
                    esd[name].data.copy_(tmp)
