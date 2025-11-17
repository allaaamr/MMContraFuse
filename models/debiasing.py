# models/debiasing.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# -------- GRL ----------
class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_out):
        return -ctx.lam * grad_out, None

def grl(x, lam: float):
    return GradReverse.apply(x, lam)

# -------- Editor (residual) ----------
class ResidualEditor(nn.Module):
    def __init__(self, d, hidden=None, resid_scale=0.1, dropout=0.0):
        super().__init__()
        hidden = hidden or max(64, 2 * d)
        self.net = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, hidden), nn.ReLU(inplace=True),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden, d),
        )
        self.resid_scale = resid_scale
    def forward(self, z):
        return z + self.resid_scale * self.net(z)

# -------- Conditional age adversary ----------
class AgeAdversary(nn.Module):
    """
    Predicts age group a∈{0,1} from [z', onehot(y)] to enforce conditional invariance.
    """
    def __init__(self, d, n_bins, hidden=None, dropout=0.0):
        super().__init__()
        hidden = hidden or max(64, 2 * d)
        self.net = nn.Sequential(
            nn.LayerNorm(d + n_bins),
            nn.Linear(d + n_bins, hidden), nn.ReLU(inplace=True),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden, 2),
        )
    def forward(self, z_prime, y_onehot):
        return self.net(torch.cat([z_prime, y_onehot], dim=-1))

# -------- Simple cosine ramp for λ_adv ----------
class LambdaScheduler:
    def __init__(self, lam_start=0.0, lam_end=1.0, warmup_epochs=5, total_epochs=50):
        self.l0, self.l1 = lam_start, lam_end
        self.warm, self.T = warmup_epochs, max(total_epochs, 1)
    def at(self, epoch):
        if epoch <= self.warm:
            return self.l0 + (self.l1 - self.l0) * (epoch / max(1, self.warm))
        # cosine to 1
        t = min(1.0, (epoch - self.warm) / max(1, self.T - self.warm))
        return self.l1 - 0.5 * (self.l1 - self.l0) * (1 - torch.tensor(t).cos().item())
