# models/debiasing.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# -------- GRL ----------
class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        # allow lam to be python float or 0-d tensor
        ctx.lam = float(lam) if not torch.is_tensor(lam) else float(lam.item())
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

# -------- Conditional age adversary (BIN-CONDITIONED) ----------
class AgeAdversary(nn.Module):
    """
    Predicts age group a∈{0,1} from [z', onehot(y_bin)] to enforce conditional invariance
    at fixed risk bins. IMPORTANT: the condition must be exactly one-hot of length n_bins.
    """
    def __init__(self, d, n_bins, hidden=None, dropout=0.0):
        super().__init__()
        hidden = hidden or max(64, 2 * d)
        self.d = int(d)
        self.n_bins = int(n_bins)
        in_dim = self.d + self.n_bins
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden), nn.ReLU(inplace=True),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden, 2),
        )

    @torch.no_grad()
    def _validate_y_onehot(self, y_onehot):
        if y_onehot.dim() != 2:
            raise ValueError(f"[AgeAdversary] y_onehot must be 2D [B, n_bins], got shape {tuple(y_onehot.shape)}")
        if y_onehot.size(-1) != self.n_bins:
            raise ValueError(
                f"[AgeAdversary] Expected y_onehot dim={self.n_bins}, but got {y_onehot.size(-1)}. "
                f"Pass ONLY one-hot risk bins (no censorship or extras)."
            )
        # Optional: verify one-hotness (cheap check)
        mx, sm = y_onehot.max(dim=-1).values, y_onehot.sum(dim=-1)
        if not torch.all((mx == 1) & (sm == 1)):
            raise ValueError("[AgeAdversary] y_onehot must be proper one-hot (rows sum to 1 and max==1).")

    def forward(self, z_prime, y_onehot):
        # strict guard against accidental cond expansion
        self._validate_y_onehot(y_onehot)
        x = torch.cat([z_prime, y_onehot], dim=-1)
        return self.net(x)

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
        # use torch.cos for numerical stability across backends, then to float
        return self.l1 - 0.5 * (self.l1 - self.l0) * (1 - torch.tensor(t).cos().item())
