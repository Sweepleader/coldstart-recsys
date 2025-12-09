import torch
import torch.nn as nn

class SoftmaxCSRTower(nn.Module):
    def __init__(self, dim=128, temperature=1.0, out_norm=True):
        super().__init__()
        self.router = nn.Linear(dim * 2, 2)
        self.fc = nn.Linear(dim, dim)
        self.use_norm = out_norm
        self.norm = nn.LayerNorm(dim)
        self.temperature = float(temperature)

    def forward(self, content_vec, cf_vec):
        fusion = torch.cat([content_vec, cf_vec], dim=-1)
        w = torch.softmax(self.router(fusion) / max(self.temperature, 1e-6), dim=-1)
        out = w[:, :1] * content_vec + w[:, 1:] * cf_vec
        if self.use_norm:
            out = self.norm(out)
        out = self.fc(out)
        return out

class SigmoidCSRTower(nn.Module):
    def __init__(self, dim=128, out_norm=False):
        super().__init__()
        self.gate = nn.Linear(dim * 2, dim)
        self.fc = nn.Linear(dim, dim)
        self.use_norm = out_norm
        self.norm = nn.LayerNorm(dim)

    def forward(self, content_vec, cf_vec):
        fusion = torch.cat([content_vec, cf_vec], dim=-1)
        gate = torch.sigmoid(self.gate(fusion))
        out = gate * content_vec + (1 - gate) * cf_vec
        if self.use_norm:
            out = self.norm(out)
        out = self.fc(out)
        return out
