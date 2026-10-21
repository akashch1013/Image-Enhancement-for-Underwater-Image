"""Full underwater enhancement network of Shi et al. (IET Image Processing 2022), Fig. 2."""
import torch
import torch.nn as nn

from .color import BlueResidualEnhancement, GreenCompensation, RedCompensation
from .contrast import ContrastEnhancement


class UIENet(nn.Module):
    """Channel split -> Rnet / Bnet / Gnet colour correction -> I1 -> contrast enhancement (Ce).

    The flags reproduce the ablation variants of Table 4: a disabled Rnet / Gnet passes its channel
    through unchanged, and a disabled Ce returns I1 directly.
    """

    def __init__(self, width: int = 32, gamma: float = 0.7, clip_limit: float = 2.0,
                 use_rnet: bool = True, use_gnet: bool = True, use_ce: bool = True, n_blue_blocks: int = 3):
        super().__init__()
        self.config = dict(width=width, gamma=gamma, clip_limit=clip_limit, use_rnet=use_rnet,
                           use_gnet=use_gnet, use_ce=use_ce, n_blue_blocks=n_blue_blocks)
        self.rnet = RedCompensation(width) if use_rnet else nn.Identity()
        self.gnet = GreenCompensation(width) if use_gnet else nn.Identity()
        self.bnet = BlueResidualEnhancement(width, n_blue_blocks)
        self.ce = ContrastEnhancement(width, gamma, clip_limit) if use_ce else None

    def forward(self, x: torch.Tensor, return_intermediate: bool = False):
        r, g, b = x.split(1, dim=1)
        i1 = torch.cat([self.rnet(r), self.gnet(g), self.bnet(b)], dim=1).clamp(0, 1)
        extras = {}
        if self.ce is None:
            out = i1
        elif return_intermediate:
            out, extras = self.ce(i1, return_intermediate=True)
        else:
            out = self.ce(i1)
        out = out.clamp(0, 1)
        if return_intermediate:
            return dict(out=out, i1=i1, **extras)
        return out


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(path, model: UIENet, **meta) -> None:
    torch.save(dict(model=model.state_dict(), config=model.config, **meta), path)


def load_model(ckpt_path, device="cpu") -> UIENet:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = UIENet(**ckpt["config"])
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval()
