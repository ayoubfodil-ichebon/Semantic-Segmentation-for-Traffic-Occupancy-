"""
VMamba-UNet – bidirectional SSM (VSS blocks) with patch embedding.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# ─── SSM helpers ──────────────────────────────────────────
class SSM1D(nn.Module):
    def __init__(self, dim, kernel_size=31):
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=kernel_size,
                                 padding=kernel_size//2, groups=dim)
        self.gate_proj = nn.Linear(dim, dim)

    def forward(self, x):
        xc = x.transpose(1, 2)
        xc = self.dwconv(xc)
        xc = xc.transpose(1, 2)
        gate = torch.sigmoid(self.gate_proj(x))
        return gate * F.gelu(xc)

class VMambaBlock(nn.Module):
    def __init__(self, dim, expansion=2, kernel_size=31, dropout=0.0):
        super().__init__()
        hidden = dim * expansion
        self.norm1 = nn.LayerNorm(dim)
        self.in_proj = nn.Linear(dim, hidden)
        self.scan_fwd = SSM1D(hidden, kernel_size)
        self.scan_bwd = SSM1D(hidden, kernel_size)
        self.merge = nn.Linear(hidden * 2, dim)
        self.drop1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, dim), nn.Dropout(dropout)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W
        s = x.permute(0, 2, 3, 1).reshape(B, N, C)
        r = s
        s = self.norm1(s)
        h = self.in_proj(s)
        fwd = self.scan_fwd(h)
        bwd = self.scan_bwd(h.flip(1)).flip(1)
        s = self.drop1(self.merge(torch.cat([fwd, bwd], dim=-1)))
        s = r + s
        s = s + self.ffn(self.norm2(s))
        return s.reshape(B, H, W, C).permute(0, 3, 1, 2)

# ─── Patch operations ─────────────────────────────────────
class PatchPartition(nn.Module):
    def __init__(self, in_ch=3, embed_dim=96, patch_size=4):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).reshape(B, H*W, C)
        x = self.norm(x)
        return x.reshape(B, H, W, C).permute(0, 3, 1, 2)

class PatchMerging(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.norm = nn.LayerNorm(4 * in_ch)
        self.linear = nn.Linear(4 * in_ch, 2 * in_ch, bias=False)

    def forward(self, x):
        B, C, H, W = x.shape
        pad_h, pad_w = H % 2, W % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        _, _, H2, W2 = x.shape
        x = x.permute(0, 2, 3, 1)
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], dim=-1)
        B2, Hh, Ww, _ = x.shape
        x = self.norm(x.reshape(B2, -1, 4*C)).reshape(B2, Hh, Ww, -1)
        x = self.linear(x)
        return x.permute(0, 3, 1, 2)

class PatchExpanding(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.expand = nn.Linear(in_ch, 2 * in_ch, bias=False)
        self.norm = nn.LayerNorm(in_ch // 2)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).reshape(B, H*W, C)
        x = self.expand(x)
        x = x.reshape(B, H, W, 2*C)
        x = x.reshape(B, H, W, 2, 2, C // 2)
        x = x.permute(0, 1, 3, 2, 4, 5).reshape(B, 2*H, 2*W, C // 2)
        x = self.norm(x)
        return x.permute(0, 3, 1, 2)

class VSSStage(nn.Module):
    def __init__(self, dim, n_blocks=2, expansion=2, kernel_size=31, dropout=0.0):
        super().__init__()
        self.blocks = nn.Sequential(*[
            VMambaBlock(dim, expansion, kernel_size, dropout) for _ in range(n_blocks)
        ])

    def forward(self, x):
        return self.blocks(x)

# ─── VMambaUNet ───────────────────────────────────────────
class VMambaUNet(nn.Module):
    def __init__(self, n_classes, C=96, n_vss_enc=2, n_vss_bot=2, n_vss_dec=2,
                 expansion=2, kernel_size=31, dropout=0.0):
        super().__init__()
        # Encoder
        self.patch_partition = PatchPartition(in_ch=3, embed_dim=C, patch_size=4)
        self.enc_stage1 = VSSStage(C, n_vss_enc, expansion, kernel_size, dropout)
        self.patch_merge1 = PatchMerging(C)
        self.enc_stage2 = VSSStage(2*C, n_vss_enc, expansion, kernel_size, dropout)
        self.patch_merge2 = PatchMerging(2*C)
        self.enc_stage3 = VSSStage(4*C, n_vss_enc, expansion, kernel_size, dropout)
        self.patch_merge3 = PatchMerging(4*C)

        # Bottleneck
        self.bottleneck = VSSStage(8*C, n_vss_bot, expansion, kernel_size, dropout)

        # Decoder
        self.patch_expand3 = PatchExpanding(8*C)
        self.dec_stage3 = VSSStage(8*C, n_vss_dec, expansion, kernel_size, dropout)
        self.dec_reduce3 = nn.Conv2d(8*C, 4*C, 1)

        self.patch_expand2 = PatchExpanding(4*C)
        self.dec_stage2 = VSSStage(4*C, n_vss_dec, expansion, kernel_size, dropout)
        self.dec_reduce2 = nn.Conv2d(4*C, 2*C, 1)

        self.patch_expand1 = PatchExpanding(2*C)
        self.dec_stage1 = VSSStage(2*C, n_vss_dec, expansion, kernel_size, dropout)
        self.dec_reduce1 = nn.Conv2d(2*C, C, 1)

        # Head
        self.final_up = nn.ConvTranspose2d(C, C, kernel_size=4, stride=4)
        self.final_norm = nn.BatchNorm2d(C)
        self.final_proj = nn.Conv2d(C, n_classes, kernel_size=1)

    def forward(self, x):
        B, _, H, W = x.shape
        e0 = self.patch_partition(x)
        e1 = self.enc_stage1(e0)
        e1m = self.patch_merge1(e1)
        e2 = self.enc_stage2(e1m)
        e2m = self.patch_merge2(e2)
        e3 = self.enc_stage3(e2m)
        e3m = self.patch_merge3(e3)

        bot = self.bottleneck(e3m)

        d3 = self.patch_expand3(bot)
        if d3.shape[2:] != e3.shape[2:]:
            e3 = F.interpolate(e3, size=d3.shape[2:], mode='bilinear', align_corners=False)
        d3 = torch.cat([d3, e3], 1)
        d3 = self.dec_stage3(d3)
        d3 = self.dec_reduce3(d3)

        d2 = self.patch_expand2(d3)
        if d2.shape[2:] != e2.shape[2:]:
            e2 = F.interpolate(e2, size=d2.shape[2:], mode='bilinear', align_corners=False)
        d2 = torch.cat([d2, e2], 1)
        d2 = self.dec_stage2(d2)
        d2 = self.dec_reduce2(d2)

        d1 = self.patch_expand1(d2)
        if d1.shape[2:] != e1.shape[2:]:
            e1 = F.interpolate(e1, size=d1.shape[2:], mode='bilinear', align_corners=False)
        d1 = torch.cat([d1, e1], 1)
        d1 = self.dec_stage1(d1)
        d1 = self.dec_reduce1(d1)

        out = self.final_up(d1)
        out = self.final_norm(out)
        if out.shape[2:] != (H, W):
            out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=False)
        return self.final_proj(out)
