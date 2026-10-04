"""
Global (whole-animal) representations: MegaDescriptor (fine-tuned) and MiewID (pre-trained).
Proposal: compare the two on the same protocol, pick the stronger one for the fusion.

Main changes versus the skeleton (reasons in DECISIONS.md #1, #2, #5):
 * Variant is selectable (L-384 for the real runs; T-224 / B-224 for debugging on a small GPU/CPU).
   The skeleton mislabelled the variant "T-224" while loading L-384, and config said 2048-d;
   Swin-L is 1536-d.
 * `init="megadescriptor"` or `"imagenet"`. The public MegaDescriptor weights were trained on
   ~80% of the images of the 29 datasets that are now INSIDE WildlifeReID-10k, so they have very
   likely seen many of our test images. `init="imagenet"` is the leakage-free option.
 * MiewID wrapper with the same interface, so both plug into the same extraction / eval code.
 * Fine-tuning: warm-up + cosine per step, higher lr for the fresh ArcFace head, gradient
   clipping, weight decay, optional gradient checkpointing, periodic validation with best-checkpoint saving.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from timm.data import resolve_data_config

from src.data_processing.augmentations import eval_transform, train_transform

VARIANTS = {   # MegaDescriptor flavours; ImageNet tags = the Swin models they started from
    "L-384": dict(hf="hf-hub:BVRA/MegaDescriptor-L-384", dim=1536,
                  imagenet="swin_large_patch4_window12_384.ms_in22k_ft_in1k"),
    "B-224": dict(hf="hf-hub:BVRA/MegaDescriptor-B-224", dim=1024,
                  imagenet="swin_base_patch4_window7_224.ms_in22k_ft_in1k"),
    "T-224": dict(hf="hf-hub:BVRA/MegaDescriptor-T-224", dim=768,
                  imagenet="swin_tiny_patch4_window7_224.ms_in22k_ft_in1k"),
}   


class _EmbeddingModel(nn.Module):
    """Common interface used by extraction / evaluation code."""
    embedding_dim: int
    input_size: int
    mean: tuple
    std: tuple

    def eval_transform(self):
        return eval_transform(self.input_size, self.mean, self.std)

    def train_transform(self, aug_cfg):
        return train_transform(self.input_size, self.mean, self.std, aug_cfg)

    @torch.no_grad()
    def embed(self, x):
        return F.normalize(self.forward(x).float(), dim=1)      # L2-normalised -> cosine = dot product

# TODO: Re-check (MegaDescriptor was trained on wildlife dataset)
class GlobalModel(_EmbeddingModel):
    """MegaDescriptor (Swin) backbone producing a pooled embedding (no classifier)."""

    def __init__(self, variant="L-384", init="megadescriptor", grad_checkpointing=False):
        super().__init__()
        spec = VARIANTS[variant]
        self.variant, self.init = variant, init
        name = spec["hf"] if init == "megadescriptor" else spec["imagenet"]
        self.backbone = timm.create_model(name, pretrained=True, num_classes=0)
        if grad_checkpointing and hasattr(self.backbone, "set_grad_checkpointing"):
            self.backbone.set_grad_checkpointing(True)
        self.embedding_dim = self.backbone.num_features
        cfg = resolve_data_config({}, model=self.backbone)
        self.data_config = cfg
        self.input_size = int(cfg["input_size"][-1])
        self.mean, self.std = tuple(cfg["mean"]), tuple(cfg["std"])

    def forward(self, x):
        return self.backbone(x)

    def save(self, path):
        torch.save({"variant": self.variant, "init": self.init, "backbone": self.backbone.state_dict()}, path)

    def load(self, path):
        self.backbone.load_state_dict(torch.load(path, map_location="cpu")["backbone"], strict=True)


class MiewIDModel(_EmbeddingModel):
    """
    MiewID (EfficientNetV2 + angular-margin loss), loaded from the Hugging Face hub.
    NOTE: I could not reach the hub from my sandbox. Input size / normalisation below follow my
    recollection of the 'miewid-msv3' card - CHECK the model card before trusting them.
    The embedding size is measured with a dummy forward pass instead of hard-coded.
    """
    HF_ID = "conservationxlabs/miewid-msv3"

    def __init__(self, hf_id=None, input_size=440):
        super().__init__()
        from transformers import AutoModel
        self.backbone = AutoModel.from_pretrained(hf_id or self.HF_ID, trust_remote_code=True)
        self.input_size = input_size
        self.mean, self.std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        with torch.no_grad():
            self.backbone.eval()
            self.embedding_dim = self.backbone(torch.zeros(1, 3, input_size, input_size)).shape[-1]

    def forward(self, x):
        return self.backbone(x)


class ArcFaceHead(nn.Module):
    """Additive angular margin (Deng et al. 2019). Logits = s * cos(theta + m) for the true class."""

    def __init__(self, embedding_dim, num_classes, scale=64.0, margin=0.5):
        super().__init__()
        self.scale = scale
        self.weight = nn.Parameter(torch.empty(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)
        self.cos_m, self.sin_m = math.cos(margin), math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(self, embeddings, labels):
        cosine = F.linear(F.normalize(embeddings.float()), F.normalize(self.weight.float()))
        sine = torch.sqrt((1.0 - cosine ** 2).clamp(1e-7, 1.0))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)     # keep monotonic when theta+m > pi
        one_hot = F.one_hot(labels, num_classes=cosine.size(1)).bool()
        return self.scale * torch.where(one_hot, phi, cosine)


def _device(device):
    return torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def extract_features(model, loader, device=None, amp=True):
    """Returns (features [N,d] float32 CPU, extras) where extras = [labels, row_idx] in loader order."""
    dev = _device(device)
    model.eval().to(dev)
    feats, extras = [], None
    for images, *rest in loader:
        with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=amp and dev.type == "cuda"):
            emb = model.embed(images.to(dev, non_blocking=True))
        feats.append(emb.float().cpu())
        if extras is None:
            extras = [[] for _ in rest]
        for bucket, field in zip(extras, rest):
            bucket.extend(field.tolist() if isinstance(field, torch.Tensor) else list(field))
    return torch.cat(feats), tuple(extras or [])


def fine_tune(model, train_loader, *, num_classes, epochs=12, lr=1e-3, head_lr_mult=10.0,
              weight_decay=1e-4, warmup_epochs=1, arcface_margin=0.5, arcface_scale=64.0,
              accum_steps=8, grad_clip=5.0, amp=True, device=None, log_every=50,
              eval_fn=None, eval_every=2, ckpt_path=None):
    """
    ArcFace fine-tuning. `train_loader` yields (images, label_idx, row_idx) with label_idx in [0, num_classes).
    `eval_fn(model) -> float` (higher is better, e.g. validation open-set NBA) is called every
    `eval_every` epochs; the best backbone is saved to `ckpt_path`.
    """
    dev = _device(device)
    model.to(dev)
    head = ArcFaceHead(model.embedding_dim, num_classes, arcface_scale, arcface_margin).to(dev)
    opt = torch.optim.SGD([{"params": model.parameters(), "lr": lr},
                           {"params": head.parameters(), "lr": lr * head_lr_mult}],
                          momentum=0.9, weight_decay=weight_decay)
    steps_per_epoch = max(1, len(train_loader) // accum_steps)
    total, warm = epochs * steps_per_epoch, warmup_epochs * steps_per_epoch
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    use_amp = amp and dev.type == "cuda"
    scaler = torch.amp.GradScaler(dev.type, enabled=use_amp)
    best = -1.0

    for epoch in range(epochs):
        model.train(); head.train()
        opt.zero_grad(set_to_none=True)
        run, n = 0.0, 0
        for step, (images, targets, _) in enumerate(train_loader):
            images, targets = images.to(dev, non_blocking=True), targets.to(dev)
            with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=use_amp):
                emb = model(images)
            loss = F.cross_entropy(head(emb, targets), targets)           # head in fp32
            scaler.scale(loss / accum_steps).backward()
            if (step + 1) % accum_steps == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(head.parameters()), grad_clip)
                scaler.step(opt); scaler.update(); sched.step()
                opt.zero_grad(set_to_none=True)
            run += loss.item(); n += 1
            if log_every and (step + 1) % log_every == 0:
                print(f"epoch {epoch+1}/{epochs} step {step+1}: loss {run/n:.4f}")
        print(f"epoch {epoch+1}/{epochs} done - mean loss {run/max(n,1):.4f}, lr {sched.get_last_lr()[0]:.2e}")
        if eval_fn is not None and ((epoch + 1) % eval_every == 0 or epoch + 1 == epochs):
            score = eval_fn(model)
            print(f"  validation score: {score:.4f} (best {max(best, score):.4f})")
            if score > best and ckpt_path:
                best = score; model.save(ckpt_path)
            best = max(best, score)
    return head
