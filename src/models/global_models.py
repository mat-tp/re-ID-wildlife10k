"""
This file implements global representations that capture broad characteristics, 
such as body shape and morphology of individual animals.

Models to be implemented are:
1. MiewID: Multi-species individual identification model
   - Based on EfficientNetV2 backbone
   - Angular-margin training objective
   - Multi-species training capabilities
   
2. MegaDescriptor: Wildlife-specific foundation model
   - Fine-tuned on WildlifeReID-10k training data
   - ArcFace-style angular-margin objective
   - Strong baseline for individual animal Re-ID

Reference: 
- MiewID: "Multi-species animal Re-ID using a large community-curated dataset"
- MegaDescriptor: WildlifeDatasets framework

TODO: Implement MiewID model
TODO: Implement MegaDescriptor model with ArcFace loss
TODO: Add fine-tuning functionality for MegaDescriptor
TODO: Implement feature extraction for global descriptors
TODO: Add model comparison functionality
"""

import math
from typing import List, Tuple, Dict, Any
import timm
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from torch.utils.data import DataLoader, Dataset

# Helper function to convert images to PIL format
def to_pil(image):
   if isinstance(image, Image.Image):
      return image.convert("RGB")
   elif isinstance(image, np.ndarray):
      return Image.fromarray(image)
   raise TypeError(f"Unsupported image type: {type(image)}. Expected PIL.Image or np.ndarray.")


# MegaDescriptor model implementation
class GlobalModel(nn.Module):
   def __init__(self):

      super().__init__()

      self.variant = "T-224"
      spec = {"hf_id": "hf-hub:BVRA/MegaDescriptor-L-384", "embedding_dim": 1536}
      self.backbone = timm.create_model(spec["hf_id"], pretrained=True)
      self.embedding_dim = self.backbone.num_features

      self.data_config = resolve_data_config({}, model=self.backbone)
      self.input_size = int(self.data_config["input_size"][-1])


   # Preprocess input image to match model requirements
   def eval_transform(self):
      return transforms.Compose([
         transforms.Lambda(to_pil),
         transforms.Resize((self.input_size, self.input_size)),
         transforms.ToTensor(),
         transforms.Normalize(self.data_config["mean"], std=self.data_config["std"]),
      ])

   # Extract embeddings from input images
   def train_transform(self):

      return transforms.Compose([
         transforms.Lambda(to_pil),
         transforms.Resize((self.input_size, self.input_size)),
         transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1),
         transforms.ToTensor(),
         transforms.Normalize(self.data_config["mean"], std=self.data_config["std"]),
      ])

   def forward(self, x):
      return self.backbone(x)

   def embed(self, x):
      with torch.no_grad():
         return F.normalize(self.forward(x).float(), dim=1)  # L2 normalization of embeddings

   # saving the embeddings
   def save(self, path):
      torch.save({"variant": self.variant, "backbone": self.backbone.state_dict()}, path)

   def load(self, path):
      state = torch.load(path, map_location="cpu")
      self.backbone.load_state_dict(state["backbone"], strict=True)

# Using ArcFace for Fine-tuning the model
class ArcFaceHead(nn.Module):
   def __init__(self, embedding_dim, num_classes, scale = 64.0, margin=0.5):

      super().__init__()
      self.scale = scale
      self.weight = nn.Parameter(torch.empty(num_classes, embedding_dim))
      nn.init.xavier_uniform_(self.weight)
      self.cos_m = math.cos(margin)
      self.sin_m = math.sin(margin)

      self.th = math.cos(math.pi - margin)
      self.mm = math.sin(math.pi - margin) * margin

   def forward(self, embeddings, labels):

      cosine = F.linear(F.normalize(embeddings.float()), F.normalize(self.weight.float()))
      sine = torch.sqrt((1.0 - cosine ** 2).clamp(1e-7, 1.0))
      phi = cosine * self.cos_m - sine * self.sin_m            # cos(theta + m)
      phi = torch.where(cosine > self.th, phi, cosine - self.mm)
      one_hot = F.one_hot(labels, num_classes=cosine.size(1)).bool()

      return self.scale * torch.where(one_hot, phi, cosine)    # logits for cross-entropy

# Helper methods to be moved into another module for clean design
def build_labels_map(labels):

   return {lab: i for i, lab in enumerate(sorted(set(labels)))}

def _as_list(x):
   return x.tolist() if isinstance(x, torch.Tensor) else list(x)

def _device(device):
   return torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

@torch.no_grad()
def extract_features(model, loader, device, amp):
   dev = _device(device)
   model.eval().to(dev)
   feats = []
   extras = None

   for batch in loader:
      images, rest = batch[0].to(dev, non_blocking = True), batch[1:]
      with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=amp and dev.type == "cuda"):
         emb = model.embed(images)

      feats .append(emb.float().cpu())
      if extras is None:
         extras = [[] for _ in rest]
      for bucket, field in zip(extras, rest):
         bucket.extend(_as_list(field))

   return torch.cat(feats), tuple(extras or [])

def fine_tune(model, train_loader, *, label_to_index, epochs = 100, lr = 1e-3, weight_decay = 0.0, arcface_margin = .5, arcface_scale = 64.0, accum_steps = 1, amp = True, device = None, log_every= 50):
   # SDG
   dev = _device(device)
   model.to(dev)
   head = ArcFaceHead(model.embedding_dim, len(label_to_index), scale = arcface_scale, margin=arcface_margin).to(dev)
   params = list(model.parameters()) + list(head.parameters())
   optimizer = torch.optim.SGD(params=params, lr=lr, momentum=0.9, weight_decay=weight_decay)
   scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
   use_amp = amp and dev.type == "cuda"
   scaler = torch.amp.GradScaler(dev.type, enabled=use_amp)

   for epoch in range(epochs):
      model.train()
      head.train()
      optimizer.zero_grad(set_to_none=True)
      running, n_batches = 0.0, 0

      for step, batch in enumerate(train_loader):
         images = batch[0].to(dev, non_blocking = True)
         targets = torch.tensor([label_to_index[l] for l in _as_list(batch[1])], device=dev, dtype=torch.long)

         with torch.autocast(device_type=dev.type, dtype= torch.float16, enabled=use_amp):
            emb = model(images)

         loss = F.cross_entropy(head(emb, targets), targets) # head runs in fp32
         scaler.scale(loss/ accum_steps).backward()

         if (step + 1) % accum_steps == 0 or (step + 1) == len(train_loader):
            scaler.step(optimizer=optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

         running += loss.item()
         n_batches +=1 

         if log_every and (step + 1) % log_every == 0:
            print(f"epoch {epoch + 1}/{epochs} step {step + 1}: loss {running / n_batches:.4f}")

      scheduler.step()
      print(f"epoch {epoch + 1}/{epochs} done - mean loss {running / max(n_batches, 1):.4f}, "
         f"lr {scheduler.get_last_lr()[0]:.2e}")
      
   return head


if __name__ == "__main__":
   m = GlobalModel().eval()
   print("data config: ", m.data_config)

   img = (np.random.rand(300, 400, 3) * 255).astype(np.uint8)
   with torch.no_grad():
      z = m.embed(m.eval_transform()(img).unsqueeze(0))
   print("embedding: ", tuple(z.shape), "norm: ", float(z.norm()))
   