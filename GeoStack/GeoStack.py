import torch
import clip
import torch.nn as nn
from GeoStack.GeoLayer import GeoLayer
from utils import logger

class GeoStackCLIP(nn.Module):
    def __init__(self, clip_model="ViT-B/16", geo_layers=None, device="cuda"):
        super().__init__()
        self.clip, self.preprocess = clip.load(clip_model, device=device, jit=False)
        self.clip.float()

        for param in self.clip.parameters():
            param.requires_grad = False

        if geo_layers is None or len(geo_layers) == 0:
            logger.warn("No GeoLayers are provided, this configuration return clip features")
        else:
            assert isinstance(geo_layers, list), "Expected a list for of GeoLayers!"

        embed_dim = self.clip.visual.output_dim
        W_stack = torch.eye(embed_dim, device=device).float()

        if geo_layers:
            for k in geo_layers:
                if hasattr(k, 'clip_model') and k.clip_model != clip_model:
                    logger.warn(f"Layer trained for {k.clip_model} loaded into {clip_model}")

                assert isinstance(k, GeoLayer), "Expected a GeoLayer!"
                W_stack = torch.mm(W_stack, k.get_weights().to(device))

        self.register_buffer('composed_weight', W_stack)

    def forward(self, images):
        x = self.clip.encode_image(images)
        x = x / x.norm(dim=-1, keepdim=True)

        x = x @ self.composed_weight
        return x / x.norm(dim=-1, keepdim=True)

