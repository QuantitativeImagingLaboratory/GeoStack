import torch
import torch.nn as nn

class GeoLayer(nn.Module):
    def __init__(self, embed_dim, clip_model="Vit-B/16"):
        super().__init__()
        self.embed_dim = embed_dim
        self.clip_model = clip_model # to check compatibility of layers

        # Initialize W as identity to learn (I + delta)
        self.W = nn.Parameter(torch.eye(embed_dim))

        # Mask for upper triangular property
        self.register_buffer('tri_mask', torch.triu(torch.ones(embed_dim, embed_dim)))

    def forward(self, x):
        """
        :param x: image features
        :return: transformed feature IW
        """
        x = x @ self.get_weights()

        return x / x.norm(dim=-1, keepdim=True)

    def get_weights(self):
        return self.W * self.tri_mask