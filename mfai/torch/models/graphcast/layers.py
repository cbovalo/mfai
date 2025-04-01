import torch
import torch.nn as nn
from mfai.torch.models.nlam.interaction_net import make_mlp, InteractionNet
from mfai.torch.models.graphcast.embeddings import (
    GraphCastEncoderEmbedding,
    GraphCastProcessorEmbedding,
    GraphCastDecoderEmbedding,
)

from typing import List, Tuple


class GraphCastEncoder(nn.Module):
    def __init__(
        self,
        in_channel: int = 128,
        blueprint: list = [128],
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.encoder_embedder = GraphCastEncoderEmbedding(
            in_channel, blueprint, use_checkpoint
        )
        self.grid2mesh_mesh_gnn = InteractionNet()
        self.grid2mesh_grid_gnn = make_mlp()

    def forward(
        self, grid_node_features, mesh_node_features, grid2mesh_edge_features
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        grid_node_features = self.encoder_embedder(grid_node_features)
        mesh_node_features = self.encoder_embedder(mesh_node_features)
        grid2mesh_edge_features = self.encoder_embedder(grid2mesh_edge_features)

        return grid_node_features, mesh_node_features, grid2mesh_edge_features


class GraphCastProcessor(nn.Module):
    def __init__(self, num_blocks: int = 4):
        super().__init__()
        self.num_blocks = num_blocks
        self.processor_embedder = GraphCastProcessorEmbedder()
        self.mesh_processor = nn.ModuleList(
            [InteractionNet() for _ in range(self.num_blocks)]
        )


class GraphCastDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder_embedder = GraphCastDecoderEmbedder()
        self.mesh2grid_gnn = InteractionNet()
