from torch import Tensor
import torch.nn as nn

# from mfai.torch.models.nlam.interaction_net import make_mlp
from mfai.torch.models.graphcast.gnn_models import MLP

# from mfai.torch.models.graphcast.models import GraphCastSettings

from typing import Tuple


class Embedding(nn.Module):
    """
    Embedding class for the encoder and decoder inputs.
    All features are embedded using a multi-layer perceptron (MLP).

    Args:
        in_channel (int):
            Input size
        out_channel (int):
            Output size
        hidden_channel (int):
            Number of neurons per hidden layer, by default 512
        num_layers (int):
            Number of hidden layers, by default 1
        use_norm (bool):
            Whether to use normalization, by default True
    """

    def __init__(
        self,
        in_channel: int = 128,
        out_channel: int = 128,
        hidden_channel: int = 128,
        num_layers: int = 1,
    ):
        super().__init__()

        self.mlp = MLP(
            in_channel=in_channel,
            out_channel=out_channel,
            hidden_channel=hidden_channel,
            num_layers=num_layers,
            use_norm=True,
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.mlp(x)


class GraphCastEncoderEmbedding(nn.Module):
    """
    This class aims at representing the input features into a latent space.
    These new features will be fed into the GraphCast Processor.

    Args:
        encoder_config (GraphCastSettings): Configuration for the encoder.

    Returns:
        Tuple[Tensor, Tensor, Tensor]: A tuple containing the embedded grid node features,
                                       mesh node features, and grid-to-mesh edge features.
    """

    def __init__(
        self,
        encoder_config: "GraphCastSettings" = None,
    ):
        super().__init__()

        self.encoder_config = encoder_config

        self.grid_node_embedding = Embedding(
            in_channel=self.encoder_config.in_channels,
            out_channel=self.encoder_config.output_channel,
            hidden_channel=self.encoder_config.hidden_channel,
            num_layers=self.encoder_config.num_layers,
        )
        self.mesh_node_embedding = Embedding(
            in_channel=self.encoder_config.in_channels+self.encoder_config.input_mesh_node_channel,
            out_channel=self.encoder_config.output_channel,
            hidden_channel=self.encoder_config.hidden_channel,
            num_layers=self.encoder_config.num_layers,
        )
        self.grid2mesh_edge_embedding = Embedding(
            in_channel=self.encoder_config.input_grid2mesh_edge_channel,
            out_channel=self.encoder_config.output_channel,
            hidden_channel=self.encoder_config.hidden_channel,
            num_layers=self.encoder_config.num_layers,
        )

    def forward(
        self,
        grid_node_feat: Tensor,
        mesh_node_feat: Tensor,
        grid2mesh_edge_feat: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        # Grid node feature embedding
        grid_node_feat = self.grid_node_embedding(grid_node_feat)

        # Mesh node feature embedding
        mesh_node_feat = self.mesh_node_embedding(mesh_node_feat)

        # Grid2Mesh edge feature embedding
        grid2mesh_edge_feat = self.grid2mesh_edge_embedding(grid2mesh_edge_feat)

        return (
            grid_node_feat,
            mesh_node_feat,
            grid2mesh_edge_feat,
        )


class GraphCastProcessorEmbedding(nn.Module):
    """
    Embed the mesh edges into a latent space.

    Args:
        processor_config (GraphCastSettings): Configuration for the processor.

    Returns:
        Tensor: The embedded mesh edge features.
    """

    def __init__(
        self,
        processor_config: "GraphCastSettings" = None,
    ):
        super().__init__()

        self.processor_config = processor_config
        self.mesh_edge_embedding = Embedding(
            in_channel=self.processor_config.input_mesh_edge_channel,
            out_channel=self.processor_config.output_channel,
            hidden_channel=self.processor_config.hidden_channel,
            num_layers=self.processor_config.num_layers,
        )

    def forward(self, mesh_edge_feat: Tensor) -> Tensor:
        # Mesh edge feature embedding
        return self.mesh_edge_embedding(mesh_edge_feat)


class GraphCastDecoderEmbedding(nn.Module):
    """
    This class aims at representing the features of the Mesh2Grid graph into a latent space.
    These new features will be fed into the GraphCast Decoder.

    Args:
        in_mesh2grid_edge_channel (int): Number of mesh2grid edge features, by default 4.
        blueprint_end (list): List specifying the size of each layer in the MLP.
        use_checkpoint (bool): Whether to use checkpointing to save memory.

    Returns:
        Tensor: The embedded mesh2grid edge features.
    """

    def __init__(
        self,
        decoder_config: "GraphCastSettings" = None,
    ):
        super().__init__()

        self.decoder_config = decoder_config
        self.mesh2grid_edge_Embedding = Embedding(
            in_channel=self.decoder_config.input_mesh2grid_edge_channel,
            out_channel=self.decoder_config.output_channel,
            hidden_channel=self.decoder_config.hidden_channel,
            num_layers=self.decoder_config.num_layers,
        )

    def forward(
        self,
        mesh2grid_edge_feat: Tensor,
    ) -> Tensor:
        # Mesh2Grid edge feature embedding
        return self.mesh2grid_edge_Embedding(mesh2grid_edge_feat)
