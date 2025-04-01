# Copyright 2024 Eviden.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from torch import Tensor
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from dataclasses_json import dataclass_json
from dataclasses import dataclass

from mfai.torch.models.graphcast.graph import Graph
from mfai.torch.models.graphcast.gnn_models import (
    GridNodeModel,
    InteractionNetwork,
    MLP,
)
from mfai.torch.models.graphcast.embeddings import (
    GraphCastEncoderEmbedding,
    GraphCastProcessorEmbedding,
    GraphCastDecoderEmbedding,
)

from typing import Tuple
from mfai.torch.models.base import ModelABC
from pathlib import Path
from py4cast.datasets.base import Statics


@dataclass_json
@dataclass(slots=True)
class GraphCastSettings:
    """
    Settings for the GraphCast model.
    """

    tmp_dir: Path | str = Path("tmp/")  # nosec B108

    # Graph configuration
    n_subdivisions: int = 6
    fraction: float = 0.6
    mesh2grid_edge_normalization_factor: float = None

    # Architecture configuration
    input_grid_node_channel: 186
    input_mesh_node_channel: 3
    input_mesh_edge_channel: 4
    input_grid2mesh_edge_channel: 4
    input_mesh2grid_edge_channel: 4
    output_grid_node_channel: 83
    output_channel: 512
    hidden_channel: int = 128
    num_layers: int = 1
    use_norm: bool = True

    use_checkpoint: bool = False
    mesh_aggr: str = "sum"
    processor_layers: int = 6


class GraphCast(ModelABC, nn.Module):

    setting_kls = GraphCastSettings
    onnx_supported = False
    input_dims: str = ("batch", "ngrid", "features")
    output_dims: int = ("batch", "ngrid", "features")

    def __init__(
        self,
        num_input_features: int,
        num_output_features: int,
        settings: GraphCastSettings = GraphCastSettings(),
        statics: Statics,
        input_shape: tuple,
    ) -> None:
        super().__init__()

        self._settings = settings

        # Let's create the graphs
        graph = Graph(
            grid_latitude=statics.grid_latitude,
            grid_longitude=statics.grid_longitude,
            n_subdivisions=settings.n_subdivisions,
            graph_dir=settings.tmp_dir,
        )
        graph.create_Grid2Mesh(fraction=settings.fraction, n_workers=4)
        graph.create_MultiMesh()
        graph.create_Mesh2Grid(
            edge_normalization_factor=settings.mesh2grid_edge_normalization_factor
        )

        self.grid2mesh_graph = graph.grid2mesh_graph
        for k, v in self.grid2mesh_graph.items():
            self.register_buffer(f"grid2mesh_{k}", v, persistent=False)

        self.mesh_graph = graph.mesh_graph
        for k, v in self.mesh_graph.items():
            self.register_buffer(f"mesh_{k}", v, persistent=False)

        self.mesh2grid_graph = graph.mesh2grid_graph
        for k, v in self.mesh2grid_graph.items():
            self.register_buffer(f"mesh2grid_{k}", v, persistent=False)

        self.num_grid_nodes = statics.grid_latitude * statics.grid_longitude

        # Instantiate the model components
        self.encoder = GraphCastEncoder(settings)
        self.processor = GraphCastProcessor(settings)
        self.decoder = GraphCastDecoder(settings)
        self.final_layer = MLP(
            in_channel=settings.output_channel,
            out_channel=settings.output_grid_node_channel,
            hidden_channel=settings.hidden_channel,
            num_layers=settings.num_layers,
            use_bias=True,
            use_norm=False,
        )

    @staticmethod
    def expand_to_batch(self, data: Tensor, batch_size: int) -> Tensor:
        """
        Expand the input data to a batch dimension.
        """
        dims = (batch_size,) + (-1,) * data.dim()
        return data.unsqueeze(0).expand(*dims)

    def forward(self, x: Tensor) -> Tensor:        
        grid_node_feat = x

        # Encoder
        ############################
        mesh_node_feat = self.mesh_x
        static_node_feat = self.expand_to_batch(self.grid2mesh_x_s, x.size(0))

        assert grid_node_feat.shape[1] == static_node_feat.shape[1]
        grid_node_feat = torch.cat([grid_node_feat, static_node_feat], dim=-1)

        """
        # We do as in GraphCast code, we add some zeros to the mesh node features to
        # 'make sure capacity of the embedded is identical for the grid nodes and the mesh nodes'.
        # https://github.com/google-deepmind/graphcast/blob/8debd7289bb2c498485f79dbd98d8b4933bfc6a7/graphcast/graphcast.py#L629
        """
        dummy_mesh_node_feat = torch.zeros(
            (mesh_node_feat.shape[1],) + (grid_node_feat.shape[1],),
            dtype=mesh_node_feat.dtype,
            device=grid_node_feat.device,
        )
        mesh_node_feat = torch.cat([dummy_mesh_node_feat, mesh_node_feat], dim=-1)
        
        grid_node_feat, mesh_node_feat, _ = self.encoder(
            grid_node_feat,
            mesh_node_feat,
            self.grid2mesh_edge_index,
            self.grid2mesh_edge_attr,
        )
        
        # Processor
        ############################
        mesh_node_feat, _ = self.processor(
            mesh_node_feat,
            self.mesh_edge_index,
            self.mesh_edge_attr,
        )
        
        # Decoder
        ############################
        grid_node_feat = self.decoder(
            grid_node_feat,
            mesh_node_feat,
            self.mesh2grid_edge_index,
            self.mesh2grid_edge_attr,
        )
        
        # Final layer
        ############################
        output = self.final_layer(grid_node_feat)

        return output


class GraphCastEncoder(nn.Module):
    def __init__(
        self,
        encoder_config: GraphCastSettings = GraphCastSettings(),
    ):
        super().__init__()

        self.encoder_embedding = GraphCastEncoderEmbedding(encoder_config)
        self.grid2mesh_mesh_gnn = InteractionNetwork(
            in_channel=encoder_config.output_channel,
            num_layers=encoder_config.num_layers,
            aggregation=encoder_config.mesh_aggr,
        )
        self.grid2mesh_grid_gnn = GridNodeModel(
            in_channel=encoder_config.output_channel,
            num_layers=encoder_config.num_layers,
        )

    def forward(
        self,
        grid_nfeat: Tensor,
        mesh_nfeat: Tensor,
        edge_index: Tensor,
        grid2mesh_efeat: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        # First, we embed the different features.
        grid_nfeat, mesh_nfeat, grid2mesh_efeat = self.encoder_embedding(
            grid_nfeat, mesh_nfeat, grid2mesh_efeat
        )
        # Then we apply the Interaction Network to the Grid2Mesh graph
        # and the MLP to the Grid node features. Residual connections are
        # done directly within the following models.
        mesh_nfeat, grid2mesh_efeat = self.grid2mesh_mesh_gnn(
            grid_nfeat, mesh_nfeat, edge_index, grid2mesh_efeat
        )
        grid_nfeat = self.grid2mesh_grid_gnn(grid_nfeat)

        return grid_nfeat, mesh_nfeat, grid2mesh_efeat


class GraphCastProcessor(nn.Module):
    def __init__(
        self,
        processor_config: GraphCastSettings = GraphCastSettings(),
    ):
        super().__init__()

        self.processor_embedding = GraphCastProcessorEmbedding(processor_config)
        self.processor = nn.ModuleList()
        for _ in range(processor_config.processor_layers):
            self.processor.append(
                InteractionNetwork(
                    in_channel=processor_config.output_channel,
                    num_layers=processor_config.num_layers,
                    aggregation=processor_config.mesh_aggr,
                )
            )

    def forward(
        self,
        mesh_nfeat: Tensor,
        edge_index: Tensor,
        mesh_efeat: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        mesh_efeat = self.processor_embedder(mesh_efeat)

        for layer in self.processor:
            mesh_nfeat, mesh_efeat = layer(
                mesh_nfeat, mesh_nfeat, edge_index, mesh_efeat
            )

        return mesh_nfeat, mesh_efeat


class GraphCastDecoder(nn.Module):
    def __init__(
        self,
        decoder_config: GraphCastSettings = GraphCastSettings(),
    ):
        super().__init__()

        self.decoder_embedding = GraphCastDecoderEmbedding(decoder_config)
        self.mesh2grid_gnn = InteractionNetwork(
            in_channel=decoder_config.output_channel,
            num_layers=decoder_config.num_layers,
            aggregation=decoder_config.mesh_aggr,
        )

    def forward(
        self,
        grid_nfeat: Tensor,
        mesh_nfeat: Tensor,
        edge_index: Tensor,
        mesh2grid_efeat: Tensor,
    ) -> Tensor:
        # First the edge attributes of Mesh2Grid are encoded into a latent representation.
        mesh2grid_efeat = self.decoder_embedding(mesh2grid_efeat)

        # Then we apply the Interaction Network to the Mesh2Grid graph.
        # Residual connections are done directly within the following models.
        # The final MLP to output the predictions is done outside this module.
        grid_nfeat, _ = self.mesh2grid_gnn(
            mesh_nfeat, grid_nfeat, edge_index, mesh2grid_efeat
        )

        return grid_nfeat
