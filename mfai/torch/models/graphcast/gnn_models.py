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

import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
from torch_geometric.utils import scatter

from typing import Tuple


class CheckpointWrapper(nn.Module):
    """Wrapper for checkpointing a module. Comes from AIFS"""

    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        return torch.utils.checkpoint.checkpoint(self.module, *args, **kwargs, use_reentrant=False)


class MLP(nn.Module):
    """
    MultiLayer Perceptron.
    Used for embeddings and mesh/node updates.

    Args:
        in_channel (int):
            Input size
        out_channel (int):
            Output size
        hidden_channel (int):
            Number of neurons per hidden layer, by default 512
        num_layers (int):
            Number of hidden layers, by default 1
        use_bias (bool):
            Whether to use bias, by default True
        use_norm (bool):
            Whether to use normalization, by default True
    """

    def __init__(
        self,
        in_channel: int,
        out_channel: int = 512,
        hidden_channel: int = 512,
        num_layers: int = 1,
        use_bias: bool = True,
        use_norm: bool = True,
    ):
        super().__init__()

        self.in_channel = in_channel
        self.out_channel = out_channel
        self.hidden_channel = hidden_channel
        self.num_layers = num_layers
        self.use_bias = use_bias
        self.use_norm = use_norm

        self.activation = nn.SiLU
        self.norm = AutoCastLayerNorm

        layers = nn.ModuleList()
        layers.append(
            nn.Linear(
                self.in_channel,
                self.hidden_channel,
                bias=self.use_bias,
            )
        )
        layers.append(self.activation())

        for _ in range(self.num_layers - 1):
            layers.append(
                nn.Linear(
                    self.hidden_channel,
                    self.hidden_channel,
                    bias=self.use_bias,
                )
            )
            layers.append(self.activation())

        layers.append(
            nn.Linear(
                self.hidden_channel,
                self.out_channel,
                bias=self.use_bias,
            )
        )

        if self.use_norm:
            layers.append(self.norm(self.out_channel))

        self.gnn_mlp = nn.Sequential(*layers)
        # self.gnn_mlp = CheckpointWrapper(self.gnn_mlp)

        self.init_weights()

    def init_weights(self):
        def custom_init(module):
            if isinstance(module, nn.Linear):
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight)
                stddev = 1.0 / np.sqrt(fan_in)
                nn.init.trunc_normal_(module.weight, std=stddev)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        self.apply(custom_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gnn_mlp(x)


class AutoCastLayerNorm(nn.LayerNorm):
    """
    This class comes from ECMWF.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x).type_as(x)


class MeshEdgeModel(nn.Module):
    """
    Model that is used to update the mesh edge attributes based on its source and target node features and
    its current edge features.

    Args:
        in_channel (int):
            Input dimension of the source and destination node features,
            and egde featuers, by default 512
        num_layers (int):
            Number of layer composing the MLP, by default 1
    """

    def __init__(
        self,
        in_channel: int = 512,
        num_layers: int = 1,
    ):
        super().__init__()

        self.mesh_edge_mlp = MLP(
            in_channel=3 * in_channel,
            out_channel=in_channel,
            hidden_channel=in_channel,
            num_layers=num_layers,
            use_norm=True,
        )

    def forward(
        self, src: Tensor, dst: Tensor, edge_index: Tensor, edge_attr: Tensor
    ) -> Tensor:
        x = torch.cat([src[edge_index[0]], dst[edge_index[1]], edge_attr], dim=1)

        return self.mesh_edge_mlp(x)


class MeshNodeModel(nn.Module):
    """
    Model that is used to update the mesh node features based on
    its current node features, its graph connectivity and its edge attributes.

    Args:
        in_channel (int):
            Input dimension of the node features, by default 512
        num_layers (int):
            Number of layer composing the MLP, by default 1
        aggregation (str):
            Aggregation operator, by default sum
    """

    def __init__(
        self,
        in_channel: int = 512,
        num_layers: int = 1,
        aggregation: str = "sum",
    ):
        super().__init__()

        self.mesh_node_mlp = MLP(
            in_channel=2 * in_channel,
            out_channel=in_channel,
            hidden_channel=in_channel,
            num_layers=num_layers,
            use_norm=True,
        )
        self.aggregation = aggregation

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        out = scatter(
            edge_attr, edge_index[1], dim=0, dim_size=x.size(0), reduce=self.aggregation
        )
        out = torch.cat([x, out], dim=1)

        return self.mesh_node_mlp(out)


class GridNodeModel(nn.Module):
    """
    Model that is used to update the grid node features. No aggregation is done as
    grid nodes are only senders in the Grid2Mesh subgraph.

    Args:
        in_channel (int):
            Input dimension of the node features, by default 512
        num_layers (int):
            Number of layer composing the MLP, by default 1
    """

    def __init__(
        self,
        in_channel: int = 512,
        num_layers: int = 1,
    ):
        super().__init__()

        self.grid_node_mlp = MLP(
            in_channel=in_channel,
            out_channel=in_channel,
            hidden_channel=in_channel,
            num_layers=num_layers,
            use_norm=True,
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.grid_node_mlp(x)


class InteractionNetwork(nn.Module):
    """
    An InteractionNetwork is a GraphNetwork without the global features.
    This method is a wrapper to the PyG implementation MetaLayer that is inspired
    by the `"Relational Inductive Biases, Deep Learning, and Graph Networks"
    <https://arxiv.org/abs/1806.01261>`_ paper.

    Args:
        in_channel (int):
            Input dimension of the source node features, by default 512
        num_layers (int):
            Number of hidden layers, by default 1
        aggregation (str):
            Aggregation function, by default sum
    """

    def __init__(
        self,
        in_channel: int = 512,
        num_layers: int = 1,
        aggregation: str = "sum",
    ):
        super().__init__()

        self.edge_model = MeshEdgeModel(in_channel=in_channel, num_layers=num_layers)
        self.node_model = MeshNodeModel(
            in_channel=in_channel, num_layers=num_layers, aggregation=aggregation
        )

    def forward(
        self, s_x: Tensor, r_x: Tensor, edge_index: Tensor, edge_attr: Tensor
    ) -> Tuple[Tensor, Tensor]:
        # We first update the edge attributes based on the source and destination node features.
        edge_attr_res = self.edge_model(s_x, r_x, edge_index, edge_attr)
        # We then update the mesh node features based on the edge attributes.
        r_x_res = self.node_model(r_x, edge_index, edge_attr_res)

        # Finally we add a residual connection to the mesh node and egde features.
        edge_attr = edge_attr + edge_attr_res
        r_x = r_x + r_x_res

        return r_x, edge_attr
