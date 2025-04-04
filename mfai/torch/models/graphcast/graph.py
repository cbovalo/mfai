from functools import cached_property
from pathlib import Path
import numpy as np
import torch
from torch import Tensor
import torch_geometric as pyg
import trimesh
from typing import List, Tuple

from mfai.torch.models.graphcast.graph_utils import (
    cartesian_to_spherical,
    create_node_edge_features,
    export_to_meshio,
    get_g2m_connectivity,
    get_m2g_connectivity,
    lat_lon_to_spherical,
    spherical_to_cartesian,
    spherical_to_lat_lon,
)

import matplotlib.pyplot as plt


class Graph:
    def __init__(
        self,
        grid_latitude: np.ndarray,
        grid_longitude: np.ndarray,
        n_subdivisions: int = 3,
        coarser_mesh: int = 0,
        graph_dir: Path = Path("tmp/"),
    ) -> None:
        # self.meshgrid = meshgrid
        self.n_subdivisions = n_subdivisions
        self.coarser_mesh = coarser_mesh
        self.graph_dir = graph_dir

        self.grid_latitude = np.unique(grid_latitude)
        self.grid_longitude = np.unique(grid_longitude)

        self.grid2mesh_graph = None
        self.mesh_graph = None
        self.mesh2grid_graph = None

        self.grid_node_longitude, self.grid_node_latitude = np.meshgrid(
            self.grid_longitude, self.grid_latitude
        )
        self.grid_node_latitude = self.grid_node_latitude.reshape([-1]).astype(np.float32)
        self.grid_node_longitude = self.grid_node_longitude.reshape([-1]).astype(np.float32)

        self.meshes = self._create_square_mesh()

        mesh_phi, mesh_theta = cartesian_to_spherical(
            self.finest_mesh.vertices[:, 0].astype(np.float32),
            self.finest_mesh.vertices[:, 1].astype(np.float32),
            self.finest_mesh.vertices[:, 2].astype(np.float32),
        )

        self.mesh_node_longitude, self.mesh_node_latitude = spherical_to_lat_lon(
            mesh_phi, mesh_theta
        )

        self._node_edge_fetures_kwargs = {
            "add_node_positions": False,
            "add_node_latitude": True,
            "add_node_longitude": True,
            "add_relative_positions": True,
            "relative_latitude_local_coordinates": True,
            "relative_longitude_local_coordinates": True,
        }

    def _create_square_mesh(self) -> List[trimesh.Trimesh]:
        x_A, y_A, z_A = spherical_to_cartesian(
            *lat_lon_to_spherical(
                np.min(self.grid_longitude), np.min(self.grid_latitude)
            )
        )
        x_B, y_B, z_B = spherical_to_cartesian(
            *lat_lon_to_spherical(
                np.max(self.grid_longitude), np.min(self.grid_latitude)
            )
        )
        x_C, y_C, z_C = spherical_to_cartesian(
            *lat_lon_to_spherical(
                np.max(self.grid_longitude), np.max(self.grid_latitude)
            )
        )
        x_D, y_D, z_D = spherical_to_cartesian(
            *lat_lon_to_spherical(
                np.min(self.grid_longitude), np.max(self.grid_latitude)
            )
        )

        center_lon = (np.min(self.grid_longitude) + np.max(self.grid_longitude)) / 2
        center_lat = (np.min(self.grid_latitude) + np.max(self.grid_latitude)) / 2
        x_O, y_O, z_O = spherical_to_cartesian(
            *lat_lon_to_spherical(center_lon, center_lat)
        )

        self.vertices = np.array(
            [
                [x_A, y_A, z_A],
                [x_B, y_B, z_B],
                [x_C, y_C, z_C],
                [x_D, y_D, z_D],
                [x_O, y_O, z_O],
            ]
        )
        self.faces = np.array([[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]])

        # Unlike an icosahedron, each edge does not belong to two faces. The boundary edges are directed.
        mesh = trimesh.Trimesh(vertices=self.vertices, faces=self.faces)

        refined_mesh = [mesh]
        for _ in range(self.n_subdivisions):
            mesh = mesh.subdivide()
            refined_mesh.append(mesh)

        return refined_mesh[self.coarser_mesh :]

    def create_Grid2Mesh(self, fraction: float = 0.5, n_workers: int = 1) -> None:
        """
        Create a graph that connects grid nodes to mesh nodes.

        This method establishes connectivity between the grid nodes and the mesh nodes
        based on a specified fraction of the distance. It generates node and edge features
        for the grid2mesh graph and stores the resulting graph in the `grid2mesh_graph` attribute.

        Args:
            fraction (float): The fraction of the distance to consider for connectivity.
                              Default is 0.5.
            n_workers (int): The number of workers to use for parallel processing.
                             Default is 1.
        """

        src, dst = get_g2m_connectivity(
            self.meshes[-1],
            self.grid_latitude,
            self.grid_longitude,
            fraction=fraction,
            n_workers=n_workers,
        )

        # Get the node and edge features for the grid2mesh graph
        src_node_features, dst_node_features, edge_features = create_node_edge_features(
            graph_type="Grid2Mesh",
            src=src,
            dst=dst,
            src_lat=self.grid_node_latitude,
            src_lon=self.grid_node_longitude,
            dst_lat=self.mesh_node_latitude,
            dst_lon=self.mesh_node_longitude,
            **self._node_edge_fetures_kwargs,
        )

        # Create the PyG graph
        self.grid2mesh_graph = pyg.data.Data(
            x_s=torch.from_numpy(src_node_features),
            x_d=torch.from_numpy(dst_node_features),
            edge_index=torch.stack(
                [torch.from_numpy(src), torch.from_numpy(dst)], dim=0
            ),
            edge_attr=torch.from_numpy(edge_features).to(torch.float32),
        )

        # Save the grid2mesh graph to disk
        torch.save(self.grid2mesh_graph, f"{self.graph_dir}/grid2mesh.pt")

    def create_MultiMesh(self, save_mesh_to_vtk: bool = False) -> None:
        """
        Create a multi-resolution mesh graph from the finest mesh and node features.

        This method combines all the faces from the subdivided meshes into the finest mesh.
        It then generates node and edge features for the mesh graph using the provided
        node latitude and longitude information. The resulting mesh graph is stored in
        the `mesh_graph` attribute.

        Args:
            save_mesh (bool): If True, the method will save the generated meshes to disk
                              in VTK format. Default is False.
        """

        all_faces = [
            trimesh.exchange.export.export_dict(mesh)["faces"] for mesh in self.meshes
        ]
        all_faces = np.concatenate(all_faces)
        self.finest_mesh.faces = all_faces

        undirected_edges = np.stack(
            [self.finest_mesh.edges[:, 0], self.finest_mesh.edges[:, 1]], axis=0
        )
        undirected_edges = pyg.utils.to_undirected(torch.from_numpy(undirected_edges))
        undirected_edges = pyg.utils.coalesce(undirected_edges)

        node_features, edge_features = create_node_edge_features(
            graph_type="MultiMesh",
            src=undirected_edges[0].numpy(),
            dst=undirected_edges[1].numpy(),
            node_lat=self.mesh_node_latitude,
            node_lon=self.mesh_node_longitude,
            **self._node_edge_fetures_kwargs,
        )
        self.mesh_graph = pyg.data.Data(
            x=torch.from_numpy(node_features),
            edge_index=torch.stack(
                [
                    torch.from_numpy(np.copy(undirected_edges[0])),
                    torch.from_numpy(np.copy(undirected_edges[1])),
                ],
                dim=0,
            ),
            edge_attr=torch.from_numpy(edge_features).to(torch.float32),
            y=None,
        )

        # Save the mesh graph to disk
        torch.save(self.mesh_graph, f"{self.graph_dir}/multi_mesh.pt")

        if save_mesh_to_vtk:
            for i, mesh in enumerate(self.meshes):
                export_to_meshio(
                    mesh, f"{self.graph_dir}/square_mesh_{self.coarser_mesh+i}.vtk"
                )
            export_to_meshio(self.finest_mesh, f"{self.graph_dir}/multi_mesh.vtk")

    def create_Mesh2Grid(self, edge_normalization_factor: float = None) -> None:
        # Get the graph connectivity for Mesh2Grid
        src, dst = get_m2g_connectivity(self.meshes[-1], self.grid_latitude, self.grid_longitude)

        # Get the node and edge features for the mesh2grid graph
        _, _, edge_features = create_node_edge_features(
            graph_type="Mesh2Grid",
            src=src,
            dst=dst,
            src_lat=self.mesh_node_latitude,
            src_lon=self.mesh_node_longitude,
            dst_lat=self.grid_node_latitude,
            dst_lon=self.grid_node_longitude,
            edge_normalization_factor=edge_normalization_factor,
            **self._node_edge_fetures_kwargs,
        )

        # Create the PyG graph
        self.mesh2grid_graph = pyg.data.Data(
            x_s=None,
            x_d=None,
            edge_index=torch.stack(
                [torch.from_numpy(src), torch.from_numpy(dst)], dim=0
            ),
            edge_attr=torch.from_numpy(edge_features).to(torch.float32),
        )

        # Save the mesh2grid graph to disk
        torch.save(self.mesh2grid_graph, f"{self.graph_dir}/mesh2grid.pt")

    @cached_property
    def finest_mesh(self) -> trimesh.Trimesh:
        return self.meshes[-1].copy()

    def visualize(self):
        import cartopy.crs as ccrs

        plt.figure(figsize=(10, 10))
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.set_extent(
            [
                self.grid_longitude.min(),
                self.grid_longitude.max(),
                self.grid_latitude.min(),
                self.grid_latitude.max(),
            ],
            crs=ccrs.PlateCarree(),
        )
        ax.triplot(
            *spherical_to_lat_lon(
                *cartesian_to_spherical(
                    self.finest_mesh.vertices[:, 0],
                    self.finest_mesh.vertices[:, 1],
                    self.finest_mesh.vertices[:, 2],
                )
            ),
            triangles=self.finest_mesh.faces,
            transform=ccrs.PlateCarree(),
        )
        ax.coastlines()
        ax.stock_img()
        # plt.show()
        plt.savefig(f"{self.graph_dir}/mesh.png", dpi=200)


if __name__ == "__main__":
    lon = np.arange(-12, 16, 1)
    lat = np.arange(37.5, 55.4, 1)
    graph = Graph(lat, lon, n_subdivisions=5, coarser_mesh=3)
    graph.create_MultiMesh(save_mesh_to_vtk=True)
    graph.visualize()
