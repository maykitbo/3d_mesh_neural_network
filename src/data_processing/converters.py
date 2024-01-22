import open3d as o3d
import numpy as np
import visualization
import torch
from torch_geometric.data import Data
import networkx as nx
from scipy.spatial.distance import cdist


def data_to_pcd(data):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(data.pos)
    pcd.normals = o3d.utility.Vector3dVector(data.x)
    return pcd


# Convert a PyG data object to Open3D mesh
def pyg_to_o3d_mesh_ball(data, radius_k = 1.5):
    pcd = data_to_pcd(data)

    distances = pcd.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    radius = radius_k * avg_dist
    # radius = np.max(distances)

    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
           pcd,
           o3d.utility.DoubleVector([radius, radius * 2]))
    
    return mesh


def pyg_to_o3d_mesh_alpha(data, alpha = 0.05):
    pcd = data_to_pcd(data)
    # pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    poisson_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
    bbox = pcd.get_axis_aligned_bounding_box()
    p_mesh_crop = poisson_mesh.crop(bbox)
    
    return p_mesh_crop


def pyg_to_o3d_mesh_surface(data, depth = 6):
    pcd = data_to_pcd(data)

    poisson_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth, width=0, scale=1.1, linear_fit=False)[0]
    bbox = pcd.get_axis_aligned_bounding_box()
    p_mesh_crop = poisson_mesh.crop(bbox)
    # pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    return p_mesh_crop


def compare_convert_algs(test_object,
                         algs = [pyg_to_o3d_mesh_ball,
                                 pyg_to_o3d_mesh_alpha,
                                 pyg_to_o3d_mesh_surface]
                        ):

    pcd = data_to_pcd(test_object)
    meshes = [pcd]
    for alg in algs:
        meshes.append(alg(test_object))

    visualization.draw_meshes(meshes)



def fix_graph(graph):
    connected_vertices = set()
    for edge in graph.edge_index.t().numpy():
        connected_vertices.add(edge[0])
        connected_vertices.add(edge[1])

    # Step 2: Identify isolated vertices
    all_vertices = set(range(graph.num_nodes))
    isolated_vertices = all_vertices - connected_vertices

    # If you want to connect isolated vertices:
    # Using a simple heuristic to connect each isolated vertex to its nearest non-isolated vertex
    if isolated_vertices:
        positions = graph.pos.numpy()
        for isolated_vertex in isolated_vertices:
            # Find the nearest non-isolated vertex
            distances = np.linalg.norm(positions[list(connected_vertices)] - positions[isolated_vertex], axis=1)
            nearest_vertex = list(connected_vertices)[np.argmin(distances)]

            # Add an edge between the isolated vertex and the nearest vertex
            new_edge = torch.tensor([[isolated_vertex, nearest_vertex], [nearest_vertex, isolated_vertex]], dtype=torch.long)
            graph.edge_index = torch.cat([graph.edge_index, new_edge], dim=1)


    # Step 1: Calculate pairwise distances between all vertices (or another suitable metric)
    positions = graph.pos.numpy()
    dist_matrix = cdist(positions, positions)

    # You might want to set a high distance for edges that already exist to avoid duplicating them
    for edge in graph.edge_index.t().numpy():
        dist_matrix[edge[0], edge[1]] = dist_matrix[edge[1], edge[0]] = np.inf

    # Step 2: Use these distances as weights to create a graph in networkx
    G = nx.from_numpy_array(dist_matrix)

    # Step 3: Compute the MST of this graph
    mst = nx.minimum_spanning_tree(G)

    # Step 4: Add the edges from the MST to your original graph
    for edge in mst.edges():
        graph.edge_index = torch.cat([graph.edge_index, torch.tensor([[edge[0], edge[1]], [edge[1], edge[0]]], dtype=torch.long)], dim=1)

    return graph



def mes_to_graph(mesh):
    triangles = mesh.triangles
    # Create edges based on triangles
    edges = []
    for v0, v1, v2 in triangles:
        edges.append([v0, v1])
        edges.append([v1, v2])
        edges.append([v2, v0])

    # Convert edges to a torch tensor with shape [2, num_edges]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    # Create the graph
    # graph = Data(x=point_cloud.x, edge_index=edge_index, pos=point_cloud.pos)
    graph = Data(x=point_cloud.x, edge_index=edge_index, pos=point_cloud.pos)
    graph = fix_graph(graph)
    return graph



def pointcloud_to_graph(point_cloud):
    mesh = pyg_to_o3d_mesh_ball(point_cloud)
    triangles = mesh.triangles
    # Create edges based on triangles
    edges = []
    for v0, v1, v2 in triangles:
        edges.append([v0, v1])
        edges.append([v1, v2])
        edges.append([v2, v0])

    # Convert edges to a torch tensor with shape [2, num_edges]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    # Create the graph
    graph = Data(x=point_cloud.x, edge_index=edge_index, pos=point_cloud.pos)
    graph = fix_graph(graph)
    return graph




