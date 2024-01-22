import open3d as o3d
import os
import torch
import numpy as np
import trimesh
from torch_geometric.data import Data
from tqdm import tqdm
import matplotlib.pyplot as plt
from statistics import mean, median, stdev
import threading
import pymeshlab
import random
import copy



raw_dir = 'datasets/Thingi10K/raw_meshes/'
graph_dir = 'datasets/Thingi10K/graphs/'


SIZE = 5000


def raw_data_file_path(id: int) -> str:
    return f'{raw_dir}{id}'


def all_raw_files():
    return [f for f in os.listdir(raw_dir) if f.endswith(".stl")]


def get_mesh(file_path):
    return o3d.io.read_triangle_mesh(file_path)


def draw_mesh(mesh, wireframe: bool = True, polygon: bool = True):
    if not polygon:
        # Extract vertices and triangles
        vertices = np.asarray(mesh.vertices)
        triangles = np.asarray(mesh.triangles)

        # Extract unique edges from triangles
        edges_set = set()
        for triangle in triangles:
            for i in range(3):
                edge = sorted([triangle[i], triangle[(i + 1) % 3]])
                edges_set.add(tuple(edge))
        edges = np.array(list(edges_set))

        # Create a LineSet object from the vertices and edges
        mesh = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(vertices),
            lines=o3d.utility.Vector2iVector(edges)
        )
    
    o3d.visualization.draw_geometries([mesh], mesh_show_wireframe=wireframe)


def draw_multiple_meshes(mesh_list, colors):
    for i, (mesh, color) in enumerate(zip(mesh_list, colors)):
        # Translate each mesh to avoid overlapping
        translation = np.array([i * max(mesh.get_max_bound() - mesh.get_min_bound()) * 1.2, 0, 0])
        mesh.translate(translation, relative=False)

        # Set color
        mesh.paint_uniform_color(color[0])
    
    # Draw all meshes in one window
    o3d.visualization.draw_geometries(mesh_list, mesh_show_wireframe=True)


def increase_vertices(mesh, target):
    mesh_trimesh = trimesh.Trimesh(np.asarray(mesh.vertices), np.asarray(mesh.triangles))
    
    while len(mesh_trimesh.vertices) < target:
        mesh_trimesh = mesh_trimesh.subdivide()
    
    # Convert back to Open3D mesh
    mesh_o3d = o3d.geometry.TriangleMesh()
    mesh_o3d.vertices = o3d.utility.Vector3dVector(mesh_trimesh.vertices)
    mesh_o3d.triangles = o3d.utility.Vector3iVector(mesh_trimesh.faces)
    return mesh_o3d


def adjust_mesh_vertices(mesh, target):
    current_vertices = len(mesh.vertices)
    
    if current_vertices > target:
        # Reduce vertices
        mesh = mesh.simplify_quadric_decimation(target)
    elif current_vertices < target:
        # Increase vertices
        mesh = increase_vertices(mesh, target)
    
    # Handle cases where exact target isn't reached
    # Further processing may be needed here

    return mesh




def compare_simplifying_algs(mesh):
    print('Original mesh size = ', len(mesh.vertices))
    # algs = [quadric_decimation_simpl, voxel_downsample_simpl, cluster_decimation_simpl, pymeshlab_simpl]
    algs = [adjust_mesh_vertices]


    colors = [[[0.8, 0.8, 0.8], 'Gray'],
              [[1, 0, 0], 'Red'],
              [[0, 1, 0], 'Green'],
              [[0, 0, 1], 'Blue'],
              [[1, 1, 0]], 'Yellow']

    meshes_to_draw = [mesh]

    for alg, color in zip(algs, colors[1:]):
        simplified_mesh = alg(mesh, target=SIZE)
        print(f'{alg.__name__} (color: {color[1]}): size = {len(simplified_mesh.vertices)}')
        meshes_to_draw.append(simplified_mesh)

    draw_multiple_meshes(meshes_to_draw, colors)


def normalize_mesh(vertices):
    # Center the vertices around the origin
    vertices -= vertices.mean(dim=0)

    # Scale the vertices
    max_distance = (vertices ** 2).sum(dim=1).sqrt().max()
    vertices /= max_distance

    return vertices


def load_stl_to_graph(file_path, simpl_alf):
    # mesh = trimesh.load(file_path, force='mesh')
    mesh = simpl_alf(file_path)

    vertices = torch.tensor(mesh.vertices, dtype=torch.float)
    faces = torch.tensor(mesh.faces, dtype=torch.long)

    # vertices = torch.tensor(vertices, dtype=torch.float)
    # faces = torch.tensor(faces, dtype=torch.long)

    vertices = normalize_mesh(vertices)

    edge_index = torch.cat([faces[:, :2], faces[:, 1:], faces[:, ::2]], dim=0)
    edge_index = edge_index.t().contiguous()

    # data = Data(x=vertices, edge_index=edge_index)

    print(vertices)
    print(edge_index)

    return Data(x=vertices, edge_index=edge_index)




def process_and_save_all_stl(simpl_alf):
    stl_files = all_raw_files()
    
    for filename in tqdm(stl_files, desc="Processing STL Files"):
        file_path = os.path.join(raw_dir, filename)
        try:
            graph_data = load_stl_to_graph(file_path, simpl_alf)
            save_path = os.path.join(graph_dir, filename.replace('.stl', '.pt'))
            torch.save(graph_data, save_path)
        except Exception as inst:
            print('Error witth ', file_path)
            print(inst)


def stat_show():
    vertices = []
    with open('stat.txt', 'r') as file:
        for line in file:
            if line.startswith('vertices:'):
                vertices.extend([int(vertex) for vertex in line.replace('vertices:', '').strip().split()])
            elif line.startswith('faces:'):
                break


    # plt.bar(range(len(vertices)), vertices, width=0.3, align='edge')
    # plt.show()
            
    print('max = ', max(vertices))
    print('min = ', min(vertices))
    print('mean = ', mean(vertices))
    print('median = ', median(vertices))
    print('stdev = ', stdev(vertices))

if __name__ == '__main__':
    # show_mwsh(34783, polygon=False)
    # load_stl_to_graph(raw_data_file_path(34783))

    # compare_simplifying_algs(get_mesh(raw_data_file_path(random.choice(all_raw_files()))))

    # process_and_save_all_stl()
    
    # with open('stat.txt', 'w') as file:
    #     file.write("vertices: ")
    #     for v in vs:
    #         file.write('%d ' % v)
    #     file.write('\nfaces: ')
    #     for f in fs:
    #         file.write('%d ' % f)
    #     file.write('\n')

    stat_show()
        


