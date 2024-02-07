import open3d as o3d
import numpy as np
import struct
import torch
import open3d as o3d

def save_graph(graph, path):
    torch.save(graph, path)


def save_mesh(mesh, path):
     o3d.io.write_triangle_mesh(mesh, path)


def load_graph(path):
    return torch.load(path)


def load_mesh(path):
    return o3d.io.read_triangle_mesh(path)

def save_voxel(voxel_grid, path):
    # Assuming voxel_grid is an instance of o3d.geometry.VoxelGrid
    voxels = np.asarray(voxel_grid.get_voxels())
    
    binary_data = b''
    for voxel in voxels:
        # Encoding voxel center position
        binary_data += struct.pack('fff', *voxel.grid_index)

    with open(path, 'wb') as file:
        file.write(binary_data)



def load_voxel(path):
    # This part is more complex as it depends on how you've saved your data
    # Here's a basic example assuming you only saved the voxel positions
    with open(path, 'rb') as file:
        binary_data = file.read()

    # Number of bytes per voxel (3 floats, 4 bytes each)
    bytes_per_voxel = 12
    num_voxels = len(binary_data) // bytes_per_voxel

    voxel_grid = o3d.geometry.VoxelGrid()  # Creating an empty VoxelGrid
    for i in range(num_voxels):
        voxel_data = binary_data[i*bytes_per_voxel:(i+1)*bytes_per_voxel]
        x, y, z = struct.unpack('fff', voxel_data)
        # Add voxel to the grid. You might need additional info like voxel size.
        # For example: voxel_grid.add_voxel(o3d.geometry.Voxel(x, y, z))
        # The exact method depends on how the VoxelGrid is supposed to be reconstructed

    return voxel_grid
