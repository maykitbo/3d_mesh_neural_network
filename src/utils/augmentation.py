import open3d as o3d
import numpy as np
import copy

def rotate_mesh(mesh, angles):
    R = mesh.get_rotation_matrix_from_axis_angle(np.asarray(angles))
    return copy.deepcopy(mesh).rotate(R, center=mesh.get_center())


def add_random_noise_to_vertices(mesh, noise_radius):
    noize_mesh = copy.deepcopy(mesh)
    vertices = np.asarray(noize_mesh.vertices)
    noise = np.random.uniform(-noise_radius, noise_radius, vertices.shape)
    noize_mesh.vertices = o3d.utility.Vector3dVector(vertices + noise)
    return noize_mesh


def flip_mesh(mesh, axis):
    flipped_mesh = copy.deepcopy(mesh)
    vertices = np.asarray(flipped_mesh.vertices)
    if axis == 'x':
        vertices[:,0] *= -1
    elif axis == 'y':
        vertices[:,1] *= -1
    elif axis == 'z':
        vertices[:,2] *= -1
    flipped_mesh.vertices = o3d.utility.Vector3dVector(vertices)
    return flipped_mesh


def shear_mesh(mesh, shear_factor, axis):
    sheared_mesh = copy.deepcopy(mesh)
    vertices = np.asarray(sheared_mesh.vertices)
    if axis == 'x':
        vertices[:,0] += shear_factor * vertices[:,1]
    elif axis == 'y':
        vertices[:,1] += shear_factor * vertices[:,0]
    elif axis == 'z':
        vertices[:,2] += shear_factor * vertices[:,0]
    sheared_mesh.vertices = o3d.utility.Vector3dVector(vertices)
    return sheared_mesh


def jitter_mesh(mesh, sigma=0.01, clip=0.05):
    jittered_mesh = copy.deepcopy(mesh)
    vertices = np.asarray(jittered_mesh.vertices)
    noise = np.clip(sigma * np.random.randn(*vertices.shape), -clip, clip)
    jittered_mesh.vertices = o3d.utility.Vector3dVector(vertices + noise)
    return jittered_mesh
    

def subsample_mesh(mesh, voxel_size):
    """
    Reduce the number of vertices in the mesh using voxel downsampling.
    :param mesh: Input Open3D mesh object.
    :param voxel_size: Size of the voxel in which vertices are combined.
    :return: Downsampled mesh.
    """
    subsampled_mesh = copy.deepcopy(mesh)
    return subsampled_mesh.simplify_vertex_clustering(voxel_size=voxel_size, contraction=o3d.geometry.SimplificationContraction.Average)


def upsample_mesh(mesh, number_of_iterations=1):
    """
    Increase the number of vertices in the mesh using midpoint subdivision.
    :param mesh: Input Open3D mesh object.
    :param number_of_iterations: Number of times to apply the subdivision.
    :return: Upsampled mesh.
    """
    opsampled_mesh = copy.deepcopy(mesh)
    for _ in range(number_of_iterations):
        opsampled_mesh = opsampled_mesh.subdivide_midpoint()
    return opsampled_mesh


def smooth_mesh_simple(mesh, number_of_iterations=10):
    smoothed_mesh = copy.deepcopy(mesh)
    return smoothed_mesh.filter_smooth_simple(number_of_iterations)


def smooth_mesh_taubin(mesh, number_of_iterations=1, lambda_filter=0.5, mu=-0.53):
    smoothed_mesh = copy.deepcopy(mesh)
    return smoothed_mesh.filter_smooth_taubin(number_of_iterations, lambda_filter, mu)
