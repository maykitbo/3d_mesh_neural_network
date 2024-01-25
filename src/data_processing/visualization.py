import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import converters as ptg
import plotly.graph_objects as pgo
# import trimesh
# from tqdm import tqdm


def vertices_distribution(dataset, bounds: lambda data: True):
    vertices_counts = [data.pos.shape[0] for data in dataset if bounds(data)]

    plt.hist(vertices_counts, bins=range(11), align='left', rwidth=0.9)
    plt.xlabel('Number of Vertices')
    plt.ylabel('Count')
    plt.title('Distribution of Vertices Across All Classes')
    plt.show()

    return vertices_counts


def class_distribution(dataset, get_category):
    category_counts = {}
    for data in dataset:
        value = get_category(data)
        if category_counts.get(value) is not None:
            category_counts[value] += 1
        else:
            category_counts[value] = 0

    plt.bar(category_counts.keys(), category_counts.values())
    # plt.hist(category_counts)
    plt.xlabel('Category')
    plt.ylabel('Count')
    plt.title('Distribution of Categories')
    # plt.xticks(rotation=45)
    plt.show()

    return category_counts


def vertices_distribution_in_class(dataset, get_category):
    vertices_counts = []
    for data in dataset:
        if get_category(data):
            vertices_counts.append(data.pos.shape[0])

    plt.hist(vertices_counts, bins=30, align='left', rwidth=0.9)
    plt.xlabel('Number of Vertices')
    plt.ylabel('Count')
    plt.title(f'Distribution of Vertices')
    plt.show()

    return vertices_counts


def combined_vertices_distribution(dataset,
                                   get_category):
    vertices_counts_dict = {}
    for data in dataset:
        category = get_category(data)
        if vertices_counts_dict.get(category) is not None:
            vertices_counts_dict[category].append(data.pos.shape[0])
        else:
            vertices_counts_dict[category] = [data.pos.shape[0]]

    # Plotting each category as a line
    for category, vertices_counts in vertices_counts_dict.items():
        # Sort and create a distribution array
        sorted_counts = sorted(vertices_counts)
        distribution = np.arange(len(sorted_counts))

        plt.plot(distribution, sorted_counts, label=category)

    plt.xlabel('Number of Vertices')
    plt.ylabel('Count')
    plt.title('Distribution of Vertices by Category')
    plt.legend()
    plt.show()


def draw_mesh_voxel(mesh, voxel, colors = [[0, 1, 0], [0, 0, 1]]):
    translation = np.array([max(mesh.get_max_bound() - mesh.get_min_bound()) * 1.2, 0, 0])
    mesh.translate(translation, relative=False)
    mesh.paint_uniform_color(colors[0])
    o3d.visualization.draw_geometries([mesh, voxel], mesh_show_wireframe=True)


def draw_meshes(mesh_list,
                        colors = [[1, 0, 0], [0, 1, 0],
                                [0, 0, 1], [1, 1, 0],
                                [1, 0, 1], [0, 1, 1]]
                      ):

    for i, (mesh, color) in enumerate(zip(mesh_list, colors)):
        # Translate each mesh to avoid overlapping
        translation = np.array([i * max(mesh.get_max_bound() - mesh.get_min_bound()) * 1.2, 0, 0])
        mesh.translate(translation, relative=False)

        # Set color
        mesh.paint_uniform_color(color)
    
    # Draw all meshes in one window
    o3d.visualization.draw_geometries(mesh_list, mesh_show_wireframe=True)


def plot_mesh(ax, mesh, name):
    ax.set_title(name)
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    ax.plot_trisurf(vertices[:, 0], vertices[:, 1], vertices[:, 2], triangles=triangles)


def plot_voxel(ax, voxel_grid, name):
    ax.set_title(name)
    voxel_indices = np.array([v.grid_index for v in voxel_grid.get_voxels()])
    grid_size = np.max(voxel_indices, axis=0) + 1
    voxels = np.zeros(grid_size, dtype=bool)
    for x, y, z in voxel_indices:
        voxels[x, y, z] = True
    ax.voxels(voxels, edgecolor='k')


def plot_graph(ax, graph, name):
    ax.set_title(name)
    # Draw vertices
    pos = graph.x
    ax.scatter(pos[:, 0].numpy(), pos[:, 1].numpy(), pos[:, 2].numpy(), c='blue', s=0.3)
    # Draw edges
    edge_index = graph.edge_index
    for i, j in edge_index.t().numpy():
        ax.plot(*zip(pos[i].numpy(), pos[j].numpy()), color='red', linewidth=0.1)


def plot_meshes(meshes, names, file_path: str | None = None):
    size = len(meshes)
    fig = plt.figure(figsize=(6 * size, 6))
    for i in range(size):
        ax = fig.add_subplot(131 + i, projection='3d')
        plot_mesh(ax, meshes[i], names[i])

    if file_path is not None:  
        plt.savefig(file_path, dpi=300)
    plt.show()



def plt_voxels_mesh_graph(voxel_grid, mesh, graph, file_path: str | None = None):
    # # Create a figure and a 3D axis
    fig = plt.figure(figsize=(18, 8))
    ax0 = fig.add_subplot(131, projection='3d')
    ax1 = fig.add_subplot(132, projection='3d')
    ax2 = fig.add_subplot(133, projection='3d')

    # Plot voxels
    plot_voxel(ax0, voxel_grid, 'Voxels')

    # # Plot mesh
    plot_mesh(ax1, mesh, 'Mesh')

    # Plot graph
    plot_graph(ax2, graph, 'Graph')

    if file_path is not None:  
        plt.savefig(file_path, dpi=300)
    plt.show()


def show_y_logic(test_object):
    pcd = ptg.data_to_pcd(test_object)

    # Ensure the colors list has enough colors for the range of your data
    colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1]]
    min_value = np.min(test_object.y.numpy())
    max_value = np.max(test_object.y.numpy())
    count = max_value - min_value + 1
    colors = np.array(colors * (count // len(colors) + 1))[:count]

    # Assign colors to each point in the point cloud
    color_map = np.array([colors[y - min_value] for y in test_object.y.numpy()])
    pcd.colors = o3d.utility.Vector3dVector(color_map)

    # Visualization
    o3d.visualization.draw_geometries([pcd], mesh_show_wireframe=True)


def show_graph(graph):
    # Extract node positions
    node_pos = graph.x.numpy()  # Convert to numpy array for easier handling

    # Prepare edge data for plotting
    edge_x = []
    edge_y = []
    edge_z = []
    for edge in graph.edge_index.t().numpy():
        src = node_pos[edge[0]]  # Start point of the edge
        dest = node_pos[edge[1]]  # End point of the edge
        edge_x += [src[0], dest[0], None]  # Add None to create a segment
        edge_y += [src[1], dest[1], None]
        edge_z += [src[2], dest[2], None]

    # Create scatter plot for nodes
    node_trace = pgo.Scatter3d(x=node_pos[:, 0], y=node_pos[:, 1], z=node_pos[:, 2],
                            mode='markers', marker=dict(size=3, color='blue'))

    # Create line plot for edges
    edge_trace = pgo.Scatter3d(x=edge_x, y=edge_y, z=edge_z,
                            mode='lines', line=dict(width=2, color='red'))

    # Create a figure and add both the node and edge traces
    fig = pgo.Figure(data=[edge_trace, node_trace])

    # Update layout for a better view
    fig.update_layout(template="plotly_dark", title="3D Graph Visualization")

    # Show the figure
    fig.show()

