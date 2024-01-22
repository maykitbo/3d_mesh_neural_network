import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import data_processing.converters as ptg
import plotly.graph_objects as pgo
# import trimesh
# from tqdm import tqdm


def vertices_distribution(dataset, bounds: lambda data: True):
    vertices_counts = [data.pos.shape[0] for data in dataset if bounds(data)]

    plt.hist(vertices_counts, bins=30)
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

    plt.hist(vertices_counts, bins=30)
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


def trimesh_to_open3d(trimesh_mesh):
    # Extract vertices and faces from the Trimesh object
    vertices = np.asarray(trimesh_mesh.vertices)
    faces = np.asarray(trimesh_mesh.faces)

    # Create an Open3D mesh object
    open3d_mesh = o3d.geometry.TriangleMesh()
    
    # Set vertices and faces
    open3d_mesh.vertices = o3d.utility.Vector3dVector(vertices)
    open3d_mesh.triangles = o3d.utility.Vector3iVector(faces)

    # Compute vertex normals
    open3d_mesh.compute_vertex_normals()

    return open3d_mesh


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
    node_pos = graph.pos.numpy()  # Convert to numpy array for easier handling

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

