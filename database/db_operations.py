import os
import mysql.connector
from mysql.connector import Error
import open3d as o3d
import torch
from sql_queries import SQLQueries as sqlq
from settings import Personal as P


class DataBaseOperations():
    def __init__(self, host=P.host, database=P.db_name, user=P.user, password=P.password, dataset_path='dataset_path', mesh_id_cache=True):
        self.dataset_path = dataset_path
        self.classes = {}
        self.authors = {}
        try:
            self.connection = mysql.connector.connect(host=host, database=database, user=user, password=password)
            self.cursor = self.connection.cursor()
            self._init_cache()
            self._update_last_mesh_id(mesh_id_cache)
        except Error as e:
            print("Error while connecting to MySQL", e)
            self.connection = None
            self.cursor = None
    

    def _update_last_mesh_id(self, mesh_id_cache=True):
        if mesh_id_cache:
            self.last_mesh_id = self._get_mesh_id_no_cache()
            self.get_mesh_id = self._get_mesh_id_cache
        else:
            self.get_mesh_id = lambda: self._get_mesh_id_no_cache() + 1


    def _get_mesh_id_cache(self):
        self.last_mesh_id += 1
        return self.last_mesh_id
    

    def _get_mesh_id_no_cache(self):
        self.cursor.execute(sqlq.MAX_MESH_ID)
        return self.cursor.fetchone()[0] or 0


    def __del__(self):
        if self.connection is not None and self.connection.is_connected():
            if self.cursor is not None:
                self.cursor.close()
            self.connection.close()


    def _init_cache(self):
        # Initialize cache for classes and authors
        self.cursor.execute('SELECT ClassID, Name FROM Class')
        self.classes = {name.lower(): id for id, name in self.cursor.fetchall()}

        self.cursor.execute('SELECT AuthorID, Name FROM Author')
        self.authors = {name.lower(): id for id, name in self.cursor.fetchall()}


    def _check_connection(self):
        # Check if the connection and cursor exist
        if not self.connection or not self.cursor or not self.connection.is_connected():
            print("Database connection is not available.")
            return False
        return True


    def _check_insert_subtable(self, query, cache):
        to_insert = to_insert.lower()
        if to_insert in cache:
            return cache[to_insert]

        # self.cursor.execute(f'INSERT INTO {table_name} (Name) VALUES (%s)', (to_insert,))
        self.cursor.execute(query)
        self.connection.commit()
        id = self.cursor.lastrowid
        cache[to_insert] = id
        return id


    def check_insert_class(self, class_name):
        return self._check_insert_subtable(class_name, sqlq.INSERT_CLASS, self.classes)


    def check_insert_author(self, author_name):
        return self._check_insert_subtable(author_name, sqlq.INSERT_AUTHOR, self.authors)


    def add_mesh_data(self, class_name, author_name, mesh, graph, voxel=None):
        # 1. Insert or find the ClassID and AuthorID
        class_id = self.check_insert_class(class_name)
        author_id = self.check_insert_author(author_name)

        # 2. Save the mesh and graph data to files
        mesh_dir = os.path.join(self.dataset_path, 'meshes')
        graph_dir = os.path.join(self.dataset_path, 'graphs')

        # Generate unique ID for the mesh and graph
        mesh_id = self.get_mesh_id()

        # File paths
        mesh_path = os.path.join(mesh_dir, f'{mesh_id}.ply')
        graph_path = os.path.join(graph_dir, f'{mesh_id}.pt')

        # Save mesh in PLY format
        o3d.io.write_triangle_mesh(mesh_path, mesh)
        # Save graph in PyTorch format
        torch.save(graph, graph_path)

        # 3. Insert the record into the MeshData table
        query = sqlq.INSERT_MESH_DATA
        num_vertices = len(mesh.vertices)
        num_edges = len(mesh.triangles)  # Assuming triangles for edges
        num_subgraphs = 1  # Modify this as per your graph analysis
        self.cursor.execute(query, (class_id, author_id, num_vertices, num_edges, num_subgraphs, mesh_path, graph_path))
        self.connection.commit()

        return mesh_id

