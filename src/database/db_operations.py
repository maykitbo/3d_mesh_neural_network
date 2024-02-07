import os
import mysql.connector
from mysql.connector import Error
import open3d as o3d
import torch
from .sql_queries import SQLQueries as sqlq
from .settings import Personal as P
import uuid
from pathlib import Path
from typing import Tuple
import numpy as np
from typing import Union, Iterable
from pandas import Interval
from .read_write import *


class DataBaseOperations():
    table_names = ['MeshData', 'AugmentationData', 'SynthesisData']

    def __init__(self, host=P.host, database=P.db_name, user=P.user, password=P.password, dataset_path=P.dataset_path):
        self.dataset_path = dataset_path
        self.classes = {}
        self.authors = {}
        try:
            self.connection = mysql.connector.connect(host=host, database=database, user=user, password=password)
            self.cursor = self.connection.cursor()
            self._init_cache()
        except Error as e:
            print("Error while connecting to MySQL", e)
            self.connection = None
            self.cursor = None


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


    def _check_insert_subtable(self, to_insert, query, cache):
        if self._check_connection() is False:
            return None

        to_insert = to_insert.lower()
        if to_insert in cache:
            return cache[to_insert]

        self.cursor.execute(query, (to_insert,))
        self.connection.commit()
        id = self.cursor.lastrowid
        cache[to_insert] = id
        return id


    def add_mesh_prepared_data(self,
                               class_id,
                               num_vert, num_e, num_t, num_vox,
                               mesh_p, graph_p, voxel_p,
                               author_id):

        print(class_id,
                num_vert, num_e, num_t, num_vox,
                mesh_p, graph_p, voxel_p,
                author_id)
        self.cursor.execute(
            sqlq.INSERT_MESH_DATA, (class_id,
                num_vert, num_e, num_t, num_vox,
                mesh_p, graph_p, voxel_p,
                author_id))
        
        self.connection.commit()
        return self.cursor.lastrowid


    def get_class_id(self, class_name):
        return self._check_insert_subtable(class_name, sqlq.INSERT_CLASS, self.classes)


    def get_author_id(self, author_name):
        return self._check_insert_subtable(author_name, sqlq.INSERT_AUTHOR, self.authors)


    def add_mesh_data(self, class_name, author_name, mesh, graph, voxel=None):
        if not self._check_connection():
            return None
        
        if voxel is None:
            num_voxeles = 0
            voxel_path = '-'
        else:
            # TODO Number of Voxeles
            # TODO Voxel Path
            pass

        # File paths with unique file name
        unique_file_name = str(uuid.uuid4())
        mesh_path = os.path.join(self.dataset_path, 'meshes', f'{unique_file_name}.ply')
        graph_path = os.path.join(self.dataset_path, 'graphs', f'{unique_file_name}.pt')

        # Save mesh in PLY format
        # o3d.io.write_triangle_mesh(mesh_path, mesh)
        save_mesh(mesh_path, mesh)
        # Save graph in PyTorch format
        # torch.save(graph, graph_path)
        save_graph(graph_path, graph)
        # Save voxel
        # TODO save_voxel(voxel, voxel_path)

        class_id = self.get_class_id(class_name)
        author_id = self.get_author_id(author_name)

        return self.add_mesh_prepared_data(
            class_id,
            len(mesh.vertices),
            graph.edge_index.shape[1],
            len(mesh.triangles),
            num_voxeles,
            mesh_path,
            graph_path,
            voxel_path,
            author_id
        )


    def _check_connection_and_table(func):
        def wrapper(self, *args, **kwargs):
            if not self._check_connection():
                return None
            if (args[0] not in DataBaseOperations.table_names):
                print(f"Error: Invalid table name: {args[0]}")
                return None
            return func(self, *args, **kwargs)
        return wrapper
    

    def _check_condition(self, condition):
        if isinstance(condition, str):
            return condition
        else:
            raise TypeError('Invalid condition type')


    def _id_condition(self, condition):
        if isinstance(condition, int):
            return f' = {condition}'
        elif isinstance(condition, (np.ndarray, Iterable)) and isinstance(condition[0], int):
            placeholders = ', '.join([str(id) for id in condition])
            return f' IN ({placeholders})'
        else:
            return None


    def _str_condition(self, condition, reference):
        if isinstance(condition, str):
            return f' = {reference[condition]}'
        elif isinstance(condition, Iterable) and isinstance(condition[0], str):
            placeholders = ', '.join([str(reference[name]) for name in condition])
            return f' IN ({placeholders})'
        else:
            return None


    def _ref_condition(self, condition, reference):
        id_result = self._id_condition(condition)
        if id_result is None:
            id_result = self._str_condition(condition, reference)
        return self._check_condition(id_result)


    def _range_condition(self, condition):
        if condition[0] is not None and condition[1] is not None:
            return ' BETWEEN %s AND %s', condition
        elif condition[0] is not None:
            return ' >= %s', (condition[0],)
        elif condition[1] is not None:
            return ' <= %s', (condition[1],)
        else:
            raise ValueError("Invalid range condition")


    def _request_preprocess(self, request: str | list[str]):
        requests_dict = {
            'count':        ('COUNT(*)', None),
            'id':           ('MeshID', None),
            'mesh_path':    ('MeshStoragePath', None),
            'graph_path':   ('GraphStoragePath', None),
            'voxel_path':   ('VoxelStoragePath', None),
            'num_vertices': ('NumVertices', None),
            'num_edges':    ('NumEdges', None),
            'num_faces':    ('NumFaces', None),
            'num_voxels':   ('NumVoxels', None),
            'created_at':   ('CreatedAt', None),
            'updated_at':   ('UpdatedAt', None),
            'author_id':    ('AuthorID', None),
            'class_id':     ('ClassID', None),
            'mesh':         ('MeshStoragePath', lambda path: load_mesh(path)),
            'graph':        ('GraphStoragePath', lambda path: load_graph(path)),
            'voxel':        ('VoxelStoragePath', lambda path: load_voxel(path)),
            'author':       ('AuthorID', lambda id: self.authors[id]),
            'class':        ('AuthorID', lambda id: self.classes[id])}

        if isinstance(request, str):
            request = [request]

        select_parts = []
        postprocess_actions = []

        for req in request:
            sql_part, postprocess = requests_dict.get(req, (None, None))
            if sql_part:
                select_parts.append(sql_part)
                postprocess_actions.append(postprocess)

        return select_parts, postprocess_actions

        
    
    def _request_postprocess(self):
        pass


    @_check_connection_and_table
    def get_from_mesh_data(self, table_name,
            id: int | np.ndarray | Iterable[int] | None = None,
            classes: int | np.ndarray | Iterable[int] | str | Iterable[str] | None = None,
            num_vertices_range: Tuple[int | None, int | None] | None = None,
            num_edges_range: Tuple[int | None, int | None] | None = None,
            num_faces_range: Tuple[int | None, int | None] | None = None,
            num_voxels_range: Tuple[int | None, int | None] | None = None,
            authors: int | np.ndarray | Iterable[int] | str | Iterable[str] | None = None,
            request: str | Iterable[str] | None = None):

        conditions = []
        if id:
            conditions.append('MeshId' + self._check_condition(self._id_condition(id)))
        if classes:
            conditions.append('ClassID' + self._ref_condition(classes, self.classes))
        if num_vertices_range:
            conditions.append('NumVertices' + self._range_condition(num_vertices_range))
        if num_edges_range:
            conditions.append('NumEdges' + self._range_condition(num_edges_range))
        if num_faces_range:
            conditions.append('NumFaces' + self._range_condition(num_faces_range))
        if num_voxels_range:
            conditions.append('NumVoxels' + self._range_condition(num_voxels_range))
        if authors:
            conditions.append('AuthorID' + self._ref_condition(authors, self.authors))

        select_parts, postprocess_actions  = self._request_preprocess(request)

        where_clause = ' AND '.join(conditions) if len(conditions) > 0 else None
        select_str = ", ".join(select_parts) if len(select_parts) > 0 else '*'

        query = f'SELECT {select_str} FROM {table_name}'
        if where_clause:
            query += f'WHERE {where_clause}'
        
        self.cursor.execute(query)
        results = self.cursor.fetchall()

        for response in results:
            processed_response = []
            for data, postprocess in zip(response, postprocess_actions):
                if postprocess:
                    processed_response.append(postprocess(data))
                else:
                    processed_response.append(data)
            yield tuple(processed_response)
        



    @_check_connection_and_table
    def get_table_size(self, table_name):
        query = sqlq.GET_TABLE_SIZE % table_name
        self.cursor.execute(query)
        return self.cursor.fetchone()[0]


    @_check_connection_and_table
    def get_mesh_path(self, table_name, mesh_id):
        # query = sqlq.GET_MESH_PATH % (table_name, '%s')
        # self.cursor.execute(query, (mesh_id,))
        self.cursor.execute(sqlq.GET_MESH_PATH % (table_name, mesh_id))
        return self.cursor.fetchone()[0]


    @_check_connection_and_table
    def get_graph_path(self, table_name, mesh_id):
        # query = sqlq.GET_GRAPH_PATH % (table_name, '%s')
        # self.cursor.execute(query, (mesh_id,))
        self.cursor.execute(sqlq.GET_GRAPH_PATH % (table_name, mesh_id))
        return self.cursor.fetchone()[0]


    @_check_connection_and_table
    def get_mesh(self, table_name, mesh_id):
        mesh_path = self.get_mesh_path(mesh_id, table_name)
        try:
            return o3d.io.read_triangle_mesh(mesh_path)
        except Exception as e:
            print(f'Error: read mesh file from {mesh_path}.', e)
            return None


    @_check_connection_and_table
    def get_grpah(self, table_name, mesh_id):
        graph_path = self.get_graph_path(table_name, mesh_id)
        try:
            return torch.load(graph_path)
        except Exception as e:
            print(f'Error: read graph file from {graph_path}.', e)
            return None
    

    @_check_connection_and_table
    def get_meshes(self, table_name, column_name, column_value):
        self.cursor.execute(sqlq.GET_MESHES % (table_name, column_name, column_value))
        return self.cursor.fetchone()[0]


    @_check_connection_and_table
    def get_graphs(self, table_name, column_name, column_value):
        query = sqlq.GET_GRAPHS % (table_name, column_name, column_value)
        self.cursor.execute(query)

        graphs = []
        for graph_path in [graph_path[0] for graph_path in self.cursor.fetchall()]:
            graphs.append(torch.load(graph_path))
        
        return graphs
    


