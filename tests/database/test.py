from database.db_operations import DataBaseOperations
import numpy as np

DB = DataBaseOperations()

# print(DataBaseOperations.table_names)

DB.get_from_mesh_data(id = [1, 2, 3, 4])

DB.get_from_mesh_data(id = 5)

DB.get_from_mesh_data(id = 'e')

