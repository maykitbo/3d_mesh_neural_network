from db_operations import DataBaseOperations
from sql_queries import SQLQueries as sqlq





db = DataBaseOperations()

# ed1 = db.get_class_id('aboba')
# ed2 = db.get_author_id('erfet')

# db.add_mesh_prepared_data(ed1,
#                           10,
#                           10,
#                           10,
#                           10,
#                           '123',
#                           '456',
#                           '-',
#                           ed2)

print(db.get_table_size('MeshData'))

