class SQLQueries:
    MAX_MESH_ID = 'SELECT MAX(MeshID) FROM MeshData'
    INSERT_CLASS = 'INSERT INTO Class (Name) VALUES (%s)'
    INSERT_AUTHOR = 'INSERT INTO Author (Name) VALUES (%s)'
    INSERT_MESH_DATA = '''INSERT INTO MeshData (
        ClassID, NumVertices, NumEdges, NumFaces, NumVoxels, 
        MeshStoragePath, GraphStoragePath, VoxelStoragePath, 
        CreatedAt, UpdatedAt, AuthorID) 
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NOW(), NOW(), %s)'''
    SELECT_ALL_CLASSES = 'SELECT ClassID, Name FROM Class'
    SELECT_ALL_AUTHORS = 'SELECT AuthorID, Name FROM Author'
    GET_TABLE_SIZE = 'SELECT COUNT(*) FROM %s'
    GET_MESH_PATH = 'SELECT MeshStoragePath FROM %s WHERE MeshID = %s'
    GET_GRAPH_PATH = 'SELECT GraphStoragePath FROM %s WHERE MeshID = %s'
    GET_MESHES = 'SELECT MeshStoragePath FROM %s WHERE %s = %s'
    GET_GRAPHS = 'SELECT GraphStoragePath FROM %s WHERE %s = %s'



