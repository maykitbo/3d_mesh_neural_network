class SQLQueries:
    MAX_MESH_ID = 'SELECT MAX(MeshID) FROM MeshData'
    INSERT_CLASS = 'INSERT INTO Class (Name) VALUES (%s)'
    INSERT_AUTHOR = 'INSERT INTO Author (Name) VALUES (%s)'
    INSERT_MESH_DATA = '''INSERT INTO MeshData (
        ClassID, AuthorID, NumVertices, NumEdges, NumIsolatedSubgraphs, 
        MeshStoragePath, GraphStoragePath, CreatedAt, UpdatedAt) 
        VALUES (%s, %s, %s, %s, %s, %s, %s, NOW(), NOW())'''

