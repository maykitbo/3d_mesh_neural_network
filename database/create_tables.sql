CREATE DATABASE 3d;

USE 3d;

CREATE TABLE Author (
    AuthorID INT AUTO_INCREMENT PRIMARY KEY,
    Name VARCHAR(255)
);

CREATE TABLE Class (
    ClassID INT AUTO_INCREMENT PRIMARY KEY,
    Name VARCHAR(255)
);

CREATE TABLE MeshData (
    MeshID INT AUTO_INCREMENT PRIMARY KEY,
    ClassID INT,
    AuthorID INT,
    NumVertices INT,
    NumEdges INT,
    NumIsolatedSubgraphs INT,
    MeshStoragePath TEXT,
    GraphStoragePath TEXT,
    -- VoxelStoragePath TEXT, -- For future use
    CreatedAt DATETIME,
    UpdatedAt DATETIME,
    FOREIGN KEY (ClassID) REFERENCES Class(ClassID),
    FOREIGN KEY (AuthorID) REFERENCES Author(AuthorID)
);

CREATE TABLE AugmentationData (
    AugmentationID INT AUTO_INCREMENT PRIMARY KEY,
    MeshID INT,
    ClassID INT,
    NumVertices INT,
    NumEdges INT,
    NumIsolatedSubgraphs INT,
    TransformationDetails TEXT,
    ModelReference TEXT,
    MeshStoragePath TEXT,
    GraphStoragePath TEXT,
    CreatedAt DATETIME,
    UpdatedAt DATETIME,
    FOREIGN KEY (MeshID) REFERENCES MeshData(MeshID),
    FOREIGN KEY (ClassID) REFERENCES Class(ClassID)
);

CREATE TABLE SynthesisData (
    SynthesisID INT AUTO_INCREMENT PRIMARY KEY,
    ClassID INT,
    NumVertices INT,
    NumEdges INT,
    NumIsolatedSubgraphs INT,
    SynthesisModel TEXT,
    ModelBackupReference TEXT,
    Details TEXT,
    MeshStoragePath TEXT,
    GraphStoragePath TEXT,
    CreatedAt DATETIME,
    UpdatedAt DATETIME,
    FOREIGN KEY (ClassID) REFERENCES Class(ClassID)
);