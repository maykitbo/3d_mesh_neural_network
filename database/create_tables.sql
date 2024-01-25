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
    NumVertices INT,
    NumEdges INT,
    NumFaces INT,
    NumVoxels INT,
    MeshStoragePath VARCHAR(255),
    GraphStoragePath VARCHAR(255),
    VoxelStoragePath VARCHAR(255),
    CreatedAt DATETIME,
    UpdatedAt DATETIME,
    AuthorID INT,
    FOREIGN KEY (ClassID) REFERENCES Class(ClassID),
    FOREIGN KEY (AuthorID) REFERENCES Author(AuthorID)
);

CREATE TABLE AugmentationData (
    AugmentationID INT AUTO_INCREMENT PRIMARY KEY,
    ClassID INT,
    NumVertices INT,
    NumEdges INT,
    NumFaces INT,
    NumVoxels INT,
    MeshStoragePath VARCHAR(255),
    GraphStoragePath VARCHAR(255),
    VoxelStoragePath VARCHAR(255),
    CreatedAt DATETIME,
    UpdatedAt DATETIME,
    SourceMeshID INT,
    TransformationDetails TEXT,
    FOREIGN KEY (SourceMeshID) REFERENCES MeshData(MeshID),
    FOREIGN KEY (ClassID) REFERENCES Class(ClassID)
);

CREATE TABLE SynthesisData (
    SynthesisID INT AUTO_INCREMENT PRIMARY KEY,
    ClassID INT,
    NumVertices INT,
    NumEdges INT,
    NumFaces INT,
    NumVoxels INT,
    MeshStoragePath VARCHAR(255),
    GraphStoragePath VARCHAR(255),
    VoxelStoragePath VARCHAR(255),
    CreatedAt DATETIME,
    UpdatedAt DATETIME,
    SynthesisModel TEXT,
    ModelBackupReference TEXT,
    FOREIGN KEY (ClassID) REFERENCES Class(ClassID)
);