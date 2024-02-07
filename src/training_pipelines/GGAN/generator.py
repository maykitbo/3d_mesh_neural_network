import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import RNNConv

class GraphRNN(torch.nn.Module):
    def __init__(self, node_input_dim, rnn_hidden_dim, output_dim):
        super(GraphRNN, self).__init__()
        self.rnn = RNNConv(node_input_dim, rnn_hidden_dim)
        self.fc = torch.nn.Linear(rnn_hidden_dim, output_dim)

    def forward(self, data):
        # Assuming data.x are node features and data.edge_index are edge indices
        x, edge_index = data.x, data.edge_index
        
        # Get node embeddings from RNN
        x = self.rnn(x, edge_index)
        
        # Use a fully connected network to predict the next node/edge
        out = self.fc(x)
        
        return out

# Example usage:
node_input_dim = 3  # as each node is a 3D coordinate
rnn_hidden_dim = 64
output_dim = 3  # adjust based on what exactly you're predicting

model = GraphRNN(node_input_dim, rnn_hidden_dim, output_dim)

# Example graph data
node_features = torch.tensor([[0, 0, 0], [1, 1, 1]], dtype=torch.float)
edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
data = Data(x=node_features, edge_index=edge_index)

# Forward pass
out = model(data)
     


# if __name__ == '__main__':
#     import sys
#     sys.path.insert(1, '../../data_processing/')
#     import visualization


#     # Instantiate the generator
#     generator = GraphGenerator(256, 128, 8)
    
#     # Set the generator in evaluation mode
#     generator.eval()
    
#     # Generate a random noise vector
#     # The size of the noise vector is (batch_size, noise_dim)
#     # Here, we're generating one graph, so batch_size is 1
#     z = torch.randn(1, 256)
    
#     # Generate a graph
#     # Turn off gradients as we're not training now
#     with torch.no_grad():
#         vertices, edges = generator(z)
    
#     # Convert the output to a suitable format for your show_graph function
#     # Note: You might need to adjust this part based on how your show_graph expects the input
#     graph = {
#         'vertices': vertices.squeeze(0).cpu().numpy(),  # Removing the batch dimension and converting to numpy
#         'edges': edges.squeeze(0).cpu().numpy() if edges is not None else None
#     }
    
#     # Visualize the graph
#     visualization.show_graph(graph)
