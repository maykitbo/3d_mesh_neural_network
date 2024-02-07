import torch
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_dense_adj, dense_to_sparse
import torch.nn.functional as F

class Encoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(Encoder, self).__init__()
        self.gconv1 = GCNConv(in_channels, hidden_channels)
        self.gconv2 = GCNConv(hidden_channels, out_channels)


    def forward(self, x, edge_index):
        x = F.relu(self.gconv1(x, edge_index))
        return self.gconv2(x, edge_index)



class Decoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(Decoder, self).__init__()
        self.linear1 = torch.nn.Linear(in_channels, hidden_channels)
        self.linear2 = torch.nn.Linear(hidden_channels, out_channels)


    def forward(self, z):
        z = F.relu(self.linear1(z))
        z = torch.sigmoid(self.linear2(z))
        adj = torch.matmul(z, z.t())
        return adj



class GVAE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GVAE, self).__init__()
        
        self.encoder = Encoder(in_channels, hidden_channels, out_channels)
        self.decoder = Decoder(out_channels, hidden_channels, in_channels)


    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar / 2)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu


    def forward(self, x, edge_index):
        z = self.encoder(x, edge_index)
        return self.decoder(z), z


# model = GVAE(in_channels=3, out_channels=64)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# # Training loop
# for epoch in range(200):
#     model.train()
#     optimizer.zero_grad()
#     reconstructed_adj, z = model(data.x, data.edge_index)
#     # Define your loss functions here
#     # loss = ...
#     loss.backward()
#     optimizer.step()
