import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCN(torch.nn.Module):
    
    def __init__(self, nfeatures, nclasses, hidden_channels) -> None:
        
        super().__init__()
        torch.manual_seed(1234)
        
        self.conv1 = GCNConv(nfeatures,hidden_channels)
        self.conv2 = GCNConv(hidden_channels,nclasses)
        
    def forward(self, x, edge_index):
        
        x = self.conv1(x, edge_index)

        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        
        x = self.conv2(x, edge_index)
        
        return x