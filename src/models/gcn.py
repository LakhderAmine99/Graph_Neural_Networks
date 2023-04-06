import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, Linear, global_mean_pool

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
    
class ProtGCN(torch.nn.Module):
    
    def __init__(self, nfeatures, nclasses, hidden_channels):
        
        super().__init__()
        torch.manual_seed(1234)
        
        self.conv1 = GCNConv(nfeatures, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        
        self.linear = Linear(hidden_channels,nclasses)
        
    def forward(self, x, edge_index, batch):
        
        # Node embeddings 
        h = self.conv1(x, edge_index)
        h = h.relu()
        h = self.conv2(h, edge_index)
        h = h.relu()
        h = self.conv3(h, edge_index)
        

        # Graph-level readout
        hG = global_mean_pool(h, batch)

        # Classifier
        h = F.dropout(hG, p=0.5, training=self.training)
        h = self.linear(h)
        
        return hG, F.log_softmax(h, dim=1)