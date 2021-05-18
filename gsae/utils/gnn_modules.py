 
  import torch
 import torch.utils.data

 
 import torch_geometric
 from torch_geometric.nn import SAGEConv
 


 class DenseSAGEConv(torch.nn.Module):
 
 
     def __init__(self, in_channels, out_channels, improved=False, bias=True):
         super(DenseSAGEConv, self).__init__()
 
         self.in_channels = in_channels
         self.out_channels = out_channels
         self.sparse_sage = SAGEConv(in_channels,out_channels)
 
 
     def forward(self, x, adj):
         # x remains unchanged
         # adj converted to edge array
 
         # adj is b x N x N
         edge_tensor = [torch_geometric.utils.dense_to_sparse(a)[0] for a in adj] # only edges
         h_list = []
         for i in range(x.shape[0]):
             
             h_i = self.sparse_sage(x[i], edge_tensor[i])
             # print("h_i shape: {}".format(h_i.shape))
 
             h_list.append(h_i)
 
         h = torch.stack(h_list)
 
         # print("h shape: {}".format(h.shape))
         return h