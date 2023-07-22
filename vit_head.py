import torch
import torch.nn as nn

class Head(nn.Module):
    def __init__(self, num_heads, embedding_size):
        super(Head, self).__init__()
        self.num_heads = num_heads
        self.embedding_size = embedding_size
        
        #Weight matrices of Q,K,V have dimensions: (embed_dim, embed_dim//no_of_heads)
        #initialize weight matrices
        self.Wq = nn.Parameter(torch.randn(embedding_size, self.embedding_size // self.num_heads)).to(device)
        self.Wk = nn.Parameter(torch.randn(embedding_size, self.embedding_size // self.num_heads)).to(device)
        self.Wv = nn.Parameter(torch.randn(embedding_size, self.embedding_size // self.num_heads)).to(device)
        
    def forward(self, x):
        self.Q = torch.matmul(x, self.Wq).to(device)
        self.K = torch.matmul(x, self.Wk).to(device)
        self.V = torch.matmul(x, self.Wv).to(device)
        self.K = torch.transpose(self.K, -2, -1)
        
        self.Q = nn.LayerNorm(self.Q.size()[1:]).to(device)(self.Q)
        self.K = nn.LayerNorm(self.K.size()[1:]).to(device)(self.K)
        self.V = nn.LayerNorm(self.V.size()[1:]).to(device)(self.V)
        
        qk = torch.matmul(self.Q, self.K)
        qk = qk / np.sqrt(self.K.shape[1])
        #print(qk.shape)
        qk = F.softmax(qk, dim=-1)
        qkv = torch.matmul(qk, self.V)
        
        return qkv