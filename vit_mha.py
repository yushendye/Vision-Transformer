import torch
import torch.nn as nn
from vit_head import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MultiHeadedAttention(nn.Module):
    def __init__(self, num_heads, embedding_size):
        super(MultiHeadedAttention, self).__init__()
        self.num_heads = num_heads
        self.embedding_size = embedding_size
        self.attention_heads = nn.ModuleList().to(device)
        
        for head_i in range(self.num_heads):
            head = Head(self.num_heads, embedding_size=self.embedding_size)
            self.attention_heads.append(head)
        self.attention_heads.to(device)
            
    def forward(self, x):
        #print('MHA x shape in forward : ', x.shape)
        
        flag=True
        for i in range(self.num_heads):
            if flag:
                out_multihead = self.attention_heads[i](x)
                flag=False
            else:
                out_multihead = torch.cat((out_multihead,self.attention_heads[i](x)),axis=2)
        
        return out_multihead
