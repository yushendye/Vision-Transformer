from vit_head import *
from vit_mha import *
class TransformerEncoder(nn.Module):
    def __init__(self, batch_size, chanels, image_size, patch_size, embedding_dim, num_heads, num_layers, encoder_mlp_dim, hidden_dim):
        super(TransformerEncoder, self).__init__()
        
        self.batch_size = batch_size
        self.image_size = image_size
        self.chanels = chanels
        self.patch_size = patch_size
        self.embedding_dim = embedding_dim
        self.encoder_mlp_dim = encoder_mlp_dim
        self.N_patches = (self.image_size ** 2) // (self.patch_size ** 2)
        #self.layernorm1 = nn.LayerNorm(normalized_shape=(batch_size, self.N_patches, self.patch_size * self.patch_size)).to(device)
        self.mha = MultiHeadedAttention(num_heads=num_heads, embedding_size=embedding_dim).to(device)
        #self.layernorm2 = nn.LayerNorm(normalized_shape=(batch_size, chanels, image_size, image_size)).to(device)
        
        self.mlp = nn.Sequential(
            nn.Linear(in_features=embedding_dim*(self.N_patches + 1), out_features=2048),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(in_features=2048, out_features=embedding_dim*(self.N_patches + 1)),
            nn.GELU(),
            nn.Dropout(0.1)
        ).to(device)
        
    def forward(self, x):
        shortcut1 = x.clone()
        #patches = get_image_patches(x)
        #print(x.shape)
        normalized_patches = nn.LayerNorm(x.size()[1:]).to(device)(x)
        #print('input to MHA ', normalized_patches.shape)
        multi_headed_op = self.mha(normalized_patches)
        #print('output of MHA ', multi_headed_op.shape)
        add_norm = shortcut1 + multi_headed_op
        shortcut2 = add_norm.clone()
        normalized_patches2 = nn.LayerNorm(add_norm.size()[1:]).to(device)(add_norm)
        
        mlp = self.mlp(normalized_patches2.reshape(
            normalized_patches2.shape[0], 
            normalized_patches2.shape[1] * normalized_patches2.shape[2])
        )
        
        #print('MLP output : ', mlp.shape)
        #print('normalized patch shape : ', normalized_patches2.shape)
        #normalized patch shape :  torch.Size([32, 50, 512])
        out = mlp + normalized_patches2.reshape(normalized_patches2.shape[0], normalized_patches2.shape[1] * normalized_patches2.shape[2])
        # MLP output :  torch.Size([32, 25600])
        # normalized patch shape :  torch.Size([32, 50, 512])
        
        return out.reshape(normalized_patches2.shape[0], normalized_patches2.shape[1], normalized_patches2.shape[2])