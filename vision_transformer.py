from vit_head import *
from vit_mha import *
from vit_encoder import *

class ViTBlock(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super(ViTBlock, self).__init__()
        self.encoders = nn.Sequential()
        for i in range(num_layers):
            self.encoders.append(encoder_layer)
            
    def forward(self, x):
        out = self.encoders(x)
        return out
    
class VisionTransformer(nn.Module):
    def __init__(self, image, batch_size, patch_size, chanels, image_size, encoder_embedding_dim, encoder_hidden_dim, num_heads, num_layers, encoder_mlp_dim, vit_mlp_dim, num_classes):
        super(VisionTransformer, self).__init__()
        self.image = image.to(device)
        print(self.image.shape)
        self.batch_size = batch_size
        self.image.to(device)
        self.image_size = image_size
        self.encoder_embedding_dim = encoder_embedding_dim
        self.chanels = chanels
        self.N_patches = (image_size * image_size) // (patch_size ** 2)
        self.patch_size = patch_size
        self.num_classes = num_classes
        embedding_rows = self.patch_size ** 2 
        embedding_cols = encoder_embedding_dim
        print('cols : ', str(embedding_cols))
        
        self.conv = nn.Conv2d(
            in_channels=self.chanels, 
            kernel_size=self.patch_size, 
            stride=self.patch_size, 
            out_channels=self.encoder_embedding_dim
        ).to(device)
        
        self.positional_embeddings = nn.Parameter(torch.randn(self.N_patches, encoder_embedding_dim)).to(device)
        
        self.class_token = nn.Parameter(torch.randn(1, 1, embedding_cols)).to(device)
        #self.embedding_matrix = nn.Parameter(torch.randn(embedding_cols, embedding_rows))
        
        encoder = TransformerEncoder(
            batch_size = batch_size,
            chanels=chanels,
            image_size=image_size,
            patch_size=patch_size,
            embedding_dim=encoder_embedding_dim,
            encoder_mlp_dim=encoder_mlp_dim, 
            hidden_dim=encoder_hidden_dim, 
            num_heads=num_heads,
            num_layers=num_layers
        ).to(device)
        self.encoder = ViTBlock(encoder, num_layers)
        self.encoder.to(device)
        self.mlp = nn.Linear(in_features=self.encoder_embedding_dim, out_features=self.num_classes)
        #self.mlp =nn.Sequential(
        #    nn.Linear(in_features=encoder_embedding_dim, out_features=vit_mlp_dim),
        #    nn.GELU(),
        #    nn.Linear(in_features=vit_mlp_dim, out_features=num_classes),
        #    nn.GELU()
        #).to(device)
        
    def forward(self, x):
        #print('Input Image : ', x.shape)
        patches = self.conv(x)
        
        embedded_patches = patches.flatten(2).transpose(1, 2)
        #print('shape of patches : ', embedded_patches.shape)
        unsq = self.positional_embeddings.unsqueeze(0).expand(embedded_patches.shape[0], -1, -1)
        
        
        #patches size shape :  torch.Size([32, 512, 7, 7])
        #embedded_patches shape :  torch.Size([32, 49, 512])
        #unsqueeze expand shape:  torch.Size([32, 49, 512])
        
        #print('patches size shape : ', patches.shape)
        #print('embedded_patches shape : ', embedded_patches.shape)
        #print('unsqueeze expand shape: ', unsq.shape)
        
        embedded_patches_with_pos_emb = embedded_patches +  unsq
        #print('After pos emb : ', embedded_patches_with_pos_emb.shape)
        
        class_token = self.class_token.repeat(embedded_patches.shape[0], 1, 1)
        with_class_token = torch.cat([embedded_patches_with_pos_emb, class_token], dim=1)
        #print('with cls token : ', with_class_token.shape)
        encoded = self.encoder(with_class_token)
        #print('shape of encoded passed : ', encoded.shape)
        #print('shape of encoded expect : ', self.encoder_embedding_dim)
        
        output = self.mlp(encoded[:,0,:])

        return output