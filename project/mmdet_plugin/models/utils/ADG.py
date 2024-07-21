import torch
import torch.nn as nn

class Mlp(nn.Module):
    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.ReLU,
                 drop=0.1):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

class ADG(nn.Module):
    def __init__(self, in_channels, n_query_channels=128, patch_size=4,shape=(16, 44), dim_out=256,
                 embedding_dim=128, num_heads=4, norm='linear'):
        super(mViT, self).__init__()
        self.norm = norm
        self.n_query_channels = n_query_channels
        self.patch_transformer = PatchTransformerEncoder(in_channels, patch_size, embedding_dim, num_heads)
        self.dot_product_layer = PixelWiseDotProduct()
        mlp_channels = (shape[0] * shape[1]) // (patch_size * patch_size)
        self.mlp = Mlp(mlp_channels, embedding_dim, embedding_dim) 
        
        self.conv3x3 = nn.Conv2d(in_channels, embedding_dim, kernel_size=3, stride=1, padding=1)
        self.regressor = nn.Sequential(nn.Linear(embedding_dim, 256),
                                       nn.LeakyReLU(0.1),
                                       nn.Linear(256, 256),
                                       nn.LeakyReLU(0.1),
                                       nn.Linear(256, dim_out))
        

    def forward(self, x):
        n, c, h, w = x.size() 
        tgt = self.patch_transformer(x.clone()) 
        x = self.conv3x3(x)
        regression_head = torch.mean(tgt,dim=0)
        queries = tgt
        queries = queries.permute(1, 2, 0).contiguous() 
        queries = self.mlp(queries) 
        queries = queries.permute(0, 2, 1).contiguous()
        range_attention_maps = self.dot_product_layer(x, queries)  
        y = self.regressor(regression_head) 
        y = torch.sigmoid(y)
        y = y / y.sum(dim=1, keepdim=True) 
        
        return y, range_attention_maps # N, dim_out [6,100]    [[6, 128, 20, 50]]
    

class PatchTransformerEncoder(nn.Module):
    def __init__(self, in_channels, patch_size=10, embedding_dim=128, num_heads=4):
        super(PatchTransformerEncoder, self).__init__()
        encoder_layers = nn.TransformerEncoderLayer(embedding_dim, num_heads, dim_feedforward=1024)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=2)  # takes shape S,N,E

        self.embedding_convPxP = nn.Conv2d(in_channels, embedding_dim,
                                           kernel_size=patch_size, stride=patch_size, padding=0)

        self.positional_encodings = nn.Parameter(torch.rand(500, embedding_dim), requires_grad=True)

    def forward(self, x):
        embeddings = self.embedding_convPxP(x).flatten(2).contiguous()  #  [bs*n_view,128,h*w/p^2]
        embeddings = embeddings + self.positional_encodings[:embeddings.shape[2], :].T.unsqueeze(0) # [bs*n_view,128,h*w/p^2]
        embeddings = embeddings.permute(2, 0, 1).contiguous() 
        x = self.transformer_encoder(embeddings) 
        return x


class PixelWiseDotProduct(nn.Module):
    def __init__(self):
        super(PixelWiseDotProduct, self).__init__()

    def forward(self, x, K):
        n, c, h, w = x.size()
        _, cout, ck = K.size()
        assert c == ck, "Number of channels in x and Embedding dimension (at dim 2) of K matrix must match"
        y = torch.matmul(x.view(n, c, h * w).permute(0, 2, 1).contiguous(), K.permute(0, 2, 1).contiguous()) 
        return y.permute(0, 2, 1).view(n, cout, h, w).contiguous() 
