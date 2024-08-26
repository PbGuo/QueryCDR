import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)

class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, dim, depth, heads, mlp_dim, channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        self.patch_size = patch_size

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        #self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.reversed_re = Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2)', p1=patch_height, p2=patch_width, h=image_height//patch_height , w=image_width//patch_width)

        #self.to_latent = nn.Identity()

        #self.mlp_head = nn.Linear(dim, num_classes)

    def forward(self, img):
        _, _, w, h = img.shape
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        #cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        #x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :n]
        x = self.dropout(x)

        x = self.transformer(x)

        #x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        #x = self.to_latent(x)
        x = self.reversed_re(x)
        return x

class CrossAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)  # Projection for q
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)  # Projection for k and v

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, y):
        x = self.norm(x)
        y = self.norm(y)

        q = self.to_q(x)
        kv = self.to_kv(y).chunk(2, dim=-1)
        k, v = kv[0], kv[1]

        q = rearrange(q, 'b n (h d) -> b h n d', h=self.heads)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.heads)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class CrossTransformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                CrossAttention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, x, y):
        for attn, ff in self.layers:
            x = attn(x, y) + x
            x = ff(x) + x

        return self.norm(x)
    
class CrossViT(nn.Module):
    def __init__(self, *, image_size, patch_size, dim, depth, heads, mlp_dim, channels1 = 512, channels2 = 128, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        # mlpdim=patch_size**2 * channel
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        self.patch_size = patch_size

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim1 = channels1 * patch_height * patch_width
        patch_dim2 = channels2 * patch_height * patch_width
        

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim1),
            nn.Linear(patch_dim1, dim),
            nn.LayerNorm(dim),
        )

        self.to_patch_embedding2 = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim2),
            nn.Linear(patch_dim2, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        self.pos_embedding2 = nn.Parameter(torch.randn(1, num_patches, dim))
        #self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = CrossTransformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.reversed_re = Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2)',
                                      p1=patch_height, p2=patch_width, h=image_height//patch_height ,
                                      w=image_width//patch_width)

        #self.to_latent = nn.Identity()

        #self.mlp_head = nn.Linear(dim, num_classes)

    def forward(self, img, fea):

        _, _, w, h = img.shape
        x = self.to_patch_embedding(img)
        y = self.to_patch_embedding2(fea)
        b, n, _ = x.shape

        #cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        #x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :n]
        y += self.pos_embedding2[:, :n]
        x = self.dropout(x)
        y = self.dropout(y)
        x = self.transformer(x,y)

        #x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        #x = self.to_latent(x)
        x = self.reversed_re(x)
        return x

class DisViT(nn.Module):
    def __init__(self, *, image_size, patch_size, dim, depth, heads, mlp_dim, channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        self.patch_size = patch_size

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
        )

        self.hw_rearrange = Rearrange('b (h w) (p1 p2 c) -> b (p1 p2 c) h w ', h=image_height//patch_height, p1 = patch_height, p2 = patch_width)
        #self.de_hw_rearrange = Rearrange('b (p1 p2 c) (h w) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width)

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))

        self.dis_embedding = nn.Parameter(torch.randn(image_height//patch_size//2, dim, 4*image_height//patch_size-4))

        self.to_dis_embedding = nn.Sequential(
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim))

        #self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.reversed_re = Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2)', p1=patch_height, p2=patch_width, h=image_height//patch_height , w=image_width//patch_width)

        #self.to_latent = nn.Identity()

        #self.mlp_head = nn.Linear(dim, num_classes)

    def forward(self, img):
        _, _, w, h = img.shape

        x = self.to_patch_embedding(img)
        b, n, pc = x.shape

        x += self.pos_embedding[:, :n]


        x = self.hw_rearrange(x)
        center = h//2//self.patch_size
        

        for dim in range(h//2//self.patch_size, 0, -1):  
            gtop = center+dim-1
            top = x[:, :, center - dim, center - dim:gtop + 1]
            bottom = x[:, :, gtop, center - dim:gtop + 1]
            left = x[:, :, center - dim + 1:gtop, center - dim]
            right = x[:, :, center - dim + 1:gtop, gtop]

            temp = torch.cat((top, left, bottom, right),dim=2)

            temp += self.dis_embedding[center-dim, :, :temp.shape[2]]
            if(dim==h//2//self.patch_size):
                out_token=temp
            else:
                out_token=torch.cat((out_token,temp),dim=2)
            # if(dim==1):
            #     temp = x[:, :, center, center].view(b,pc,1)
            #     print("temp",temp.shape)
            #     temp += self.dis_embedding[center-1, :, :temp.shape[2]]
            #     out_token=torch.cat((out_token, temp),dim=2)
            #     break

        x = out_token.transpose(1,2)

        x = self.to_dis_embedding(x)

        x = self.dropout(x)

        x = self.transformer(x)

        #x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        #x = self.to_latent(x)
        x = self.reversed_re(x)
        return x