import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
# class FusionNet(nn.Module):
#     def __init__(self,feature_list,num_class,out_shape=(8,3)):
#         super(FusionNet,self).__init__()
#         self.num=len(feature_list)
#         self.feature_names=[ 'top','front','rgb']
#         self.roi_features_list=nn.ModuleList()
#         for n in range(self.num):
#             feature=feature_list[n][0]
#             roi=feature_list[n][1]
#             pool_height =feature_list[n][2]
#             pool_width=feature_list[n][3]
#             pool_scale=feature_list[n][4]
#             if (pool_height==0 or pool_width==0):
#                 continue
            
class CrossAttention(nn.Module):
    def __init__(self, emb_dim):
        # super(SelfAttention, self).__init__()
        super().__init__()

        self.emb_dim = emb_dim

        self.Wq = nn.Linear(emb_dim, emb_dim, bias=False)
        self.Wk = nn.Linear(emb_dim, emb_dim, bias=False)
        self.Wv = nn.Linear(emb_dim, emb_dim, bias=False)

        self.fc = nn.Linear(emb_dim, emb_dim)

    def forward(self, x1, x2,pad_mask=None):
        # [batch_szie, seq_len, emb_dim] = [3, 5, 512]
        print(x1.shape)
        Q = self.Wq(x1)
        K = self.Wk(x2)
        V = self.Wv(x2)

        att_weights = torch.bmm(Q, K.transpose(1, 2))   # [batch_szie, seq_len, seq_len] = [3, 5, 5]
        att_weights = att_weights / math.sqrt(self.emb_dim)

        if pad_mask is not None:
            att_weights = att_weights.masked_fill(pad_mask, -1e9)

        att_weights = F.softmax(att_weights, dim=-1)
        output = torch.bmm(att_weights, V)   # [batch_szie, seq_len, emb_dim] = [3, 5, 512]
        output = self.fc(output)
        print(output.shape)

        return output, att_weights



class Cross_MultiAttention(nn.Module):
    def __init__(self, in_channels, emb_dim, num_heads, att_dropout=0.0, aropout=0.0):
        super(Cross_MultiAttention, self).__init__()
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.scale = emb_dim ** -0.5

        assert emb_dim % num_heads == 0, "emb_dim must be divisible by num_heads"
        self.depth = emb_dim // num_heads

        self.proj_in = nn.Conv2d(in_channels, emb_dim, kernel_size=1, stride=1, padding=0)

        self.Wq = nn.Linear(emb_dim, emb_dim)
        self.Wk = nn.Linear(emb_dim, emb_dim)
        self.Wv = nn.Linear(emb_dim, emb_dim)

        self.proj_out = nn.Conv2d(emb_dim, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x, context, pad_mask=None):
        '''

        :param x: [batch_size, c, h, w]
        :param context: [batch_szie, seq_len, emb_dim]
        :param pad_mask: [batch_size, seq_len, seq_len]
        :return:
        '''
        
        b, c, h, w = x.shape
        batch_size=b

        x = self.proj_in(x)   # [batch_size, c, h, w] = [3, 512, 512, 512]
        x = rearrange(x, 'b c h w -> b (h w) c')   # [batch_size, h*w, c] = [3, 262144, 512]

        Q = self.Wq(x)  # [batch_size, h*w, emb_dim] = [3, 262144, 512]
        K = self.Wk(context)  # [batch_szie, seq_len, emb_dim] = [3, 5, 512]
        V = self.Wv(context)

        Q = Q.view(batch_size, -1, self.num_heads, self.depth).transpose(1, 2)  # [batch_size, num_heads, h*w, depth]
        K = K.view(batch_size, -1, self.num_heads, self.depth).transpose(1, 2)  # [batch_size, num_heads, seq_len, depth]
        V = V.view(batch_size, -1, self.num_heads, self.depth).transpose(1, 2)

        # [batch_size, num_heads, h*w, seq_len]
        att_weights = torch.einsum('bnid,bnjd -> bnij', Q, K)
        att_weights = att_weights * self.scale

        if pad_mask is not None:
            # 因为是多头，所以mask矩阵维度要扩充到4维  [batch_size, h*w, seq_len] -> [batch_size, nums_head, h*w, seq_len]
            pad_mask = pad_mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
            att_weights = att_weights.masked_fill(pad_mask, -1e9)

        att_weights = F.softmax(att_weights, dim=-1)
        out = torch.einsum('bnij, bnjd -> bnid', att_weights, V)
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.emb_dim)   # [batch_size, h*w, emb_dim]

        out = rearrange(out, 'b (h w) c -> b c h w', h=h, w=w)   # [batch_size, c, h, w]
        out = self.proj_out(out)   # [batch_size, c, h, w]

        return out, att_weights