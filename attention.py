import torch
from torch import nn
from torch.nn import functional as F

#selfattentionの説明がスライドであった

class SelfAttention(nn.Module):
    def __init__(self,n_heads:int,d_embed:int , in_proj_bias=True,out_proj_bias=True)
        super().__init__()
        self.in_proj=nn.Linear(d_embed*3,3*d_embed,bias=in_proj_bias)
        self.out_proj=nn.Linear(d_embed,bias=out_proj_bias)
        self.n_heads=n_heads
        self.d_head=d_embed//n_heads

    def forward(self,x:torch.Tensor,causal_mask=False):
        input_shape=x.shape
        batch_size,sequence_length,d_embed=input_shape

        intermim_shape=(batch_size,sequence_length,self.n_heads,self.d_head)
        
        #(Batch_size,seq_len,dim) -> (Batch_size, seq_len, dim*3)  -> 3tensprs of shape(Batch_size,Seq_len, Dim)
        q,k,v=self.in_proj(x).chunk(3,dim=-1)
        
        #(Batch_Size,Seq_len,Dim)  -> (Batch_size, Seq_len ,H,  Dim/H)  -> (Batch_Size,H,Seq_len,Dim/H)
        q=q.view(intermim_shape).transpose(1,2)
        k=k.view(intermim_shape).transpose(1,2)
        v=v.view(intermim_shape).transpose(1,2)

        #(Batch_Size,H,Seq_Len,Seq_Len)
        weight=q@k.transpose(-1,-2)

        if causal_mask:
            #Mask where t
            mask=torch.ones_like(weight,dtype=torch.bool).triu(1)

            #1:43:00あたりから再開
