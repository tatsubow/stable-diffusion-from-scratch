import torch
from torch import nn
from torch.nn import functional as F 
from attention import SelfAttention

class VAE_AttentionBlock(nn.Module):
    def __init__(self,channels:int):
        super().__init__()
        self.groupnorm=nn.GroupNorm(32,channels) #レイヤー正規化　の説明がスライド付きであった
        #あるステップの出力が0~1で次のステップの入力が3~5などだと損失関数が変動しすぎる
        self.attention=SelfAttention(1,channels)

    def forward(self,x:torch.Tensor) -> torch.Tensor:

        #x:(Batch_Size,Features,Height,Width)

        residue=x

        n,c,h,w=x.shape

        #(Batch_Size,Features,Height,Width)->(Batch_Size,Features,Height*Width)
        x=x.view(n,c,h*w)
        
        #Batch_Size,Features,Height*Width)->(Batch_Size,　Height*Width,　　Features,)
        x=x.transpose(-1,-2)
       
       #形状は同じ
        x=self.attention(x)
        
        #Batch_Size,Features,Height*Width)->(Batch_Size,　Features,  Height, Width)
        x=x.transpose(-1,-2)
        
        #(Batch_Size, Features,Height*Width) -> (Batch_Size,Features, Height , Width)
        x=x.view((n,c,h,w))

        x+=residue

        return x


class VAE_ResidualBlock(nn.Module):
    def __init__(self,in_channles,out_channels):
        super().__init__()
        self.groupnorm_1=nn.GroupNorm(32,in_channles)
        self.conv_1=nn.Conv2d(in_channles,out_channels,kernel_size=3,padding=1)

        self.groupnorm_2=nn.GroupNorm(32,out_channels)
        self.conv_2=nn.Conv2d(in_channles,out_channels,kernel_size=3,padding=1)

        if in_channles==out_channels:
            self.residual_layer=nn.Identity()
        else:
            self.residual_layer=nn.Conv2d(in_channles,out_channels,kernel_size=1,padding=0)

    def forward(self,x: torch.Tensor)  ->torch.Tensor:
        #x: (Batch_Size,In_channels,Height,Width)


#ここは何をしているのか
        residue=x
        x=self.groupnorm_1(x)

        x=F.silu(x)

        x=self.conv_1(x)  #paddingしてから畳み込みなので画像のサイズは同じ

        x=self.groupnorm_2(x)

        x=F.silu(x)

        x=self.conv_2(x)

        return x+self.residual_layer(residue)

        #スキップ接続

