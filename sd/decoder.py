import torch
from torch import nn
from torch.nn import functional as F 
from attention import SelfAttention

class VAE_AttentionBlock(nn.Module):
    def __init__(self,channels:int):
        super().__init__()
        self.groupnorm=nn.GroupNorm(32,channels) #グループ正規化　32グループに分けて各グループで正規化をする
        #あるステップの出力が0~1で次のステップの入力が3~5などだと損失関数が変動しすぎる
        self.attention=SelfAttention(1,channels)  #SelfAttention(ヘッドの数、このchannnelは何か)
    
    #残差接続になっている
    #まずVAE_AttentioinBlockへの入力xをresidueという変数に保存
    #xをSelfAttentionに通した出力と最初に保存したresidueを足している
    def forward(self,x:torch.Tensor) -> torch.Tensor:  #テンソルxを入力、出力もテンソル

        #入力xの形状:(Batch_Size,Features,Height,Width)

        residue=x  

        n,c,h,w=x.shape

        #(Batch_Size,Features,Height,Width)->(Batch_Size,Features,Height*Width)
        x=x.view(n,c,h*w)  #viewで入力テンソルxの形状を変える
        
        #形状の変化　Batch_Size,Features,Height*Width)->(Batch_Size,　Height*Width,　　Features,)
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
        self.groupnorm_1=nn.GroupNorm(32,in_channles)  #グループ正規化　32グループに分割して各グループ内で正規化
        self.conv_1=nn.Conv2d(in_channles,out_channels,kernel_size=3,padding=1)

        self.groupnorm_2=nn.GroupNorm(32,out_channels)
        self.conv_2=nn.Conv2d(in_channles,out_channels,kernel_size=3,padding=1)
        
        #スキップ接続
        #層を飛ばして接続する
        #入力チャンネル数と出力チャンネル数が異なる時、
        #なぜか
        #xのチャンネル数は入力チャンネル数、forwardメソッドのreturnでx+residueとしたときに次元数が合わないかららしい
        if in_channles==out_channels:
            self.residual_layer=nn.Identity()  #Identity()を使い入力と出力が同じ
        else:
            self.residual_layer=nn.Conv2d(in_channles,out_channels,kernel_size=1,padding=0)

    def forward(self,x: torch.Tensor)  ->torch.Tensor:  #入力がテンソルxで出力もテンソル
        #入力xの形状: (Batch_Size,In_channels,Height,Width)　ここのInchannelsとはどこのこと

        residue=x  #入力xを残差として保存
        x=self.groupnorm_1(x)  #入力テンソルをグループ正規化

        x=F.silu(x) #活性化関数　なぜか学習プロセスが安定するらしい

        x=self.conv_1(x)  #paddingしてから畳み込みなので画像のサイズは同じ

        x=self.groupnorm_2(x)

        x=F.silu(x)

        x=self.conv_2(x) #形状は同じ

        return x+self.residual_layer(residue) 

        #スキップ接続

#デコーダー内に残差ブロックがいくつもあるのはなぜか
    
#VAEのデコーダー
#nn.sequentialとすることで各層を順番に実行
#VAEの図
class VAE_Decoder(nn.Sequential):

    def __init__(self):
        super().__init__(

            #デコーダーでは潜在表現の次元数から元の画像のサイズに拡大する
            #conv2d(入力チャンネル数、出力チャンネル数、カーネルサイズ、パディング)
            nn.Conv2d(4,4,kernel_size=1,padding=0),

            nn.Conv2d(4,512,kernel_size=3,padding=1),

            VAE_ResidualBlock(512,512),

            VAE_AttentionBlock(512,),

            VAE_ResidualBlock(512,512),
            VAE_ResidualBlock(512,512),
            VAE_ResidualBlock(512,512),

            #畳み込みがないので(Batch,512,Height/8,Width/8)
            VAE_ResidualBlock(512,512),
            
            #(Batch_Size,512,Height/8,Width/8) -> (Batch_Size,512,Height/4,Width/4)
            nn.Upsample(scale_factor=2),  #upsampleは高さと幅を増やす　この場合は2倍にしている

            nn.Conv2d(512,512,kernel_size=3,padding=1),

            VAE_ResidualBlock(512,512),
            VAE_ResidualBlock(512,512),
            VAE_ResidualBlock(512,512),
            
            #(Batch_Size,512,Height/4,Width/4) -> (Batch_Size,512,Height/2,Width/2)
            nn.Upsample(scale_factor=2),  
            nn.Conv2d(512,512,kernel_size=3,padding=1),
            
            #特徴の数を減らす
            VAE_ResidualBlock(512,256),
            VAE_ResidualBlock(256,256),
            VAE_ResidualBlock(256,256),
            
            #(Batch_Size,256,Height/2,Width/2) -> (Batch_Size,256,Height,Width)
            nn.Upsample(scale_factor=2),
            nn.Conv2d(256,256,kernel_size=3,padding=1),

            VAE_ResidualBlock(256,128),
            VAE_ResidualBlock(128,128),
            VAE_ResidualBlock(128,128),

            nn.GroupNorm(32,128), #32グループ、特徴量は128次元

            nn.SiLU(),
            
            #(Batch_Size,128,Height,Width) -> (BAtch_Size,3,Height,Width)
            #入力の128チャンネルからRGBの3チャンネルにする    conv2d(入力チャンネル、出力チャンネル、カーネルサイズ、パディング)
            nn.Conv2d(128,3,kernel_size=3,padding=1)
        )

    def forward(self,x:torch.Tensor)  ->torch.Tensor: #入力がテンソルで出力もテンソル

        x/=0.18215  #エンコーダーの最後の部分でx*=0.1825としているので、デコーダーで割る
        for module in self:  #初期化の部分の各層が順にmoduleになり計算される
            x=module(x)
        return x


