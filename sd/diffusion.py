import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention,CrossAttention

#super__init__()の意味

class Upsample(nn.Module):

    def __init__(self,channels:int):
        super().__init__()  
        self.conv=nn.Conv2d(channels,channels,kernel_size=3,padding=1)
    
    def forward(self,x):
        x=F.interpolate(x,scale_factor=2,mode="nearest")  #interpolate(入力、scale_factor=2倍,mode=nearest　アップサンプリング時に各ピク節の値を最も近い入力ピクセルの値に置き換える方法)
        return self.conv(x)   #アップサンプリングされた画像xを畳み込み

#時間情報を埋め込むクラス
#時間情報は通常連続値だが、ニューラルネットワークは離散値を扱うので線形変換を用いて時間情報を高次元ベクトルに変換する　というのはどういうことか

class TimeEmbedding(nn.Module):
    #Unetはノイズ化されるところで時間埋め込みを受け取る
    def __init__(self,n_embd:int):
        super().__init__()   #Linear(入力特徴量数、出力特徴量数)
        self.linear_1=nn.Linear(n_embd,4*n_embd)#特徴量数4倍になる 特徴量が４倍に増えるというのはどういうこと
        self.linear_2=nn.Linear(4*n_embd,4*n_embd)
    
    def forward(self,x:torch.Tensor)->torch.Tensor:
        #x:(1,320)

        x=self.linear_1(x)   #特徴量数が４倍になる

        x=F.silu(x)  

        x=self.linear_2(x) 

        #(1,1280)
        return x 

#残差ブロック
class UNET_ResidualBlock(nn.Module):
     
     #この中の意味
    #Unetはtime_embeddingとプロンプトと入力の3つを受け取る
    def __init__(self,in_channles,out_channels:int,n_time=1280):
        super().__init__()
        #GroupNorm(幾つのグループに分割するか、入力テンソルのチャンネル数)
        self.groupnorm_feature=nn.GroupNorm(32,in_channles)  #各グループの特徴マップが平均0、分散1になる
        self.conv_feature=nn.Linear(n_time,out_channels,kernel_size=3,padding=1) 
        self.linear_time=nn.Linear(n_time,out_channels)

        self.groupnorm_merged=nn.GroupNorm(32,out_channels)
        self.conv_merged=nn.Conv2d(out_channels,out_channels,kernel_size=3,padding=1)

        if in_channles==out_channels:
            self.residual_layer=nn.Identity()
        
        #入力と出力のチャンネル数を同じにして足し算をできるようにする
        else:
            self.residual_layer=nn.Conv2d(in_channles,out_channels,kernel_size=1,padding=0)

    def forward(self,feature,time):
        #feature  (Batch_size,入力チャンネル、高さ、幅)
        #time (1,1280)    timeは何か
        resiidue=feature

        feature=self.groupnorm_feature(feature)  #特徴をグループ正規化　

        feature=F.silu(feature)  #活性化関数に通す

        feature=self.conv_feature(feature)  #畳み込み

        time=F.silu(time) #時間を活性化関数に通す

        time=self.linear_time(time) #time=1280が入力で出力outchannels とは何のこと
        
        #featureとtimeを足すが、timeにはバッチサイズと入力チャンネル数がないので形状を揃える
        #unsqueeze(-1)最後の次元に2つ次元を追加 ブロードキャストを行なっている
        #足し算したときにfeatureは(バッチ、入力チャンネル数。高さ、幅)、timeは(1 バッチサイズを模しているわけではない、1280 入力チャンネル数、高さの次元になる、幅の次元になる)
        merged=feature+time.unsqueeze(-1).unsqueeze(-1)

        merged=self.groupnorm_merged(merged)

        merged=F.silu(merged)
 
        merged=self.conv_merged(merged)

        return merged+self.residual_layer(resiidue)

class UNET_AttentionBlock(nn.Module):

    #n_head アテンションヘッドの数　入力をいくつの部分に分けて処理するかを示す
    #n_embd   各アテンションヘッドの埋め込みの次元数　　アテンションヘッドごとの特徴ベクトルのサイズ
    #d_context  これは何か

    def __init(self,n_head:int,n_embd:int,d_context=768):  #d_contextはtransormerモデルでよく使われrているらしい
        super().__init__()
        channels=n_head*n_embd

        self.groupnorm=nn.nn.GroupNorm(32,channel,eps=1e-6)
        self.conv_input=nn.Conv2d(channels,channels,kernel_size=1,padding=0)

        self.layernorm_1=nn.LayerNorm(channels)
        self.attention_1=SelfAttention(n_head,channels,in_proj_bias=False)
        self.layernorm_2=nn.LayerNorm(channels)
        self.attention_2=CrossAttention(n_head,channels,d_context,in_proj_bias=False)
        self.layernorm_3=nn.LayerNorm(channels)
        self.linear_geglu_1=nn.Linear(channels,4*channels*2)
        self.linear_geglu_2=nn.Linear(4*channels,channels)

        self.conv_output=nn.COnv2d(channels,channels,kernel_size=1,padding=0)

    def forward(self,x,context):
        #x:(Batch_Size,Seq_Len,Dim)
        #context (Batch_Size,Seq_Len,Dim)
        residue_long=x

        x=self.groupnorm(x)

        self.conv_input(x)

        n,c,h,w=x.shape
        
        #(Batch_Size,Features,Height*Width)->(Batch_Size,Features,Height*Width)
        x=x.view((n,c,h*w))
        
        #(Batch_Size,Features,Height*Width)->(Batch_Size,Height*Width,Features)
        x=x.transpose(-1,-2)
        
        #スキップ接続とセルフアテンション
        #スキップ接続の残差　アテンションを通らない矢印
        residue_short=x  #残差を保存
        x=self.layernorm_1(x)  #スキップ接続のMulti-Head Attentionの部分
        self.attention_1(x)  #セルフアテンション
        #スキップ接続　アテンションを通らない矢印
        x+=residue_short #アテンションを通った後のxに保存してあった残差を足す

        #スキップ接続とクロスアテンション
        residue_short=x   #1つ目のスキップ接続の出力についてスキップ接続を用いた残差ブロック
        #Normalization+self attention with skip connection
        x=self.layernorm_1(x)  #引数はチャンネル数
        self.attention_2(x,context)  #潜在表現とプロンプトのクロスアテンション　引数はヘッド数、チャンネル数、d_context
        x+=residue_short
        
        #スキップ接続とレイヤー正規化
        residue_short=x
        #feed forward
        x=self.layernorm_3(x) #レイヤー正規化
        x,gate=self.linear_geglu_1(x).chunk(2,dim=-1)
        x=x*F.gglu(gate)
        x=self.linear_geglu_2(x)
        x+=residue_short

        x=x.tranpose(-1,-2)  #クロスアテンションを使うために転置していた部分を元の状態にする
        
        x=x.view((n,c,h,w))  #テンソルはバッチサイズ、チャンネル数、高さ、幅に再形成される

        return self.conv_output(x)+residue_long  #forwardメソッドの最初で定義したresidue_longを足す


class SwitchSequential(nn.Sequential):

    def forward(self,x:torch.Tensor,context:torch.Tensor,time:torch.Tensor)->torch.Tensor:
        
        #入力のすべてのxとcontextについて繰り返し
        for layer in self:
            #isinstance関数は指定されたオブジェクトが指定のものの場合にTrueを返す
            if isinstance(layer,UNET_AttentionBlock):  #layerがUNET_AttentionBlockの時
                x=layer(x,context)  #xとプロンプトの情報とクロスアテンションで計算 　UNET_AttentionBlockのforwardメソッドはxとcontextを入力とする

            elif isinstance(layer,UNET_ResidualBlock):   #layerがUNET_ResidualBlockの時
                x=layer(x,time)   #残差ブロックは特徴とtimeを入力とする
           
            else:
                x=layer(x)
            return x
        
class UNET_OutputLayer(nn.Module):  #なぜこれだけ独立しているのか
    def __init__(self,in_channles:int,out_channels:int):
        super().__init__()
        self.groupnorm=nn.GroupNorm(32,in_channles)
        self.conv=nn.Conv2d(in_channles,out_channels,kernel_size=3,padding=1)
    def forward(self,x):
        #x:入力　(Batch_Size,320,Heoght  / 8  ,Width  /  8)

        x=self.groupnorm(x)

        x=F.silu(x)

        x=self.conv(x)

        return x


#Unetに入力して出力されるまでの部分
class Diffusion(nn.Module):
    #timeは潜在表現がノイズ化された時の時刻を記録
    #Unetはノイズ除去のタイムステップを受け取る必要がある
    def __init__(self):
        self.time_embedding=TimeEmbedding(320)
        self.unet=UNET()
        self.final=UNET_OutputLayer(320,4)

    def forward(self,latent:torch.Tensor,context:torch.Tensor,time:torch.Tensor):
        #latent :(Batch_Size,4,Heoght  / 8  ,Width  /  8)  4チャンネルといううのはエンコーダーの出力
        #context プロンプト　:(Batch_Size,Seq_Len,Dim)
        #time:(1,320)  latentがノイズ化された時の時間情報
        
        #(1,320) -> (1,1280)  
        time=self.time_embedding(time)  #timeを時間埋め込みにする　トランスフォーマーのポジショナルエンコーディングとほぼ同じ　どういうことか
        
        #潜在表現を他の潜在表現にする
        #((Batch,4,Heoght  /  8 ,Width / 8) -> (Batch,320,Heoght  /  8 ,Width / 8) )
        #
        output=self.unet(latent,context,time)

        #(Batch,320,Heoght  /  8 ,Width / 8)->(Batch,4,Heoght  /  8 ,Width / 8)
        #エンコーダーで増えた特徴量の次元数を入力の時の特徴量の次元数にする　
        output=self.final(output)
        
        #(Batch,4,Heoght  /  8 ,Width / 8)
        return output  #Unetの出力

class UNET(nn.Module):

    def __init__(self):
        super().__init__()

        self.encoders=nn.Module([

            #(Batch_Size,4,Height  /  8 ,Width  /  8)
            #conv2d(入力チャンネル数、出力チャンネル数、カーネルサイズ、パディング)
            SwitchSequential(nn.Conv2d(4,320,kernel_size=3,padding=1)),  #Unetの図の一番左のチャンネル数を増やす3層を一度に4から320チャンネルにしている

            SwitchSequential(UNET_ResidualBlock(320,320),UNET_AttentionBlock),

            SwitchSequential(UNET_ResidualBlock(320,320),UNET_AttentionBlock),
            

            #エンコーダーなので畳み込みで形状を小さくする、特徴量数は増える
            #潜在表現の形状(Batch_Size,4,Height  /  8 ,Width  /  8)->(Batch_Size,4,Height  /  16 ,Width  /  16)
            SwitchSequential(nn.Conv2d(320,320,kernel_size=3,stride=2,padding=1)),
            #画像のサイズを小さくしていく
            #UNET_AttentionBlock(8,80)はアテンションのヘッド数8、各アテンションヘッドが80次元の埋め込みを使うということ　全体としてのチャンネル数は640になる
            #ここではUNET_AttentionBlockの初期化の部分

            #UNET_ResidualBlock(320,640)は入力チャンネルが320、出力が640ということ、何を入力するかは別の部分で指定される

            SwitchSequential(UNET_ResidualBlock(320,640),UNET_AttentionBlock(8,80)),   

            SwitchSequential(UNET_ResidualBlock(640,640),UNET_AttentionBlock(8,80)),

            
            #(Batch_Size,4,Height  /  16 ,Width  /  16)->(Batch_Size,4,Height  /  32 ,Width  /  32)
            SwitchSequential(nn.Conv2d(640,640,kernel_size=3,stride=2,padding=1)),

            SwitchSequential(UNET_ResidualBlock(640,1280),UNET_AttentionBlock(8,160)),

            SwitchSequential(UNET_ResidualBlock(1280,1280),UNET_AttentionBlock(8,160)),


            #(Batch_Size,4,Height  /  32 ,Width  /  32)->(Batch_Size,4,Height  /  64 ,Width  /  64)
            SwitchSequential(nn.Conv2d(1280,1280,kernel_size=3,stride=2,padding=1)),

            SwitchSequential(UNET_ResidualBlock(1280,1280)),

            SwitchSequential(UNET_ResidualBlock(1280,1280))
        ])
        
        #Unetの図の一番下の部分
        self.bottleneck=SwitchSequential(
            UNET_ResidualBlock(1280,1280),

            UNET_AttentionBlock(8,160),

            UNET_ResidualBlock(1280,1280),
        )
        #bottleneckの出力の特徴量次元数は1280次元
   
        self.decoders=nn.ModuleList([
            #(Batch_Size,2560,Height  /  64 ,Width  /  64)->(Batch_Size,2560,Height  /  64 ,Width  /  64)
            SwitchSequential(UNET_ResidualBlock(2560,1280)),#bottleneckの出力とスキップ接続で合わせて2560　Unetの図の一番下の部分デコーダーの最初
            
            SwitchSequential(UNET_ResidualBlock(2560,1280)),

            SwitchSequential(UNET_ResidualBlock(2560,1280),Upsample(1280)),

            SwitchSequential(UNET_ResidualBlock(2560,1280),UNET_AttentionBlock(8,160)),

            SwitchSequential(UNET_ResidualBlock(2560,1280),UNET_AttentionBlock(8,160)),

            SwitchSequential(UNET_ResidualBlock(1920,1280),UNET_AttentionBlock(8,160),Upsample(1280)),

            SwitchSequential(UNET_ResidualBlock(1920,640),UNET_AttentionBlock(8,80)),
            
            SwitchSequential(UNET_ResidualBlock(1280,640),UNET_AttentionBlock(8,80)),

            SwitchSequential(UNET_ResidualBlock(960,640),UNET_AttentionBlock(8,40)),

            SwitchSequential(UNET_ResidualBlock(960,320),UNET_AttentionBlock(8,40)),

            SwitchSequential(UNET_ResidualBlock(640,320),UNET_AttentionBlock(8,80)),

            SwitchSequential(UNET_ResidualBlock(640,320),UNET_AttentionBlock(8,40)),

        ])