import torch
from torch import nn
from torch.nn import functional as F #動画ではここを書いた後pythonのバージョン指定をしていた
from decoder import VAE_AttentionBlock,VAE_ResidualBlock

class VAE_Encoder(nn.Sequential):
    def __init__(self):
        super().__init(
            #VAEのEncoderの形状
            #3チャンネル,512,512
            #(Batch_Size,Channel,Height,Width)ー> (Batch_Size,128,Height,Width) 元画像の周囲にパディングをしたので畳みこみをしても大きさが元の画像と同じ
            nn.Conv2d(3,128,kernel_size=3,padding=1),#畳み込みを定義、最初は3チャンネル、動画では畳み込みの説明があった
            
            #次のブロック  第一引数は入力チャンネル数,第二引数は出力チャンネル数
            
            #(Batch_Size,128,Height,Width)ー> (Batch_Size,128,Height,Width) 
            VAE_ResidualBlock(128,128),  #畳み込みとnormalizationが合わさっている
            
            #(Batch_Size,128,Height,Width)ー> (Batch_Size,128,Height 　/  2 , Width  /  2) 
            #strideが2なので大きさが1/2になる
            nn.Conv2d(128,128,kernel_size=3,stride=2,padding=0),#1行前でoutputが128チャンネルなのでこの行は入力128チャンネル   ネットのonvolutional visualizerというのを使って説明していた
            
            #(Batch_Size,128,Height/2 ,Width/2  )ー> (Batch_Size,256,Height 　/  2 , Width  /  2) 
            VAE_ResidualBlock(128,256), #featureの数を増やした
            
            #(Batch_Size,256,Height/2 ,Width/2  )ー> (Batch_Size,256,Height 　/  2 , Width  /  2) 
            VAE_ResidualBlock(256,256),

            #各ステップで画像のサイズは小さくなるが、特徴の数は増える　それぞれのピクセルはより多くの情報をもつ

            #(Batch_Size,256,Height/2 ,Width/2  )ー> (Batch_Size,256,Height 　/  4 , Width  /  4) 
            nn.Conv2d(256,256,kernel_size=3,stride=2,padding=0),

            #(Batch_Size,256,Height/4 ,Width/4  )ー> (Batch_Size,512,Height 　/  4 , Width  /  4) 
            VAE_ResidualBlock(256,512),

            #ResidualBlockは何をしているのか
            #(Batch_Size,512,Height/4 ,Width/4  )ー> (Batch_Size,512,Height 　/  4 , Width  /  4) 
            VAE_ResidualBlock(512,512),

            #(Batch_Size,512,Height/4 ,Width/4  )ー> (Batch_Size,512,Height 　/  8 , Width  /  8) 
            nn.Conv2d(512,512,kernel_size=3,stride=2,padding=0),

            VAE_ResidualBlock(512,512),

            VAE_ResidualBlock(512,512),

            #(Batch_Size,512,Height/8 ,Width/8  )ー> (Batch_Size,512,Height 　/  8 , Width  /  8) 
            VAE_ResidualBlock(512,512),

            #なぜ何回もResidualBlockがあるのか
            
            #それぞれのピクセルに対するself attentionとして動く
            #Attentionは文中でトークンを関連づける(?)1:00:40 　ピクセルの列として捉えられAttentionはそれぞれのピクセルを関連づける方法として捉えられる
            #畳み込みでも近接するピクセルは関連づけられたが、Attentionは近接していないピクセルも関連づける
            VAE_AttentionBlock(512),
            
            #(Batch_Size,512,Height/8 ,Width/8  )ー> (Batch_Size,512,Height 　/  8 , Width  /  8) 
            VAE_ResidualBlock(512,512),

            #(Batch_Size,512,Height/8 ,Width/8  )ー> (Batch_Size,512,Height 　/  8 , Width  /  8) 
            nn.GroupNorm(32,512),

            #(Batch_Size,512,Height/8 ,Width/8  )ー> (Batch_Size,512,Height 　/  8 , Width  /  8) 
            nn.SiLU(),    #これを用いるのはこのようなモデルで一般的によく使われなぜこのようなモデルに適しているのかは不明らしい,この人のLAMAの動画でも紹介したらしい

            #(Batch_Size,512,Height/8 ,Width/8  )ー> (Batch_Size,8,Height 　/  8 , Width  /  8) 
            #strideは1でパディングをしているので画像の大きさは変わらない
            nn.Conv2d(512,8,kernel_size=3,padding=1)
            
            #(Batch_Size,8,Height/8 ,Width/8  )ー> (Batch_Size,8,Height /  8 , Width  /  8) 
            nn.Conv2d(8,8,kernel_size=1,padding=0)

            #なぜカーネルサイズ1で畳み込みしてるのか、畳み込みになっていないのではないか
        )

#引数の型はテンソル、xとnoiseが関数の引数  ->tensorは関数の戻り値の型指定
#xとnoiseを入力してテンソル型を出力ということ
def forward(self,x:torch.Tensor,noise:torch.Tensor)->torch.Tensor:
    #x:(Batch_size,channnel,Height,Width)
    #noise:(Batch_size,Out_Channels,Height  /8 ,Width  /  8)

    #getattr関数は、第一引数に指定されたオブジェクトから、第二引数で指定された名前の属性を取得しようとします。もし属性が存在しない場合は、第三引数で指定されたデフォルト値（この場合はNone）を返します。
    for module in self:
        if getattr(module,'stride',None)==(2,2): #各モジュールからstride属性を取得しその値が(2,2)であるかどうかを確認
            #(Padding Left, Padding Right, Padding Top, Padding Bottom)
            x= F.pad(x,(0,1,0,1))  #なぜ(0,1,0,1)
        x=module(x)
    
    #ここから数行は何をしているのか

    #(Batch_Size,8,Height/8 ,Width/8  )  -> two tensors of shape (Baatch_size,4, Height, Width)
    mean,log_varianece=torch.chun(x,2,dim=1)

    #(Batch_Size,4,Height/8,Width/8) -> (Batch_Size,4,Height/8, Width / 8)
    variance=log_varianece.exp()
    
    #サイズ変化はなし
    stdev=variance.sqrt()

    #Z= N(0,1)から取り出す N(mean,variance)
    #x= mean +stdev *Z
    x=mean +stdev*noise

    #定数によって出力をscale
    x*=0.18215

    return x
    




