import torch
from torch import nn
from torch.nn import functional as F #動画ではここを書いた後pythonのバージョン指定をしていた
from decoder import VAE_AttentionBlock,VAE_ResidualBlock

class VAE_Encoder(nn.Sequential):
    def __init__(self):
        #super().__init__()の)は動画内ではこの形になっていた
        super().__init__(
            #VAEのEncoderの形状
            #3チャンネル,512,512
            #(Batch_Size,Channel,Height,Width)ー> (Batch_Size,128,Height,Width) 元画像の周囲にパディングをしたので畳みこみをしても大きさが元の画像と同じ
            nn.Conv2d(3,128,kernel_size=3,padding=1),#畳み込みを定義、最初は3チャンネル、動画では畳み込みの説明があった
            
            #次のブロック  第一引数は入力チャンネル数,第二引数は出力チャンネル数
            
            #(Batch_Size,128,Height,Width)ー> (Batch_Size,128,Height,Width)
            #残差ブロック
            #CNNは層を深くすると勾配消失があったが残差ブロックを使うと勾配消失が起きにくくなる
            #なぜ　調べてもよく分からないので質問
            VAE_ResidualBlock(128,128),  #畳み込みとnormalizationが合わさっている
            
            #(Batch_Size,128,Height,Width)ー> (Batch_Size,128,Height 　/  2 , Width  /  2) 
            #strideが2なので大きさが1/2になる
            nn.Conv2d(128,128,kernel_size=3,stride=2,padding=0),#1行前でoutputが128チャンネルなのでこの行は入力128チャンネル   ネットのonvolutional visualizerというのを使って説明していた
            
            #(Batch_Size,128,Height/2 ,Width/2  )ー> (Batch_Size,256,Height 　/  2 , Width  /  2) 
            VAE_ResidualBlock(128,256), #featureの数を増やした　入力の特徴量が128で出力の特徴量が256ということ
            
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
            #512次元の特徴量マップが入力、アタンションで入力特徴量の中から特定の情報に焦点を当てる
            #焦点を当てた情報から有意義な特徴量表現を生成しVAEモデルの後続部分に渡す
            VAE_AttentionBlock(512),
            
            #(Batch_Size,512,Height/8 ,Width/8  )ー> (Batch_Size,512,Height 　/  8 , Width  /  8) 
            VAE_ResidualBlock(512,512),

            #(Batch_Size,512,Height/8 ,Width/8  )ー> (Batch_Size,512,Height 　/  8 , Width  /  8) 
            #バッチ正規化とは異なる　第一引数はグループの数、チャンネルが幾つのグループに分割されるか　第二引数は正規化を適用するチャンネルの総数
            nn.GroupNorm(32,512),

            #(Batch_Size,512,Height/8 ,Width/8  )ー> (Batch_Size,512,Height 　/  8 , Width  /  8) 
            nn.SiLU(),    #この活性化関数を用いるのはこのようなモデルで一般的によく使われなぜこのようなモデルに適しているのかは不明らしい,この人のLAMAの動画でも紹介したらしい

            #(Batch_Size,512,Height/8 ,Width/8  )ー> (Batch_Size,8,Height 　/  8 , Width  /  8) 
            #strideは1でパディングをしているので画像の大きさは変わらない
            #conv2d(入力チャンネル数、出力チャンネル数、カーネルサイズ、パディング)
            #padding=1なので入力テンソルの各辺に1ピクセルずつ0パディング
            nn.Conv2d(512,8,kernel_size=3,padding=1)
            
            #(Batch_Size,8,Height/8 ,Width/8  )ー> (Batch_Size,8,Height /  8 , Width  /  8) 
            #padding=0なのでパディングは追加されない
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
                
                #各モジュールからstride属性を取得というのはどういうことか

                #(Padding Left, Padding Right, Padding Top, Padding Bottom)
                x= F.pad(x,(0,1,0,1))  #なぜ(0,1,0,1)
            x=module(x)

        #(Batch_Size,8,Height/8 ,Width/8  )  -> two tensors of shape (Baatch_size,4, Height, Width)
        #指定されたdimに沿って入力テンソルxを2このテンソルに均等に分割する関数
        #分割した1つめのテンソルはmeanに、2つめのテンソルはlog_varianceとする
        mean,log_varianece=torch.chunk(x,2,dim=1) 

        #(Batch_Size,4,Height/8,Width/8) -> (Batch_Size,4,Height/8, Width / 8)
        variance=log_varianece.exp()  #logvarianceの各要素に対して指数を適用 e^(log_variance)
        
        #サイズ変化はなし
        stdev=variance.sqrt() #平方根

        #Z= N(0,1)から取り出す N(mean,variance)
        #x= mean +stdev *Z
        x=mean +stdev*noise  #noiseはforwardの引数でテンソル型、正規分布からサンプリングされている

        #定数によって出力をscale
        x*=0.18215  #なぜこの値なのか

        return x
        




