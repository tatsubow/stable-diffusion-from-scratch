import torch
from torch import nn
from torch.nn import functional as F

#self attentionは基本的にはそれぞれのトークンを関連づける
#最初はトークンの列、サイズd_modelの埋め込みを持つ、それをクエリ、キー、バリューに変換
#次にヘッドの数に分割する　今回のアテンションはヘッド数が１つ
#それぞれのヘッドについてアテンションを計算する
#クエリ、キー、バリューは全て同じ行列から作られる
#今回はトークンではなくピクセル　それぞれのピクセルのチャンネルの数=ピクセルの埋め込み

class SelfAttention(nn.Module):

    #引数はn_heads=ヘッドの数、d_embed=埋め込み、それぞれのピクセルのチャンネル数、それぞれのピクセルはそのピクセルの情報をとらえた多くのチャンネルによって表現されている
    def __init__(self,n_heads:int,d_embed:int , in_proj_bias=True,out_proj_bias=True):
        super().__init__()

        #WQ,WK,WVを3つの異なる行列として表現するのではなく一つの大きなlinear layerとする
        #これはattentionを適用する前の入力のprojection
        #スライドのWO行列の形状はdmodel*dmoedl
        #Linear(入力特徴量、出力特徴量、バイアス)
        self.in_proj=nn.Linear(d_embed*3,d_embed*3,bias=in_proj_bias) #self attentionを適用する前に加えるバイアス
        self.out_proj=nn.Linear(d_embed,d_embed, bias=out_proj_bias)  #self attentionを適用した後に加えるバイアス
        self.n_heads=n_heads  #ヘッドの数を保存
        self.d_head=d_embed//n_heads  #マルチヘッドはそれぞれのトークンの埋め込みの一部を見るので割り算　動画の1:36:17
    


    #マスク　あるトークンより前のみを参照するようにする
    def forward(self,x:torch.Tensor,causal_mask=False):
        input_shape=x.shape  #入力の形状を抽出
        batch_size,sequence_length,d_embed=input_shape
        
        #入力の形状から別の形状にする 中間形状 intermidiate shape
        intermim_shape=(batch_size,sequence_length,self.n_heads,self.d_head)
        
        #(Batch_size,seq_len,dim) -> (Batch_size, seq_len, dim*3)  -> 3tensprs of shape(Batch_size,Seq_len, Dim)
        #xはクエリ、キー、バリューが結合された行列
        #self.in_projがWQ,WK,WVにあたる
        q,k,v=self.in_proj(x).chunk(3,dim=-1)  #全てに重みをかけた後に3分割してQ',K',V'にする
        
        #chunkで最後の次元(最後の次元って何か)に沿って3分割する
        #分割された1,k,vにあたる行列の形状を変える
        #ここの形状の変化?
        #view(interim_shape)　(Batch_Size,Seq_len,Dim)  -> (Batch_size, Seq_len ,H,  Dim/H)  
        #transpose(1,2,)　　　(Batch_size, Seq_len ,H,  Dim/H)  -> (Batch_Size,H,Seq_len,Dim/H)
        q=q.view(intermim_shape).transpose(1,2)  #2次元目と3次元目の位置を入れ替える
        k=k.view(intermim_shape).transpose(1,2)
        v=v.view(intermim_shape).transpose(1,2)

        
        #Attention(Q,K,V)の式のQKT

        #入力q,k,vから出力までを表した関数の一部　スライド
        #q       　　　　　　　　(Batch_Size,H,Seq_len,Dim/H)
        #k.transpose(-1,-2)   (Batch_Size,Seq_len,H, Dim/H)
        #weight               (Batch_Size,H,Seq_Len,Seq_Len)

        #q@k.tranpose(-1,-2)でどうweightの形状になるのか

        weight=q@k.transpose(-1,-2)#テンソルの最後の次元と最後から2番目の次元を入れ替える
        
        #未来の時点の情報を考慮しないようにするマスクが作成される
        #因果マスクを適用するかを指定 causal_mask=Trueならマスクが適用
        if causal_mask:
            #weightと同じ形状のテンソルがtorch.one_likeで作成されデータ型はbool　triuで上三角行列を生成　対角線より上のみ1にしそれ以外0にする
            # 1 1 1
            # 0 1 1
            # 0 0 1   
            #あるトークンiに対してi以降のトークンjにあたる部分を1にする

            #weightは4チャンネルで対角線より上というのはどう考えるのか
         
            mask=torch.ones_like(weight,dtype=torch.bool).triu(1) #weightテンソルと同じ形状の真偽値テンソルを作成し、対角線上と対角線より上をTrue 1にする
            
            weight.masked_fill(mask,-torch.inf) #masked_fill関数はマスクのTrueに対応する位置の要素を指定した値にする　ここでは-toch.inf -∞
        
        weight/=mask.sqrt(self.d_head) #Attentionの式の分数の部分、weightはQKT

        weight=F.softmax(weight,dim=-1)  #weightテンソルの対角線より上の部分はマイナス無限で、ソフトマックス関数に通すとほぼ0になる
        
        #(Batch_Size,H,Seq_len,Seq_len)@(Batch_Size,H,Seq_len,Dim/H)->(Batch_Size,H,Seqlen,Dim/H)
        output=weight@v   #weightテンソルの対角線より上の部分は0に近い値のため、未来の情報はoutputにほぼ含まれない
        
        #(Batch_Size,H,Seqlen,Dim/H) ->(Batch_SIze,Sequencelen, H,Dim/H)
        output=output.tranpose(1,2)
        
        output=output.reshape(input_shape) #入力の時のテンソルの形状にする

        output=self.output_proj(output)  #バイアスを加える
        
        #(Batch_Size,Seqlen,Dim)
        return output
    
#セルフアテンションはdiffusion.py Unetの　UNET_AttentionBlockで使われる


#2つの文字列間の関係性などを扱う
#クロスアテンション　クエリが第１sequence  キーとバリューは他のsequence 


#self attention class とcross attention クラスの違い
#セルフアテンションの入力は　x=  クエリ、キー、バリューが合わさった行列
#クロスアテンションの入力は　x=クエリ y=キー、バリュー

class CrossAttention(nn.Module):

    def __init__(self,n_heads:int,d_embed:int,d_cross:int,in_proj_bias=True,out_proj_bias=True):
        super().__init__()
        #スライドのWq,Wk,Wvに対応
        self.q_proj=nn.Linear(d_embed,d_embed,bias=in_proj_bias)  #線形層は入力特徴量に対してy=xA^T+bを適用する b=バイアス項
        self.k_proj=nn.Linear(d_cross,d_embed,bias=in_proj_bias)  #重みAとバイアスbは訓練時にデータから学習されるモデルのパラメータ、モデルはタスクに最適な特徴表現を見つけることができる
        self.v_proj=nn.Linear(d_cross,d_embed,bias=in_proj_bias)
        self.out_proj=nn.Linear(d_embed,d_embed,bias=in_proj_bias)
        self.n_heads=n_heads
        self.d_head=d_embed//n_heads  #マルチヘッドアテンションの各ヘッドに割り当てられる特徴量次元数を、モデルの入力に対する埋め込みの次元数//ヘッドの数で計算

    def forward(self,x,y):
        #x:(latent):(Batch_Size,Seq_Len_Q,Dim_Q)
        #y:(context):(Batch_Size,Seq_Len_KV,Dim_KV)=(Batch_Size,77,768)
        #xはクエリ、yはキーとバリュー

        input_shape=x.shape
        batch_size,sequence_length,d_embed=input_shape

        intermim_shape=(batch_size,-1,self.n_heads,self.d_head) #-1は指定された他の次元サイズに基づいて自動的にこの次元のサイズを計算
        
        #クエリにWq,Wk,Wvをかける
        q=self.q_proj(x)
        k=self.k_proj(y)
        v=self.v_proj(x)

        q=q.view(intermim_shape).tranpose(1,2)
        k=k.view(intermim_shape).tranpose(1,2)
        v=v.view(intermim_shape).tranpose(1,2)
        
        #attentionの式の一部
        weight=q@k.transpose(-1,-2)
        
        weight/=math.sqrt(self.d_head)

        weight=F.softmax(weight,dim=-1)

        output=weight@v

        output=output.transpose(1,2).contiguous()  #transposeは各ヘッドの結果を横に並べて元のシーケンス長に対応させるために必要、contiguous()はテンソルがメモリ上で連続的に配置されていることを保証

        output=output.view(input_shape)   #入力の形状と同じにする  viewはテンソルの形状を変更

        output=self.out_proj(output)  #バイアスを加える

        return output
