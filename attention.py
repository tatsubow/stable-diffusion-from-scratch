import torch
from torch import nn
from torch.nn import functional as F

#selfattentionの説明がスライドであった
#最初はトークンの列、サイズDの埋め込みを持つ、それをクエリ、キー、バリューに変換
#セルフアテンションではクエリ、キー、バリューは全て同じ行列

class SelfAttention(nn.Module):

    #引数はn_heads=ヘッドの数、d_embed=単語の意味、それぞれのピクセルのチャンネル数、それぞれのピクセルはそのピクセルの情報をとらえた多くのチャンネルによって表現されている
    def __init__(self,n_heads:int,d_embed:int , in_proj_bias=True,out_proj_bias=True):
        super().__init__()

        #WQ,WK,WVを3つの異なる行列として表現するのではなく一つの大きなlinear layerとする
        #これはattentionを適用する前の入力のprojection
        #スライドのWO行列の形状はdmodel*dmoedl
        self.in_proj=nn.Linear(d_embed*3,3*d_embed,bias=in_proj_bias)
        self.out_proj=nn.Linear(d_embed,d_embed, bias=out_proj_bias)
        self.n_heads=n_heads  #ヘッドの数を保存
        self.d_head=d_embed//n_heads  #マルチヘッドはそれぞれのトークンの埋め込みの一部を見るので割り算　動画の1:36:17
    
    #マスク　あるトークンより前のみを参照するようにする
    def forward(self,x:torch.Tensor,causal_mask=False):
        input_shape=x.shape  #入力の形状を抽出
        batch_size,sequence_length,d_embed=input_shape
        
        #入力の形状から別の形状にする
        intermim_shape=(batch_size,sequence_length,self.n_heads,self.d_head)
        
        #(Batch_size,seq_len,dim) -> (Batch_size, seq_len, dim*3)  -> 3tensprs of shape(Batch_size,Seq_len, Dim)
        #self.in_projがWQ,WK,WVにあたる　
        q,k,v=self.in_proj(x).chunk(3,dim=-1)  
        
        #chunkで最後の次元(最後の次元って何か)に沿って3分割する
        
        #(Batch_Size,Seq_len,Dim)  -> (Batch_size, Seq_len ,H,  Dim/H)  -> (Batch_Size,H,Seq_len,Dim/H)
        q=q.view(intermim_shape).transpose(1,2)  #2次元目と3次元目の位置を入れ替える
        k=k.view(intermim_shape).transpose(1,2)
        v=v.view(intermim_shape).transpose(1,2)

        #(Batch_Size,H,Seq_Len,Seq_Len)
        #Attention(Q,K,V)の式のQKT

        #なぜこれが重みなのか
        weight=q@k.transpose(-1,-2)#テンソルの最後の次元と最後から2番目の次元を入れ替える
        
        #因果マスクを適用するかを指定 causal_mask=Trueならマスクが適用
        if causal_mask:
            #weightと同じ形状のテンソルがtorch.one_likeで作成されデータ型はbool　triuで上三角行列を生成　対角線より上のみ1にしそれ以外0にする
            #未来の時点の情報を考慮しないようにするマスクが作成される
            #なぜこれで未来を考慮しなくなるのか
            mask=torch.ones_like(weight,dtype=torch.bool).triu(1)
            
            weight.masked_fill(mask,-torch.inf)
        
        weight/=mask.sqrt(self.d_head) #Attentionの式の分数の部分

        weight=F.softmax(weight,dim=-1)
        
        #(Batch_Size,H,Seq_len,Seq_len)@(Batch_Size,H,Seq_len,Dim/H)->(Batch_Size,H,Seqlen,Dim/H)
        output=weight@v
        
        #(Batch_Size,H,Seqlen,Dim/H) ->(Batch_SIze,Sequencelen, H,Dim/H)
        output=output.tranpose(1,2)

#なぜ出力を入力と揃えるのか
        
        output=output.reshape(input_shape) #入力の時のテンソルの形状にする


        output=self.output_proj(output)
        
        #(Batch_Size,Seqlen,Dim)
        return output
