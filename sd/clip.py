import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention

class CLIPEmbedding(nn.Module):
    def __init__(self, n_vocab: int, n_embd: int, n_token: int):
        #入力はトークン化されたもの
        #n_vocabはモデルが扱う単語の数、n_embdは埋め込む次元、n_tokenは一度に処理できる最大のトークンの数

        super().__init__()
        #nn.Moduleのコンストラクタを呼び出す

        self.token_embedding = nn.Embedding(n_vocab, n_embd)
        #nn.Embedding(num_embeddings, embedding_dim)
        #埋め込み次元数(n_embd)が大きいほどより豊かな表現が可能になるがパラメータの数も増える

        self.position_embedding = nn.Parameter(torch.zeros((n_token, n_embd)))
        #nn.Parameterはモジュール内で学習可能なテンソルを表すために使用
        #全ての要素がゼロで形状が(n_token, n_embd)のテンソルを作成
        #モデルによって学習済みのパラメータで、モデルをロードする際にこれらのパラメータを読み込む(動画2:02:10)
        #(model_converter.pyの886行目で読み込んでいる)
        #埋め込み表現は単語の意味や特徴を表現し、位置エンコーディングはトークンの位置情報を表現する
         
    def forward(self, tokens):
        x = self.token_embedding(tokens)
        x += self.position_embedding
        #各位置のトークンに対して位置エンコーディングを追加
        #トークンの意味的な情報と位置情報が組み合わさった埋め込み表現が得られる
       
        return x
    
class CLIPLayer(nn.Module):
    def __init__(self, n_head: int, n_embd: int):
        super().__init__()

        self.layernorm_1 = nn.LayerNorm(n_embd)
        self.attention = SelfAttention(n_head, n_embd)
        self.layernorm_2 = nn.LayerNorm(n_embd)
        self.linear_1 = nn.Linear(n_embd, 4 * n_embd)
        #引数は、nn.linear(in_features, out_features, bias=True)
        #入力の次元数はn_embdであり、出力の次元数は4 * n_embd
       

        self.linear_2 = nn.Linear(4 * n_embd, n_embd)
        #入力の次元数は4 * n_embdであり、出力の次元数は元のn_embdとなり、出力テンソルの形状は元に戻っている
        #次元変換は、一般的にモデルの表現能力を高めるために使用されるらしい

    def forward(self, x:torch.Tensor) ->torch.Tensor:
        residue = x
        #1回目の残差接続を行うための準備

        ### SELF ATTENTION ###

        x = self.layernorm_1(x)
        #xに正規化を行う

        x = self.attention(x, causal_mask=True)
        #causal_mask=Trueは各トークンが次のトークンを見ることができないことを意味
        #実際のテキストモデルでは、単語が前に来る単語だけを見ることが望ましいらしい

        x += residue
        #アテンション層の出力とresidue(入力）を足し合わせることで残差接続を行う

        ### FEEDFORWARD LAYER ###(全結合層)

        residue = x
        #2回目の残差接続を行うための準備
        
        x = self.layernorm_2(x) 
        x = self.linear_1(x)
        #xに線形変換を行う

        x = x * torch.sigmoid(1.702 * x) 
        #なぜかsigmoidをかけると良い結果になるらしい（理由はわかっていないらしい）

        x = self.linear_2(x)
        x += residue
        #2回目の残差接続を行う

        return x

class CLIP(nn.Module):

    def __init__(self):
        super().__init__()
        self.embedding = CLIPEmbedding(49408, 768, 77)
        #n_vocab=49408,n_emd=768(埋め込まれた後の次元),最大のトークンの長さが77
        #tokenizerは49408個のトークンを含むトークン辞書を持っている

        self.layers = nn.ModuleList([
            CLIPLayer(12, 768) for i in range(12)
            #multi-head attention の数が12,埋め込みの次元が768のlayerを12個作成
            #nn.ModuleListはモジュールをリスト形式で保持する
        ])

        self.layernorm = nn.LayerNorm(768)
        #nn.LayerNormでレイヤーを正規化する
        #768は正規化する特徴量の次元数
    
    def forward(self,tokens:torch.LongTensor) ->(torch.FloatTensor):
        #入力は整数値のテンソル、出力では浮動小数点のテンソル

        #(Batch_Size, Seq_Len) -> (Batch_Size, Seq_Len, Dim)
        tokens=tokens.type(torch.long)
        #tokensを整数に変換
        #self.embeddingの入力は整数にする必要がある

        state = self.embedding(tokens)
        #入力されたトークンに対して埋め込みを行う

        for layer in self.layers:
            state = layer(state)
        #stateはself.layersに含まれる複数のCLIPLayerを順番に通過する
        #CLIPLayerでは、SelfAttentionとFeedforward Layerが交互に適用されている
        #SelfAttentionではトークン間の関連性を捉える
        #for文を回すことでより正確にプロンプトを理解できるようになる
            
        #このときにstateは浮動小数点になると思われる

        output = self.layernorm(state)
        #正規化する

        return output
        #最終的な出力
