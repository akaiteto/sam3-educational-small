"""
model.py : SAM 3 の画像用の検出器（Sam3Image）を、CPUで動く小さなサイズで再現したもの

部品の対応（左がこのファイル、右が公式リポジトリ facebookresearch/sam3。*はファイル名を未確認の推定）
  ImageEncoder     … *sam3/model/vitdet.py の ViT ＋ Sam3DualViTDetNeck（ネック）
  TextEncoder      … sam3/model/text_encoder_ve.py（パーセプション・エンコーダの言語側）
  ExemplarEncoder  … *sam3/model/geometry_encoders.py（枠・見本のプロンプトの符号化）
  FusionEncoder    … sam3/model/encoder.py（画像特徴とプロンプトの融合）
  Decoder          … sam3/model/decoder.py（オブジェクトクエリ＋プレゼンストークン）
  MaskHead         … *sam3/model/maskformer_segmentation.py（ピクセルデコーダ＋内積でマスク、意味的ヘッド）
  MiniSam3         … sam3/model/sam3_image.py の Sam3Image

論文（arXiv:2511.16719 付録C）に合わせて入れたもの（Config のフラグで ON/OFF できる）
  use_box_rpb        … デコーダの画像へのクロスアテンションに、枠とマス目の相対位置のバイアスを足す
  look_forward_twice … 層ごとの枠の修正で、1つ前の層の枠にも勾配を流す
  num_hybrid         … 学習時だけ使う「1対多」用の補助クエリ（ハイブリッドマッチング）
  exemplar_layers    … 見本トークンを小さなTransformerに通す
  意味的ヘッド       … 各画素がプロンプトに当てはまるかを出す（常にON）

省略しているもの：動画用の追跡器、クリック操作（PVS）、DAC-DETR、曖昧さ対策のモジュール。
"""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

# demo.py から True にすると、各工程でテンソルの形を表示する
VERBOSE = False


def show(msg, t=None):
    if VERBOSE:
        shape = f"   形 {tuple(t.shape)}" if t is not None else ""
        print(f"  {msg}{shape}")


@dataclass
class Config:
    img_size: int = 64        # SAM 3: 1008
    patch: int = 8            # SAM 3: 14   -> マス目 8x8   （SAM 3: 72x72）
    dim: int = 64             # SAM 3: 画像エンコーダ1024、検出器256
    heads: int = 4            # SAM 3: 画像エンコーダ16
    enc_depth: int = 4        # SAM 3: 32層
    window: int = 4           # SAM 3: 24  -> 窓は 2x2=4個 （SAM 3: 3x3=9個）
    global_blocks: tuple = (1, 3)  # SAM 3: (7, 15, 23, 31)  全体アテンションの層
    pretrain_grid: int = 4    # SAM 3: 24（336/14）事前学習時のマス目の並び
    vocab_size: int = 9       # data.py の VOCAB の数
    text_len: int = 4         # SAM 3: 32
    text_layers: int = 2      # SAM 3: 24
    exemplar_layers: int = 1  # 見本トークン用の小さなTransformer（0で無効）
    fusion_layers: int = 3    # SAM 3: 6
    dec_layers: int = 3       # SAM 3: 6
    num_queries: int = 10     # SAM 3: 200
    num_hybrid: int = 20      # 1対多の補助クエリ（学習時だけ使う。0で無効）
    hybrid_k: int = 3         # 補助クエリでは、正解1つを何個のクエリに割り当てるか
    use_box_rpb: bool = True
    look_forward_twice: bool = True
    aux_loss: bool = True     # 各層の出力にも損失をかける
    mask_res: int = 32        # マスクを出す解像度

    @property
    def grid(self):
        return self.img_size // self.patch


# ---------------------------------------------------------------------------
# 共通の部品
# ---------------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, i, h, o, n):
        super().__init__()
        dims = [i] + [h] * (n - 1) + [o]
        self.layers = nn.ModuleList(nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:]))

    def forward(self, x):
        for k, layer in enumerate(self.layers):
            x = layer(x)
            if k < len(self.layers) - 1:
                x = F.relu(x)
        return x


class Attention(nn.Module):
    """
    アテンション。Q は q_in から、K と V は k_in・v_in から作る。
    q_in と k_in が同じならセルフアテンション、違えばクロスアテンションになる。
    attn_bias は、内積に足すバイアス（boxRPB で使う）。
    last_attn に、最後に計算したアテンションの重み（softmax後）を保存する（可視化用）。
    """

    def __init__(self, d, heads):
        super().__init__()
        self.h = heads
        self.q = nn.Linear(d, d)
        self.k = nn.Linear(d, d)
        self.v = nn.Linear(d, d)
        self.o = nn.Linear(d, d)
        self.last_attn = None

    def forward(self, q_in, k_in, v_in, key_pad=None, attn_mask=None, attn_bias=None):
        B, Nq, d = q_in.shape
        Nk = k_in.shape[1]
        h, hd = self.h, d // self.h
        q = self.q(q_in).view(B, Nq, h, hd).transpose(1, 2)  # (B,h,Nq,hd)
        k = self.k(k_in).view(B, Nk, h, hd).transpose(1, 2)  # (B,h,Nk,hd)
        v = self.v(v_in).view(B, Nk, h, hd).transpose(1, 2)
        att = q @ k.transpose(-2, -1) / math.sqrt(hd)       # 内積 (B,h,Nq,Nk)
        if attn_bias is not None:                            # (B,h,Nq,Nk)
            att = att + attn_bias
        if attn_mask is not None:                            # (Nq,Nk) True=見ない
            att = att.masked_fill(attn_mask, float("-inf"))
        if key_pad is not None:                              # (B,Nk) True=パディング
            att = att.masked_fill(key_pad[:, None, None, :], float("-inf"))
        att = att.softmax(-1)
        self.last_attn = att.detach()
        out = (att @ v).transpose(1, 2).reshape(B, Nq, d)
        return self.o(out)


def masked_mean(x, pad):
    """パディングでないトークンだけを平均する。x:(B,L,d) pad:(B,L) -> (B,d)"""
    keep = (~pad).float().unsqueeze(-1)
    return (x * keep).sum(1) / keep.sum(1).clamp(min=1.0)


def sine_pos_2d(h, w, d, temp=10000.0):
    """マス目の位置 (x, y) を sin・cos の波で d 次元にする（DETR系の位置エンコーディング）。"""
    n = d // 4
    dim_t = temp ** (torch.arange(n).float() / n)
    ys, xs = torch.meshgrid(torch.arange(h).float(), torch.arange(w).float(), indexing="ij")
    xs = (xs.flatten() + 0.5) / w * 2 * math.pi
    ys = (ys.flatten() + 0.5) / h * 2 * math.pi
    px = xs[:, None] / dim_t
    py = ys[:, None] / dim_t
    return torch.cat([px.sin(), px.cos(), py.sin(), py.cos()], dim=-1)  # (h*w, d)


def sine_embed_boxes(boxes, d, temp=10000.0):
    """枠 (cx, cy, w, h) の4つの数を、それぞれ sin・cos の波で d/4 次元にして連結する。"""
    n = d // 8
    dim_t = temp ** (torch.arange(n, device=boxes.device).float() / n)
    x = boxes[..., None] * 2 * math.pi / dim_t             # (...,4,n)
    emb = torch.cat([x.sin(), x.cos()], dim=-1)            # (...,4,d/4)
    return emb.flatten(-2)                                 # (...,d)


def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(eps, 1 - eps)
    return torch.log(x / (1 - x))


# ---------------------------------------------------------------------------
# 1. 画像エンコーダ（ViT ＋ 窓アテンション/全体アテンション ＋ RoPE ＋ ネック）
# ---------------------------------------------------------------------------
def build_rope(grid, head_dim, theta=100.0):
    """2次元RoPE用の回転角を作る。ヘッド次元の前半を x 座標、後半を y 座標の回転に使う。"""
    d = head_dim // 2
    freqs = 1.0 / (theta ** (torch.arange(0, d, 2).float() / d))  # (d/2)
    ys, xs = torch.meshgrid(torch.arange(grid), torch.arange(grid), indexing="ij")
    ax = xs.flatten()[:, None].float() * freqs
    ay = ys.flatten()[:, None].float() * freqs
    ang = torch.cat([ax, ay], dim=-1)                              # (T, head_dim/2)
    return ang.cos(), ang.sin()


def apply_rope(x, cos, sin):
    """x の次元を2つずつ組にして、マス目の位置に応じた角度だけ回転させる。x:(B,h,T,hd)"""
    x1, x2 = x[..., 0::2], x[..., 1::2]
    r1 = x1 * cos - x2 * sin
    r2 = x1 * sin + x2 * cos
    return torch.stack([r1, r2], dim=-1).flatten(-2)


def window_partition(t, grid, w):
    """(B,h,T,D) のマス目を w x w の窓に分けて (B*窓数, h, w*w, D) にする。"""
    B, H, T, D = t.shape
    n = grid // w
    t = t.reshape(B, H, n, w, n, w, D).permute(0, 2, 4, 1, 3, 5, 6)
    return t.reshape(B * n * n, H, w * w, D)


def window_merge(t, B, grid, w):
    """window_partition の逆。(B*窓数, h, w*w, D) -> (B,h,T,D)"""
    n = grid // w
    H, D = t.shape[1], t.shape[-1]
    t = t.reshape(B, n, n, H, w, w, D).permute(0, 3, 1, 4, 2, 5, 6)
    return t.reshape(B, H, grid * grid, D)


class ViTBlock(nn.Module):
    def __init__(self, dim, heads, grid, window, use_global):
        super().__init__()
        self.heads, self.grid, self.window, self.use_global = heads, grid, window, use_global
        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))

    def forward(self, x, rope):
        B, T, C = x.shape
        H, hd = self.heads, C // self.heads
        h = self.norm1(x)
        q, k, v = self.qkv(h).reshape(B, T, 3, H, hd).permute(2, 0, 3, 1, 4).unbind(0)
        # RoPE：Q と K だけを、マス目の縦横の位置に応じて回転させる
        q = apply_rope(q, *rope)
        k = apply_rope(k, *rope)
        if self.use_global:
            # 全体アテンション：各マス目が、全マス目と内積を取る
            att = (q @ k.transpose(-2, -1) / math.sqrt(hd)).softmax(-1)
            out = att @ v
        else:
            # 窓アテンション：各マス目が、同じ窓の中のマス目とだけ内積を取る
            qw, kw, vw = (window_partition(t, self.grid, self.window) for t in (q, k, v))
            att = (qw @ kw.transpose(-2, -1) / math.sqrt(hd)).softmax(-1)
            out = window_merge(att @ vw, B, self.grid, self.window)
        out = out.transpose(1, 2).reshape(B, T, C)
        x = x + self.proj(out)              # 残差
        x = x + self.mlp(self.norm2(x))     # MLP ＋ 残差
        return x


class Neck(nn.Module):
    """
    ViTDet方式のネック（論文では SimpleFPN）。1種類の特徴マップ (8x8) から、
    マスクヘッド用の細かい特徴マップ high (32x32) と、検出器用の特徴マップ mid (8x8) を作る。
    （論文：画像エンコーダは1種類の解像度だけを出し、複数解像度はセグメンテーションのヘッドだけが使う）
    """

    def __init__(self, cfg):
        super().__init__()
        D = d = cfg.dim
        self.to_high = nn.Sequential(
            nn.ConvTranspose2d(D, D // 2, 2, 2), nn.GELU(),
            nn.ConvTranspose2d(D // 2, D // 4, 2, 2),
            nn.Conv2d(D // 4, d, 1), nn.Conv2d(d, d, 3, padding=1),
        )
        self.to_mid = nn.Sequential(nn.Conv2d(D, d, 1), nn.Conv2d(d, d, 3, padding=1))

    def forward(self, fmap):
        return {"high": self.to_high(fmap), "mid": self.to_mid(fmap)}


class ImageEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        G, D = cfg.grid, cfg.dim
        self.patch_embed = nn.Conv2d(3, D, kernel_size=cfg.patch, stride=cfg.patch)
        # 位置埋め込みは、事前学習時のマス目の並び (4x4) の大きさで持つ
        self.pos_embed = nn.Parameter(torch.zeros(1, D, cfg.pretrain_grid, cfg.pretrain_grid))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.blocks = nn.ModuleList(
            ViTBlock(D, cfg.heads, G, cfg.window, use_global=(i in cfg.global_blocks))
            for i in range(cfg.enc_depth)
        )
        cos, sin = build_rope(G, D // cfg.heads)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.neck = Neck(cfg)

    def encode_tokens(self, img):
        """ネックの前までを計算し、画像トークン (B, T, D) を返す（段階1の事前学習でも使う）。"""
        G, pg = self.cfg.grid, self.cfg.pretrain_grid
        x = self.patch_embed(img)
        show("[1-1] パッチ埋め込みは、画像を8x8画素ずつ切り、各マス目を64次元にする", x)
        pos = self.pos_embed.repeat(1, 1, G // pg, G // pg)
        show("[1-2] 4x4の位置埋め込みを縦横2回ずつ並べて8x8に広げる（タイル）", pos)
        x = x + pos
        x = x.flatten(2).transpose(1, 2)
        show("[1-3] マス目をトークンの列に並べ直す（64トークン x 64次元）", x)
        for i, blk in enumerate(self.blocks):
            x = blk(x, (self.rope_cos, self.rope_sin))
            kind = "全体アテンション" if blk.use_global else "窓アテンション(4窓 x 16トークン)"
            show(f"[1-4] ブロック{i + 1}：{kind} ＋ MLP", x)
        return x

    def forward(self, img):
        x = self.encode_tokens(img)
        B, T, D = x.shape
        G = self.cfg.grid
        fmap = x.transpose(1, 2).reshape(B, D, G, G)
        feats = self.neck(fmap)
        show("[1-5] ネックは、細かい特徴マップ high を作る", feats["high"])
        show("[1-5] ネックは、検出器用の特徴マップ mid を作る", feats["mid"])
        return feats


# ---------------------------------------------------------------------------
# 2. テキストエンコーダ（因果マスク付きのTransformer）
# ---------------------------------------------------------------------------
class TextLayer(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.n1 = nn.LayerNorm(d)
        self.attn = Attention(d, heads)
        self.n2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x, causal, pad):
        h = self.n1(x)
        x = x + self.attn(h, h, h, key_pad=pad, attn_mask=causal)
        return x + self.mlp(self.n2(x))


class TextEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg.dim
        self.tok = nn.Embedding(cfg.vocab_size, d)
        self.pos = nn.Parameter(torch.zeros(1, cfg.text_len, d))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.layers = nn.ModuleList(TextLayer(d, cfg.heads) for _ in range(cfg.text_layers))
        self.norm = nn.LayerNorm(d)
        self.resizer = nn.Linear(d, d)  # 公式でも、言語側の出力を検出器の次元(256)に線形変換する
        causal = torch.triu(torch.ones(cfg.text_len, cfg.text_len, dtype=torch.bool), diagonal=1)
        self.register_buffer("causal", causal, persistent=False)

    def forward(self, tokens, pad):
        x = self.tok(tokens) + self.pos
        show("[2-1] テキストエンコーダは、単語IDを埋め込みに変え、位置埋め込みを足す", x)
        for layer in self.layers:
            x = layer(x, self.causal, pad)
        x = self.resizer(self.norm(x))
        show("[2-2] テキストエンコーダは、因果マスク付きセルフアテンションを通し、テキストトークンを出す", x)
        return x


# ---------------------------------------------------------------------------
# 3. 見本（exemplar）エンコーダ：枠で指定した見本をトークンにする
# ---------------------------------------------------------------------------
class ExemplarLayer(nn.Module):
    """論文：見本トークンは小さなTransformerで、セルフアテンションと画像埋め込みへのクロスアテンションをする。"""

    def __init__(self, d, heads):
        super().__init__()
        self.n1, self.n2, self.n3 = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.self_attn = Attention(d, heads)
        self.cross = Attention(d, heads)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, tok, feat, pos):
        h = self.n1(tok)
        tok = tok + self.self_attn(h, h, h)                  # 見本トークン同士
        h = self.n2(tok)
        tok = tok + self.cross(h, feat + pos, feat)          # Q=見本トークン, K・V=画像
        return tok + self.mlp(self.n3(tok))


class ExemplarEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d, G = cfg.dim, cfg.grid
        self.box_proj = nn.Linear(4, d)        # 枠の4つの数を直接 d 次元へ
        self.roi_proj = nn.Linear(d, d)        # 枠の中の画像特徴の平均（ROIプーリングの代わり）
        self.pos_proj = nn.Linear(d, d)        # 枠の位置エンコーディング
        self.label_embed = nn.Embedding(2, d)  # 0=負の見本、1=正の見本
        self.norm = nn.LayerNorm(d)
        self.layers = nn.ModuleList(ExemplarLayer(d, cfg.heads) for _ in range(cfg.exemplar_layers))
        ys, xs = torch.meshgrid(torch.arange(G).float(), torch.arange(G).float(), indexing="ij")
        self.register_buffer("cx", (xs.flatten() + 0.5) / G, persistent=False)
        self.register_buffer("cy", (ys.flatten() + 0.5) / G, persistent=False)
        self.register_buffer("img_pos", sine_pos_2d(G, G, d)[None], persistent=False)

    def forward(self, boxes, labels, fmap):
        d = fmap.shape[1]
        feat = fmap.flatten(2).transpose(1, 2)                                 # (B,T,d)
        cx, cy, w, h = boxes.unbind(-1)                                        # (B,K)
        inside = ((self.cx - cx[..., None]).abs() <= w[..., None] / 2) & \
                 ((self.cy - cy[..., None]).abs() <= h[..., None] / 2)         # (B,K,T)
        wts = inside.float() + 1e-6
        roi = (wts @ feat) / wts.sum(-1, keepdim=True)                         # (B,K,d)
        tok = self.box_proj(boxes) + self.roi_proj(roi) \
            + self.pos_proj(sine_embed_boxes(boxes, d)) + self.label_embed(labels)
        tok = self.norm(tok)
        for layer in self.layers:
            tok = layer(tok, feat, self.img_pos)
        show("[3] 見本エンコーダは、枠の座標・枠内の特徴・位置・正負ラベルを足し、小さなTransformerに通す", tok)
        return tok


# ---------------------------------------------------------------------------
# 4. 融合エンコーダ：画像トークンがプロンプトトークンをクロスアテンションする
# ---------------------------------------------------------------------------
class FusionLayer(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.n1, self.n2, self.n3 = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.self_attn = Attention(d, heads)
        self.cross = Attention(d, heads)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x, pos, prompt, prompt_pad):
        h = self.n1(x)
        x = x + self.self_attn(h + pos, h + pos, h)                       # 画像同士のセルフアテンション
        h = self.n2(x)
        x = x + self.cross(h + pos, prompt, prompt, key_pad=prompt_pad)   # Q=画像, K・V=プロンプト
        return x + self.mlp(self.n3(x))


class FusionEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg.dim
        self.text_pool_proj = nn.Linear(d, d)
        self.layers = nn.ModuleList(FusionLayer(d, cfg.heads) for _ in range(cfg.fusion_layers))
        self.register_buffer("pos", sine_pos_2d(cfg.grid, cfg.grid, d)[None], persistent=False)

    def forward(self, fmap, text, text_pad, prompt, prompt_pad):
        x = fmap.flatten(2).transpose(1, 2)                                    # (B,T,d)
        pooled = masked_mean(text, text_pad)
        x = x + self.text_pool_proj(pooled)[:, None]
        show("[4-1] 融合エンコーダは、テキストトークンの平均を全画像トークンに加算する（早期融合）", x)
        for layer in self.layers:
            x = layer(x, self.pos, prompt, prompt_pad)
        show("[4-2] 融合エンコーダは、セルフ → クロス(Q=画像,K・V=プロンプト) → MLP を繰り返す", x)
        return x


# ---------------------------------------------------------------------------
# 5. デコーダ：オブジェクトクエリ ＋ プレゼンストークン ＋ 枠の反復修正 ＋ boxRPB
# ---------------------------------------------------------------------------
class BoxRPB(nn.Module):
    """
    boxRPB（枠とマス目の相対位置のバイアス）。
    各クエリの枠の左右の辺と各マス目の x 座標の距離、上下の辺と y 座標の距離を、
    符号付きの log2 で縮めてから MLP に通し、ヘッドごとのバイアスにする。
    x のバイアスと y のバイアスを足して、(クエリ, マス目) ごとのバイアスを作る。
    これを内積に足すと、各クエリは自分の枠の中や近くのマス目を強く見るようになる。
    """

    def __init__(self, cfg):
        super().__init__()
        self.G = cfg.grid
        self.mlp_x = MLP(2, 32, cfg.heads, 2)
        self.mlp_y = MLP(2, 32, cfg.heads, 2)
        self.register_buffer("centers", (torch.arange(cfg.grid).float() + 0.5) / cfg.grid, persistent=False)

    def forward(self, boxes):
        """boxes: (B,N,4) cx,cy,w,h  ->  (B, ヘッド数, 1+N, マス目数)  先頭の1行はプレゼンストークン用で0"""
        B, N, _ = boxes.shape
        G = self.G
        cx, cy, bw, bh = boxes.unbind(-1)
        c = self.centers
        dx = torch.stack([c - (cx - bw / 2)[..., None], c - (cx + bw / 2)[..., None]], dim=-1) * G  # (B,N,G,2)
        dy = torch.stack([c - (cy - bh / 2)[..., None], c - (cy + bh / 2)[..., None]], dim=-1) * G
        dx = torch.sign(dx) * torch.log2(dx.abs() + 1)
        dy = torch.sign(dy) * torch.log2(dy.abs() + 1)
        bx = self.mlp_x(dx)                                     # (B,N,G,ヘッド)
        by = self.mlp_y(dy)
        bias = by[:, :, :, None, :] + bx[:, :, None, :, :]      # (B,N,Gy,Gx,ヘッド)
        bias = bias.reshape(B, N, G * G, -1).permute(0, 3, 1, 2)  # (B,ヘッド,N,T)
        zero = bias.new_zeros(B, bias.shape[1], 1, G * G)
        return torch.cat([zero, bias], dim=2)


class DecoderLayer(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.n1, self.n2, self.n3, self.n4 = (nn.LayerNorm(d) for _ in range(4))
        self.self_attn = Attention(d, heads)
        self.text_cross = Attention(d, heads)
        self.img_cross = Attention(d, heads)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, tgt, tgt_pos, memory, mem_pos, prompt, prompt_pad, self_mask=None, img_bias=None):
        h = self.n1(tgt)
        tgt = tgt + self.self_attn(h + tgt_pos, h + tgt_pos, h, attn_mask=self_mask)   # クエリ同士
        h = self.n2(tgt)
        tgt = tgt + self.text_cross(h + tgt_pos, prompt, prompt, key_pad=prompt_pad)  # Q=クエリ, K・V=プロンプト
        h = self.n3(tgt)
        tgt = tgt + self.img_cross(h + tgt_pos, memory + mem_pos, memory, attn_bias=img_bias)  # Q=クエリ, K・V=画像
        return tgt + self.mlp(self.n4(tgt))


class Decoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg.dim
        self.N, self.Nh = cfg.num_queries, cfg.num_hybrid
        self.lft = cfg.look_forward_twice
        self.query_embed = nn.Embedding(self.N + self.Nh, d)   # オブジェクトクエリ（中身）。後ろの Nh 個は補助クエリ
        self.ref_boxes = nn.Embedding(self.N + self.Nh, 4)     # 各クエリの初期の枠（シグモイド前）
        with torch.no_grad():
            self.ref_boxes.weight[:, :2].uniform_(-1.5, 1.5)
            self.ref_boxes.weight[:, 2:].fill_(-1.5)
        self.presence_token = nn.Embedding(1, d)               # プレゼンストークン
        self.query_pos = MLP(d, d, d, 2)
        self.layers = nn.ModuleList(DecoderLayer(d, cfg.heads) for _ in range(cfg.dec_layers))
        self.norm = nn.LayerNorm(d)
        self.box_head = MLP(d, d, 4, 3)
        nn.init.zeros_(self.box_head.layers[-1].weight)
        nn.init.zeros_(self.box_head.layers[-1].bias)
        self.box_rpb = BoxRPB(cfg) if cfg.use_box_rpb else None

    def forward(self, memory, mem_pos, prompt, prompt_pad):
        B, _, d = memory.shape
        nq = self.N + (self.Nh if self.training else 0)   # 補助クエリは学習時だけ使う
        tgt = torch.cat([self.presence_token.weight[None].expand(B, -1, -1),
                         self.query_embed.weight[:nq][None].expand(B, -1, -1)], dim=1)
        show(f"[5-1] デコーダは、プレゼンストークン1個とオブジェクトクエリ{nq}個を連結する", tgt)

        # 補助クエリと通常のクエリ（＋プレゼンストークン）は、互いにセルフアテンションで見えないようにする
        self_mask = None
        if nq > self.N:
            L = 1 + nq
            self_mask = torch.zeros(L, L, dtype=torch.bool, device=memory.device)
            self_mask[:1 + self.N, 1 + self.N:] = True
            self_mask[1 + self.N:, :1 + self.N] = True

        ref = self.ref_boxes.weight[:nq].sigmoid()[None].expand(B, -1, -1)
        prev = ref                     # look-forward-twice 用：勾配を切っていない1つ前の枠
        zeros = memory.new_zeros(B, 1, d)
        hs_list, box_list = [], []
        for i, layer in enumerate(self.layers):
            qpos = self.query_pos(sine_embed_boxes(ref, d))       # 今の枠を位置エンコーディングにする
            bias = self.box_rpb(ref) if self.box_rpb is not None else None
            tgt = layer(tgt, torch.cat([zeros, qpos], dim=1), memory, mem_pos, prompt, prompt_pad,
                        self_mask=self_mask, img_bias=bias)
            hs = self.norm(tgt)
            delta = self.box_head(hs[:, 1:])                      # 枠の修正量
            new_ref = (inverse_sigmoid(ref) + delta).sigmoid()    # 次の層に渡す枠
            # 損失に使う枠：look-forward-twice なら、勾配を切っていない1つ前の枠に修正量を足す
            out_box = (inverse_sigmoid(prev) + delta).sigmoid() if self.lft else new_ref
            hs_list.append(hs)
            box_list.append(out_box)
            prev = new_ref
            ref = new_ref.detach()
            rpb = "（boxRPBあり）" if bias is not None else ""
            show(f"[5-2] デコーダ層{i + 1}：セルフ → テキストへのクロス → 画像へのクロス{rpb} → MLP、枠を修正", out_box)
        return hs_list, box_list


# ---------------------------------------------------------------------------
# 6. マスクヘッド：ピクセル埋め込みとマスク埋め込みの内積 ＋ 意味的ヘッド
# ---------------------------------------------------------------------------
class MaskHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg.dim
        self.G, self.res = cfg.grid, cfg.mask_res
        self.pixel_decoder = nn.Sequential(
            nn.Conv2d(d, d, 3, padding=1), nn.GroupNorm(8, d), nn.GELU(), nn.Conv2d(d, d, 3, padding=1))
        self.mask_embed = MLP(d, d, d, 3)
        self.semantic_head = nn.Conv2d(d, 1, 1)

    def forward(self, memory, high, queries):
        B, T, d = memory.shape
        enc = memory.transpose(1, 2).reshape(B, d, self.G, self.G)
        enc_up = F.interpolate(enc, size=(self.res, self.res), mode="bilinear", align_corners=False)
        pixel = self.pixel_decoder(high + enc_up)
        show("[6-1] ピクセルデコーダは、融合後の画像トークンを拡大し high に加算して、ピクセル埋め込みを作る", pixel)
        semantic = self.semantic_head(pixel)
        show("[6-2] 意味的ヘッドは、各画素がプロンプトに当てはまるかを出す（物体を区別しない）", semantic)
        me = self.mask_embed(queries)
        show("[6-3] MLPは、各クエリの出力からマスク埋め込みを作る", me)
        logits = torch.einsum("bnd,bdhw->bnhw", me, pixel)
        show("[6-4] マスク埋め込みと各画素のピクセル埋め込みの内積が、クエリごとのマスクになる", logits)
        return logits, semantic


# ---------------------------------------------------------------------------
# 全体
# ---------------------------------------------------------------------------
class MiniSam3(nn.Module):
    def __init__(self, cfg=None):
        super().__init__()
        cfg = cfg or Config()
        self.cfg = cfg
        d = cfg.dim
        self.image_encoder = ImageEncoder(cfg)
        self.text_encoder = TextEncoder(cfg)
        self.exemplar_encoder = ExemplarEncoder(cfg)
        self.fusion = FusionEncoder(cfg)
        self.decoder = Decoder(cfg)
        self.mask_head = MaskHead(cfg)
        self.score_q = nn.Linear(d, d)
        self.score_p = nn.Linear(d, d)
        self.presence_head = MLP(d, d, 1, 2)
        # 段階1（画像と文の対照学習）用の射影と温度。検出には使わない
        self.clip_img = nn.Linear(d, d)
        self.clip_txt = nn.Linear(d, d)
        self.clip_scale = nn.Parameter(torch.tensor(math.log(10.0)))
        self.clip_bias = nn.Parameter(torch.tensor(-5.0))

    def contrastive_logits(self, images, tokens, text_pad):
        """
        段階1：画像エンコーダとテキストエンコーダだけを使い、画像と文の一致度 (画像数 x 文数) を出す。
        パーセプション・エンコーダの事前学習（画像と文の組の対照学習）を小さく再現したもの。
        """
        img = self.image_encoder.encode_tokens(images).mean(1)
        txt = masked_mean(self.text_encoder(tokens, text_pad), text_pad)
        img = F.normalize(self.clip_img(img), dim=-1)
        txt = F.normalize(self.clip_txt(txt), dim=-1)
        return img @ txt.t() * self.clip_scale.exp() + self.clip_bias

    def forward(self, images, tokens, text_pad, ex_boxes=None, ex_valid=None, ex_labels=None):
        N, d = self.cfg.num_queries, self.cfg.dim
        show("[0] 入力画像", images)
        feats = self.image_encoder(images)
        text = self.text_encoder(tokens, text_pad)

        # プロンプトトークン ＝ テキストトークン と 見本トークン を列方向に連結したもの
        if ex_boxes is not None:
            if ex_labels is None:
                ex_labels = torch.ones(ex_boxes.shape[:2], dtype=torch.long, device=ex_boxes.device)
            ex = self.exemplar_encoder(ex_boxes, ex_labels, feats["mid"])
            prompt = torch.cat([text, ex], dim=1)
            prompt_pad = torch.cat([text_pad, ~ex_valid], dim=1)
        else:
            prompt, prompt_pad = text, text_pad
        show("[3'] テキストトークンと見本トークンを連結して、プロンプトトークンにする", prompt)

        memory = self.fusion(feats["mid"], text, text_pad, prompt, prompt_pad)
        hs_list, box_list = self.decoder(memory, self.fusion.pos, prompt, prompt_pad)

        # 各クエリの「プロンプトに合う度合い」＝ クエリ出力 と プロンプトの平均 の内積
        p = self.score_p(masked_mean(prompt, prompt_pad))[:, None]

        def score(h):
            return (self.score_q(h) * p).sum(-1) / math.sqrt(d)

        logits_layers = [score(h[:, 1:1 + N]) for h in hs_list]
        final = hs_list[-1]
        presence_logit = self.presence_head(final[:, 0]).squeeze(-1)
        show("[5-3] 各クエリの一致スコア（内積）", logits_layers[-1])
        show("[5-4] プレゼンスヘッドは、プレゼンストークンの出力から「画像内に存在するか」を出す", presence_logit)

        masks, semantic = self.mask_head(memory, feats["high"], final[:, 1:1 + N])
        scores = logits_layers[-1].sigmoid() * presence_logit.sigmoid()[:, None]
        show("[7] 最終スコア ＝ p(存在する) x p(このクエリが一致 | 存在する)", scores)

        hybrid = None
        if box_list[-1].shape[1] > N:
            hybrid = {"logits": score(final[:, 1 + N:]), "boxes": box_list[-1][:, N:]}
        return {
            "pred_logits": logits_layers[-1],   # (B,N)
            "presence_logit": presence_logit,   # (B,)
            "pred_boxes": box_list[-1][:, :N],  # (B,N,4) cx,cy,w,h (0〜1)
            "pred_masks": masks,                # (B,N,32,32) ロジット
            "semantic": semantic,               # (B,1,32,32) ロジット
            "scores": scores,                   # (B,N)
            "aux": [{"logits": logits_layers[i], "boxes": box_list[i][:, :N]} for i in range(len(box_list) - 1)],
            "hybrid": hybrid,                   # 学習時だけ
            "boxes_per_layer": [b[:, :N] for b in box_list],
        }
