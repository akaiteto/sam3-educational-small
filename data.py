"""
data.py : 学習・確認用の合成データを作る

このファイルは、64x64画素の画像に「色つきの図形」を1〜3個描き、
「red circle」「circle」「blue」のような短い名詞句（コンセプト）と、
そのコンセプトに当てはまる全物体の「枠」と「マスク」を正解として返す。

SAM 3 のタスク（Promptable Concept Segmentation）と同じく、
1つのコンセプトに当てはまる物体を「全部」見つけることが目的になる。
コンセプトに当てはまる物体が1つもない画像も作る（プレゼンストークンの学習に使う）。
"""
import math
import random

import numpy as np
import torch

IMG = 64
COLORS = {"red": (220, 50, 50), "green": (40, 170, 70), "blue": (50, 90, 230)}
SHAPES = ["circle", "square", "triangle"]

# 語彙：単語1つが1トークンになる（本物のSAM 3はBPEで分割する）
VOCAB = ["<pad>", "<bos>", "<eos>"] + list(COLORS.keys()) + SHAPES
WORD2ID = {w: i for i, w in enumerate(VOCAB)}
TEXT_LEN = 4  # <bos> 単語 単語 <eos>  （本物のSAM 3は32トークン）

# 指定できるコンセプトの一覧
ALL_CONCEPTS = [f"{c} {s}" for c in COLORS for s in SHAPES] + SHAPES + list(COLORS.keys())


def tokenize(text):
    """トークナイザは、文字列を単語IDの列と、パディング位置（True=空き）に変える。"""
    words = text.split()
    ids = [WORD2ID["<bos>"]] + [WORD2ID[w] for w in words] + [WORD2ID["<eos>"]]
    ids = ids[:TEXT_LEN]
    pad = [False] * len(ids) + [True] * (TEXT_LEN - len(ids))
    ids = ids + [WORD2ID["<pad>"]] * (TEXT_LEN - len(ids))
    return ids, pad


def shape_mask(shape, cx, cy, r):
    """この関数は、図形1つ分の2値マスク（64x64のTrue/False）を作る。"""
    yy, xx = np.mgrid[0:IMG, 0:IMG]
    xx = xx + 0.5
    yy = yy + 0.5
    if shape == "circle":
        m = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
    elif shape == "square":
        m = (np.abs(xx - cx) <= r) & (np.abs(yy - cy) <= r)
    else:  # triangle（上向き）
        m = (yy >= cy - r) & (yy <= cy + r) & (np.abs(xx - cx) <= (yy - (cy - r)) / 2)
    return m


def random_scene(rng):
    """この関数は、図形を1〜3個、重ならないように置いた画像を作る。"""
    n = rng.randint(1, 3)
    objs = []
    tries = 0
    while len(objs) < n and tries < 200:
        tries += 1
        r = rng.uniform(6, 11)
        cx = rng.uniform(r + 1, IMG - r - 1)
        cy = rng.uniform(r + 1, IMG - r - 1)
        if all(math.hypot(cx - o["cx"], cy - o["cy"]) > (r + o["r"]) * 1.45 + 2 for o in objs):
            objs.append(dict(color=rng.choice(list(COLORS)), shape=rng.choice(SHAPES), cx=cx, cy=cy, r=r))

    np_rng = np.random.default_rng(rng.randrange(1 << 30))
    img = np.full((IMG, IMG, 3), 235.0) + np_rng.normal(0, 6, (IMG, IMG, 3))
    masks = []
    for o in objs:
        m = shape_mask(o["shape"], o["cx"], o["cy"], o["r"])
        img[m] = np.array(COLORS[o["color"]]) + np_rng.normal(0, 8, (int(m.sum()), 3))
        masks.append(m)
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img, objs, masks


def matches(concept, obj):
    """コンセプトの単語がすべて物体の属性（色・形）に含まれていれば、その物体は正解になる。"""
    return all(w in (obj["color"], obj["shape"]) for w in concept.split())


def mask_to_box(m):
    """この関数は、マスクから枠（cx, cy, w, h を 0〜1 に正規化）を作る。"""
    ys, xs = np.nonzero(m)
    x0, x1 = xs.min(), xs.max() + 1
    y0, y1 = ys.min(), ys.max() + 1
    return [(x0 + x1) / 2 / IMG, (y0 + y1) / 2 / IMG, (x1 - x0) / IMG, (y1 - y0) / IMG]


def targets_for(concept, objs, masks):
    """この関数は、コンセプトに当てはまる全物体の枠とマスクを正解として返す。"""
    idx = [i for i, o in enumerate(objs) if matches(concept, o)]
    boxes = torch.tensor([mask_to_box(masks[i]) for i in idx], dtype=torch.float32).reshape(-1, 4)
    if idx:
        tmasks = torch.from_numpy(np.stack([masks[i] for i in idx]).astype(np.float32))
    else:
        tmasks = torch.zeros(0, IMG, IMG)
    return {"boxes": boxes, "masks": tmasks}


def image_to_tensor(img):
    """この関数は、画像(64,64,3 の 0〜255)を、(3,64,64) の -0.5〜0.5 のテンソルに変える。"""
    return torch.from_numpy(img.astype(np.float32) / 255.0 - 0.5).permute(2, 0, 1)


def pick_concept(rng, objs):
    """7割は画像内にある物体からコンセプトを選び、3割はランダムに選ぶ（無い場合も作るため）。"""
    if rng.random() < 0.7:
        o = rng.choice(objs)
        k = rng.random()
        if k < 0.5:
            return f"{o['color']} {o['shape']}"
        return o["shape"] if k < 0.75 else o["color"]
    return rng.choice(ALL_CONCEPTS)


def concept_tokens():
    """全コンセプト（15個）のトークン列とパディングを返す（段階1の事前学習で使う）。"""
    ids, pads = zip(*(tokenize(c) for c in ALL_CONCEPTS))
    return torch.tensor(ids, dtype=torch.long), torch.tensor(pads, dtype=torch.bool)


def make_pretrain_batch(batch_size, rng):
    """
    段階1（画像と文の対照学習）用のバッチ。
    labels[i, c] は、画像 i にコンセプト c に当てはまる物体が1つでもあれば 1 になる。
    枠やマスクは使わない（本物の事前学習も、画像と文の組だけで学習する）。
    """
    images, labels = [], []
    for _ in range(batch_size):
        img, objs, _ = random_scene(rng)
        images.append(image_to_tensor(img))
        labels.append([float(any(matches(c, o) for o in objs)) for c in ALL_CONCEPTS])
    return torch.stack(images), torch.tensor(labels)


def make_batch(batch_size, rng, exemplar_prob=0.3):
    """
    この関数は、学習用のバッチを作る。
    exemplar_prob の確率で、正解物体のうち1つの枠を「見本（exemplar）」としてプロンプトに加える。
    """
    images, tokens, pads, targets, concepts = [], [], [], [], []
    ex_boxes, ex_valid = [], []
    for _ in range(batch_size):
        img, objs, masks = random_scene(rng)
        concept = pick_concept(rng, objs)
        t = targets_for(concept, objs, masks)
        ids, pad = tokenize(concept)
        images.append(image_to_tensor(img))
        tokens.append(ids)
        pads.append(pad)
        targets.append(t)
        concepts.append(concept)
        if len(t["boxes"]) > 0 and rng.random() < exemplar_prob:
            ex_boxes.append([t["boxes"][rng.randrange(len(t["boxes"]))].tolist()])
            ex_valid.append([True])
        else:
            ex_boxes.append([[0.5, 0.5, 0.1, 0.1]])  # 使わない見本（パディングで無視される）
            ex_valid.append([False])
    return {
        "images": torch.stack(images),                          # (B,3,64,64)
        "tokens": torch.tensor(tokens, dtype=torch.long),       # (B,4)
        "text_pad": torch.tensor(pads, dtype=torch.bool),       # (B,4)  True=パディング
        "ex_boxes": torch.tensor(ex_boxes, dtype=torch.float32),  # (B,1,4)
        "ex_valid": torch.tensor(ex_valid, dtype=torch.bool),   # (B,1)
        "ex_labels": torch.ones(batch_size, 1, dtype=torch.long),  # (B,1) 1=正の見本
        "targets": targets,
        "concepts": concepts,
    }
