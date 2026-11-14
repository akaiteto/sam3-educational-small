"""
demo_pretrain.py : 段階1（画像と文の対照学習）の結果を確認する

  python demo_pretrain.py                         # mini_sam3_stage1.pt を使う
  python demo_pretrain.py --ckpt mini_sam3.pt     # 最終の重みで、事前学習の性質が残っているかを見る

画像6枚と15個のコンセプトの一致度（シグモイド後）を表にし、demo_pretrain.png に保存する。
白い枠は正解（その画像にそのコンセプトに当てはまる物体がある）を表す。
枠やマスクを一度も使っていないのに、画像エンコーダとテキストエンコーダが「言葉と画像」を結びつけているかを確かめる。
"""
import argparse
import os
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

import model as M
from data import ALL_CONCEPTS, concept_tokens, image_to_tensor, matches, random_scene

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(HERE, "mini_sam3_stage1.pt"))
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu")
    net = M.MiniSam3(M.Config(**ckpt["config"]))
    net.load_state_dict(ckpt["model"])
    net.eval()
    print(f"読み込みました: {args.ckpt}")

    rng = random.Random(args.seed)
    scenes = [random_scene(rng) for _ in range(6)]
    images = torch.stack([image_to_tensor(s[0]) for s in scenes])
    tokens, pad = concept_tokens()
    with torch.no_grad():
        sim = net.contrastive_logits(images, tokens, pad).sigmoid()   # (6, 15)

    fig = plt.figure(figsize=(14, 5.5))
    for i, (img, objs, _) in enumerate(scenes):
        ax = fig.add_subplot(2, 6, i + 1)
        ax.imshow(img)
        ax.set_title(f"img {i}", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    ax = fig.add_subplot(2, 1, 2)
    ax.imshow(sim.numpy(), cmap="viridis", vmin=0, vmax=1, aspect="auto")
    for i, (_, objs, _) in enumerate(scenes):
        for c, concept in enumerate(ALL_CONCEPTS):
            if any(matches(concept, o) for o in objs):
                ax.add_patch(plt.Rectangle((c - 0.5, i - 0.5), 1, 1, fill=False, color="white", lw=2))
    ax.set_xticks(range(len(ALL_CONCEPTS)))
    ax.set_xticklabels(ALL_CONCEPTS, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(6))
    ax.set_yticklabels([f"img {i}" for i in range(6)])
    ax.set_title("image-text similarity after stage 1 (white box = correct)")
    plt.tight_layout()
    path = os.path.join(HERE, "demo_pretrain.png")
    plt.savefig(path, dpi=110)
    acc = ((sim > 0.5).float() == torch.tensor(
        [[float(any(matches(c, o) for o in objs)) for c in ALL_CONCEPTS] for _, objs, _ in scenes])).float().mean()
    print(f"6枚 x 15コンセプトの正解率: {acc:.3f}")
    print(f"保存しました: {path}")


if __name__ == "__main__":
    main()
