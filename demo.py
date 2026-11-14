"""
demo.py : mini SAM 3 を1枚の画像で動かし、各工程と結果を確認する

  python demo.py                              # 最終の重み mini_sam3.pt を使う（なければランダムな重み）
  python demo.py --prompts "red circle" blue  # コンセプトを指定
  python demo.py --seed 3                     # 別の画像
  python demo.py --ckpt mini_sam3_stage2.pt --no_presence
                                              # 段階2の重み（プレゼンス未学習）で、プレゼンスを掛けずに見る

出力
  ・コンソール：各工程のテンソルの形、クエリごとのスコア
  ・demo_result.png   ：入力画像、プロンプトごとのマスクと枠、意味的ヘッドの出力、クエリのスコア
  ・demo_attention.png：融合エンコーダの各層で、画像の各マス目が各プロンプトトークンをどれだけ見たか
"""
import argparse
import os
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

import model as M
from data import ALL_CONCEPTS, IMG, image_to_tensor, matches, random_scene, tokenize

HERE = os.path.dirname(os.path.abspath(__file__))


def load_model(path):
    if os.path.exists(path):
        ckpt = torch.load(path, map_location="cpu")
        net = M.MiniSam3(M.Config(**ckpt["config"]))
        net.load_state_dict(ckpt["model"])
        print(f"学習済みの重みを読み込みました: {path}")
    else:
        net = M.MiniSam3(M.Config())
        print(f"{path} がないので、ランダムな重みで動かします（形の流れの確認用）")
    return net.eval()


def default_prompts(objs):
    o = objs[0]
    absent = [c for c in ALL_CONCEPTS if not any(matches(c, x) for x in objs)]
    return [f"{o['color']} {o['shape']}", o["shape"], absent[0] if absent else "green"]


def run(net, img_t, prompt):
    ids, pad = tokenize(prompt)
    with torch.no_grad():
        return net(img_t[None], torch.tensor([ids]), torch.tensor([pad]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--prompts", nargs="*", default=None)
    ap.add_argument("--ckpt", default=os.path.join(HERE, "mini_sam3.pt"))
    ap.add_argument("--th", type=float, default=0.5, help="最終スコアのしきい値")
    ap.add_argument("--no_presence", action="store_true", help="プレゼンスを掛けずに、クエリの一致スコアだけで判定する")
    args = ap.parse_args()

    net = load_model(args.ckpt)
    img, objs, masks = random_scene(random.Random(args.seed))
    img_t = image_to_tensor(img)
    print("画像に描かれている物体:", ", ".join(f"{o['color']} {o['shape']}" for o in objs))
    prompts = args.prompts or default_prompts(objs)

    results = []
    for i, p in enumerate(prompts):
        print(f"\n==== プロンプト「{p}」 ====")
        M.VERBOSE = (i == 0)  # 1つ目のプロンプトだけ、全工程の形を表示する
        out = run(net, img_t, p)
        M.VERBOSE = False
        presence = out["presence_logit"].sigmoid()[0].item()
        q_prob = out["pred_logits"].sigmoid()[0]
        scores = q_prob if args.no_presence else out["scores"][0]
        print(f"  p(存在する) = {presence:.3f}" + ("  （--no_presence のため掛けない）" if args.no_presence else ""))
        print("  クエリ  p(一致|存在)  最終スコア  枠(cx,cy,w,h)[画素]")
        for n in scores.argsort(descending=True)[:5].tolist():
            b = (out["pred_boxes"][0, n] * IMG).tolist()
            mark = " <- 採用" if scores[n] > args.th else ""
            print(f"  {n:4d}    {q_prob[n]:.3f}        {scores[n]:.3f}     "
                  f"({b[0]:.0f},{b[1]:.0f},{b[2]:.0f},{b[3]:.0f}){mark}")
        attn = [layer.cross.last_attn[0].mean(0) for layer in net.fusion.layers]  # 層ごと (64マス, トークン)
        results.append((p, out, presence, scores, attn))

    # ---- 結果の図：1行目=マスクと枠、2行目=意味的ヘッド、3行目=クエリのスコア ----
    P = len(prompts)
    fig, axes = plt.subplots(3, P + 1, figsize=(3.2 * (P + 1), 9.4))
    axes[0, 0].imshow(img)
    axes[0, 0].set_title("input")
    axes[1, 0].axis("off")
    axes[2, 0].axis("off")
    axes[1, 0].text(0, 0.5, "objects:\n" + "\n".join(f"{o['color']} {o['shape']}" for o in objs), fontsize=11)
    colors = plt.cm.tab10.colors
    for j, (p, out, presence, scores, _) in enumerate(results, start=1):
        ax = axes[0, j]
        ax.imshow(img)
        keep = (scores > args.th).nonzero().flatten().tolist()
        if keep:
            mlog = F.interpolate(out["pred_masks"][:, keep], size=(IMG, IMG), mode="bilinear", align_corners=False)[0]
        for k, n in enumerate(keep):
            c = colors[k % 10]
            m = (mlog[k].sigmoid() > 0.5).numpy()
            overlay = np.zeros((IMG, IMG, 4))
            overlay[m] = (*c, 0.5)
            ax.imshow(overlay)
            cx, cy, w, h = (out["pred_boxes"][0, n] * IMG).tolist()
            ax.add_patch(plt.Rectangle((cx - w / 2 - 0.5, cy - h / 2 - 0.5), w, h, fill=False, color=c, lw=2))
            ax.text(cx - w / 2, cy - h / 2 - 1, f"{scores[n]:.2f}", color=c, fontsize=8)
        ax.set_title(f'"{p}"\npresence={presence:.2f}  found={len(keep)}')

        sem = F.interpolate(out["semantic"], size=(IMG, IMG), mode="bilinear", align_corners=False)[0, 0].sigmoid()
        axes[1, j].imshow(sem.numpy(), cmap="magma", vmin=0, vmax=1)
        axes[1, j].set_title("semantic head")

        ax2 = axes[2, j]
        ax2.bar(range(len(scores)), out["pred_logits"].sigmoid()[0].numpy(), color="lightgray", label="p(match|present)")
        ax2.bar(range(len(scores)), scores.numpy(), color="tab:blue", label="final score")
        ax2.axhline(presence, color="tab:red", ls="--", label="p(present)")
        ax2.axhline(args.th, color="black", ls=":", lw=1)
        ax2.set_ylim(0, 1)
        ax2.set_xlabel("object query")
        ax2.legend(fontsize=7, loc="upper right")
    for ax in list(axes[0]) + list(axes[1]):
        ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    path = os.path.join(HERE, "demo_result.png")
    plt.savefig(path, dpi=110)
    print(f"\n保存しました: {path}")

    # ---- 融合エンコーダのクロスアテンションの図（1つ目のプロンプト、層ごと）----
    p, _, _, _, attn = results[0]
    words = ["<bos>"] + p.split() + ["<eos>"]
    G = net.cfg.grid
    L = len(attn)
    fig, axes = plt.subplots(L, len(words) + 1, figsize=(2.6 * (len(words) + 1), 2.6 * L), squeeze=False)
    for li in range(L):
        axes[li, 0].imshow(img)
        axes[li, 0].set_title(f"layer {li + 1}")
        for k, wd in enumerate(words):
            axes[li, k + 1].imshow(attn[li][:, k].reshape(G, G).numpy(), cmap="viridis", vmin=0, vmax=1)
            axes[li, k + 1].set_title(f'"{wd}"')
    for ax in axes.flat:
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f'fusion encoder: image(Q) -> prompt(K,V) attention per layer, prompt="{p}"')
    plt.tight_layout()
    path = os.path.join(HERE, "demo_attention.png")
    plt.savefig(path, dpi=110)
    print(f"保存しました: {path}")


if __name__ == "__main__":
    main()
