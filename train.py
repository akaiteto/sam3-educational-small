"""
train.py : mini SAM 3 を、論文と同じ順番の段階に分けてCPU学習する

  段階1  画像と文の対照学習（パーセプション・エンコーダの事前学習の代わり）
         画像エンコーダとテキストエンコーダだけを学習する。枠やマスクは使わない。
  段階2  検出器の学習（プレゼンストークンなし）
         画像側・言語側のエンコーダは学習率を1/10にして、事前学習の結果を壊しにくくする。
  段階3  プレゼンストークンを加えた追加学習（論文では学習率を下げた短い post-training）
         プレゼンスヘッドだけ通常の学習率、それ以外は1/10。

  python train.py                 # 既定（目安：2コアのCPUで20〜25分）
  python train.py --scale 0.5     # 全段階のステップ数を半分にする（目安：10〜12分）
  python train.py --no_box_rpb    # boxRPB を外して比べる（他に --no_hybrid --no_aux --no_lft --skip_stage1）

保存されるもの（それぞれ約4MB）：mini_sam3_stage1.pt / mini_sam3_stage2.pt / mini_sam3.pt（最終）
"""
import argparse
import os
import random
import time
from dataclasses import asdict

import torch
import torch.nn.functional as F

from data import concept_tokens, make_batch, make_pretrain_batch
from losses import compute_loss
from model import Config, MiniSam3

HERE = os.path.dirname(os.path.abspath(__file__))


def save(model, cfg, name):
    path = os.path.join(HERE, name)
    torch.save({"config": asdict(cfg), "model": model.state_dict()}, path)
    print(f"保存しました: {path}")


def make_optimizer(model, lr, slow_prefixes, slow_factor, fast_prefixes=()):
    """slow_prefixes で始まる名前のパラメータは学習率を slow_factor 倍にする。fast_prefixes は通常の学習率。"""
    slow, normal = [], []
    for name, p in model.named_parameters():
        is_slow = any(name.startswith(s) for s in slow_prefixes) and not any(name.startswith(f) for f in fast_prefixes)
        (slow if is_slow else normal).append(p)
    return torch.optim.AdamW([{"params": normal, "lr": lr}, {"params": slow, "lr": lr * slow_factor}],
                             weight_decay=1e-4)


def make_scheduler(opt, steps, warmup=100):
    return torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warmup) * (0.1 if s > 0.8 * steps else 1.0))


def log(step, steps, avg, t0, every=100):
    if step % every == 0 or step == 1:
        msg = " ".join(f"{k}={v:.3f}" for k, v in avg.items())
        print(f"  step {step:5d}/{steps}  {msg}  ({time.time() - t0:.0f}s)", flush=True)


def update_avg(avg, parts):
    for k, v in parts.items():
        avg[k] = 0.98 * avg.get(k, v) + 0.02 * v


def stage1(model, steps, lr, rng):
    print(f"\n===== 段階1：画像と文の対照学習（{steps}ステップ） =====")
    tokens, pad = concept_tokens()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = make_scheduler(opt, steps)
    avg, t0 = {}, time.time()
    for step in range(1, steps + 1):
        images, labels = make_pretrain_batch(32, rng)
        logits = model.contrastive_logits(images, tokens, pad)              # (32画像, 15文)
        loss = F.binary_cross_entropy_with_logits(logits, labels)           # SigLIP型（シグモイドの対照損失）
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        acc = ((logits > 0).float() == labels).float().mean().item()
        update_avg(avg, {"loss": loss.item(), "acc": acc})
        log(step, steps, avg, t0)


def detector_stage(model, cfg, stage, steps, lr, batch, rng):
    name = "検出器の学習（プレゼンスなし）" if stage == 2 else "プレゼンストークンを加えた追加学習"
    print(f"\n===== 段階{stage}：{name}（{steps}ステップ） =====")
    if stage == 2:
        opt = make_optimizer(model, lr, ("image_encoder.", "text_encoder."), 0.1)
    else:
        opt = make_optimizer(model, lr, ("",), 0.1, fast_prefixes=("presence_head.",))
    sched = make_scheduler(opt, steps)
    avg, t0 = {}, time.time()
    for step in range(1, steps + 1):
        b = make_batch(batch, rng)
        out = model(b["images"], b["tokens"], b["text_pad"], b["ex_boxes"], b["ex_valid"], b["ex_labels"])
        loss, parts = compute_loss(out, b["targets"], stage, cfg)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        parts["total"] = loss.item()
        update_avg(avg, parts)
        log(step, steps, avg, t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps1", type=int, default=800)
    ap.add_argument("--steps2", type=int, default=3000)
    ap.add_argument("--steps3", type=int, default=600)
    ap.add_argument("--scale", type=float, default=1.0, help="全段階のステップ数に掛ける倍率")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip_stage1", action="store_true", help="事前学習なしで比べる")
    ap.add_argument("--no_box_rpb", action="store_true")
    ap.add_argument("--no_hybrid", action="store_true")
    ap.add_argument("--no_aux", action="store_true")
    ap.add_argument("--no_lft", action="store_true")
    ap.add_argument("--init", default=None, help="この重みから始める（例：mini_sam3_stage2.pt）。設定も重みのものを使う")
    ap.add_argument("--stages", default="123", help="実行する段階（例：3 なら段階3だけ）")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    torch.set_num_threads(max(1, os.cpu_count() or 1))

    if args.init:
        ckpt = torch.load(os.path.join(HERE, args.init), map_location="cpu")
        cfg = Config(**ckpt["config"])
        model = MiniSam3(cfg)
        model.load_state_dict(ckpt["model"])
        print(f"重みを読み込みました: {args.init}")
    else:
        cfg = Config(use_box_rpb=not args.no_box_rpb,
                     num_hybrid=0 if args.no_hybrid else Config.num_hybrid,
                     aux_loss=not args.no_aux,
                     look_forward_twice=not args.no_lft)
        model = MiniSam3(cfg)
    model.train()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"パラメータ数: {n_params:,}  （本物のSAM 3は約8億5000万）")
    print(f"設定: boxRPB={cfg.use_box_rpb} 補助クエリ={cfg.num_hybrid} 補助損失={cfg.aux_loss} LFT={cfg.look_forward_twice}")

    s1, s2, s3 = (max(1, int(n * args.scale)) for n in (args.steps1, args.steps2, args.steps3))
    if "1" in args.stages and not args.skip_stage1:
        stage1(model, s1, args.lr, rng)
        save(model, cfg, "mini_sam3_stage1.pt")
    if "2" in args.stages:
        detector_stage(model, cfg, 2, s2, args.lr, args.batch, rng)
        save(model, cfg, "mini_sam3_stage2.pt")
    if "3" in args.stages:
        detector_stage(model, cfg, 3, s3, args.lr, args.batch, rng)
        save(model, cfg, "mini_sam3.pt")


if __name__ == "__main__":
    main()
