"""
losses.py : SAM 3 方式の損失

1. ハンガリアン法で、正解の物体1つにクエリ1つを割り当てる（1対1、NMS不要の理由）
2. 割り当てたクエリに：一致スコアの損失（focal）、枠の損失（L1 ＋ GIoU）、マスクの損失（BCE ＋ Dice）
3. 補助損失：デコーダの各層の出力にも、一致スコアと枠の損失をかける
4. ハイブリッドマッチング：補助クエリには、正解1つを k 個のクエリに割り当てる（1対多）
5. 意味的ヘッド：各画素が「プロンプトに当てはまる物体のどれか」に含まれるか（BCE ＋ Dice）
6. プレゼンストークン：画像内にコンセプトが存在するか（BCE）。段階3から加える

SAM 3 の特徴（論文 付録C）：
  段階3では、コンセプトが画像内に存在しない画像で、クエリの一致スコアを学習しない。
  「存在するか」はプレゼンストークンだけが学び、クエリは「存在するなら、どれが一致するか」だけを学ぶ。
  段階2（プレゼンストークンなし）では、存在しない画像で全クエリを「一致しない」として学習する。
"""
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

W_CLS, W_L1, W_GIOU, W_BCE, W_DICE = 2.0, 5.0, 2.0, 5.0, 5.0


def box_cxcywh_to_xyxy(b):
    cx, cy, w, h = b.unbind(-1)
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)


def generalized_iou(a, b, eps=1e-7):
    """a:(N,4) b:(M,4) xyxy -> (N,M)"""
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    lt = torch.max(a[:, None, :2], b[None, :, :2])
    rb = torch.min(a[:, None, 2:], b[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    union = area_a[:, None] + area_b[None] - inter
    iou = inter / (union + eps)
    lt2 = torch.min(a[:, None, :2], b[None, :, :2])
    rb2 = torch.max(a[:, None, 2:], b[None, :, 2:])
    wh2 = (rb2 - lt2).clamp(min=0)
    area_c = wh2[..., 0] * wh2[..., 1]
    return iou - (area_c - union) / (area_c + eps)


def sigmoid_focal_loss(logits, targets, alpha=0.25, gamma=2.0):
    p = logits.sigmoid()
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce * (1 - p_t) ** gamma
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    return (alpha_t * loss).sum()


def dice_loss(logits, targets):
    """logits, targets: (n, 画素数) -> (n,)"""
    p = logits.sigmoid()
    num = 2 * (p * targets).sum(-1) + 1
    den = p.sum(-1) + targets.sum(-1) + 1
    return 1 - num / den


@torch.no_grad()
def hungarian_match(logits, boxes, tgt_boxes):
    """コスト行列（クエリ数 x 正解数）を作り、合計コストが最小になる組を選ぶ。"""
    c_cls = -logits.sigmoid()[:, None]
    c_l1 = torch.cdist(boxes, tgt_boxes, p=1)
    c_giou = -generalized_iou(box_cxcywh_to_xyxy(boxes), box_cxcywh_to_xyxy(tgt_boxes))
    C = 2 * c_cls + 5 * c_l1 + 2 * c_giou
    r, c = linear_sum_assignment(C.cpu().numpy())
    return torch.as_tensor(r, dtype=torch.long), torch.as_tensor(c, dtype=torch.long)


def detection_loss(logits, boxes, targets, supervise_absent, masks=None, repeat=1):
    """
    1組のクエリ出力に対する損失。
      logits (B,N), boxes (B,N,4), masks (B,N,R,R) または None
      repeat > 1 のとき、正解を repeat 回複製して割り当てる（1対多＝ハイブリッドマッチング）
    """
    zero = logits.sum() * 0
    acc = {"cls": zero, "l1": zero, "giou": zero, "mask_bce": zero, "dice": zero}
    N = logits.shape[1]
    num_pos = max(sum(len(t["boxes"]) for t in targets) * repeat, 1)
    for b, t in enumerate(targets):
        n = len(t["boxes"])
        if n == 0:
            if supervise_absent:  # 段階2：存在しない画像では、全クエリを「一致しない」として学習
                acc["cls"] = acc["cls"] + sigmoid_focal_loss(logits[b], torch.zeros(N))
            continue
        tgt = t["boxes"].repeat(repeat, 1)
        r, c = hungarian_match(logits[b], boxes[b], tgt)
        cls_t = torch.zeros(N)
        cls_t[r] = 1.0
        acc["cls"] = acc["cls"] + sigmoid_focal_loss(logits[b], cls_t)
        pb, tb = boxes[b][r], tgt[c]
        acc["l1"] = acc["l1"] + F.l1_loss(pb, tb, reduction="sum")
        acc["giou"] = acc["giou"] + (1 - torch.diag(generalized_iou(box_cxcywh_to_xyxy(pb), box_cxcywh_to_xyxy(tb)))).sum()
        if masks is not None:  # マスクは1対1のクエリにだけかける（repeat=1 のとき）
            pm = masks[b][r]                                                    # (n,32,32)
            tm = F.interpolate(t["masks"][c][:, None], size=pm.shape[-2:], mode="area")[:, 0]
            acc["mask_bce"] = acc["mask_bce"] + F.binary_cross_entropy_with_logits(pm, tm, reduction="none").mean((1, 2)).sum()
            acc["dice"] = acc["dice"] + dice_loss(pm.flatten(1), tm.flatten(1)).sum()
    w = {"cls": W_CLS, "l1": W_L1, "giou": W_GIOU, "mask_bce": W_BCE, "dice": W_DICE}
    return {k: w[k] * v / num_pos for k, v in acc.items()}


def semantic_loss(sem_logits, targets):
    """意味的ヘッド：プロンプトに当てはまる全物体のマスクを1枚に重ねたものが正解。"""
    R = sem_logits.shape[-1]
    unions = []
    for t in targets:
        if len(t["masks"]) > 0:
            unions.append(t["masks"].amax(0))
        else:
            unions.append(torch.zeros(t["masks"].shape[-2:]))
    tgt = torch.stack(unions)[:, None]                                          # (B,1,64,64)
    tgt = F.interpolate(tgt, size=(R, R), mode="area")
    bce = F.binary_cross_entropy_with_logits(sem_logits, tgt)
    dice = dice_loss(sem_logits.flatten(1), tgt.flatten(1)).mean()
    return 2 * bce + 2 * dice


def compute_loss(out, targets, stage, cfg):
    """stage: 2=検出器の学習（プレゼンスなし）、3=プレゼンストークンを加えた学習"""
    absent = (stage == 2)
    parts = detection_loss(out["pred_logits"], out["pred_boxes"], targets, absent, masks=out["pred_masks"])

    if cfg.aux_loss:
        aux = out["pred_logits"].sum() * 0
        for a in out["aux"]:
            aux = aux + sum(detection_loss(a["logits"], a["boxes"], targets, absent).values())
        parts["aux"] = aux

    if out["hybrid"] is not None:
        h = out["hybrid"]
        parts["hybrid"] = sum(detection_loss(h["logits"], h["boxes"], targets, absent, repeat=cfg.hybrid_k).values())

    parts["semantic"] = semantic_loss(out["semantic"], targets)

    if stage == 3:
        present = torch.tensor([float(len(t["boxes"]) > 0) for t in targets])
        parts["presence"] = 2 * F.binary_cross_entropy_with_logits(out["presence_logit"], present)

    total = sum(parts.values())
    return total, {k: float(v.detach()) for k, v in parts.items()}
