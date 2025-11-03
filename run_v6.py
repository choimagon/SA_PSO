# ============================================================
# 하이브리드 하이퍼파라미터 탐색 (SIH+정체종료, 로그/타이밍 확장판, ★단계별 배너 로그 강화★)
# - L/U/D: 스텝 없는 동시 구간 절반(SIH)로 전역 축소 → 최종 추정점
# - 최종점 근방에서 Model1(prev)/Model2(*)/Model3(next) 생성
# - 각 베이스: (활성화·초기화 전수: ReLU/LeakyReLU/Sigmoid/Tanh)
#              → PSO(optimizer + lr + weight_decay 동시 탐색)
# - 데이터: CIFAR-10 (3072 입력), 에폭 10, 단일 신뢰수준 과적합 판정
# - 추가: 각 단계 시간, 에폭별 acc/loss, 최종 파라미터를 run_log.txt에 기록
# - [CHANGED] 오버피팅 감지 시: 조기종료 + 후보 제외(점수 -inf 처리)
# - [NEW] 단계 1~5에서 선택 결과를 배너 형태로 크게 출력
# - [NEW] SIH(1단계)에서만 act/init/optimizer 랜덤 샘플링으로 (L,U,D) 편향 완화
# - [UPDATED] 오버피팅 체크: Loss + Accuracy 동시 판정
# - [NEW] 갭 결합 트리거: signed + abs(게이트드) 동시 사용
# ============================================================

import os, sys, math, copy, time, random, subprocess, datetime
from typing import Dict, Any, Tuple, List
import numpy as np

# ---------- 필요한 패키지 자동 설치 ----------
def _ensure(pkg):
    try:
        __import__(pkg)
    except ImportError:
        print(f"[setup] installing {pkg} ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

for _p in ["torch", "torchvision", "matplotlib", "pyyaml"]:
    _ensure(_p)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
import yaml

# ============================================================
# [A] 하이퍼파라미터 & 전역 설정
# ============================================================
import time

now = time.localtime()   # 현재 로컬 시간
tss = time.strftime("%Y%m%d_%H%M%S", now)

with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

val = cfg["value"]
print("현재 설정 값:", val)

# ---- 재현성 & 로깅 ----
SEED            = 123
VERBOSE         = True
LOG_PATH        = f"run_log_{cfg['value']}_{tss}.txt"

# ---- 디바이스/학습 루프 ----
GLOBAL_EPOCHS   = 10     # 프록시/정밀/최종 공통
FINAL_EPOCHS    = 10
DEFAULT_OF_CONF = cfg["value"]     # 과적합 단일 신뢰수준(%): 90/95/98/99 등

# ---- 데이터/로더 ----
DATA_ROOT       = "./data"
BATCH_TRAIN     = 256
BATCH_TEST      = 1024
VAL_RATIO       = 0.2
IN_DIM          = 3 * 32 * 32
OUT_DIM         = 10

# ---- 탐색 구간(스텝 없음) ----
LAYER_MIN, LAYER_MAX = 2, 10          # 정수, 스텝 없음
UNITS_MIN, UNITS_MAX = 64, 1024       # 정수, 스텝 없음
DROPOUT_MIN, DROPOUT_MAX = 0.0, 0.5   # 연속

# ---- SIH(동시 구간 절반) 설정 ----
SIH_ROUNDS          = 7
SIH_TOP_K           = 3
SIH_RAND_PER_ROUND  = 2
# ★ 라운드 개선 정체(early stop) 종료 조건
SIH_IMPROVE_TOL         = 5e-4
SIH_NO_IMPROVE_PATIENCE = 2

# prev/next 생성폭(최종점 주변):
L_NEIGHBOR_DELTA  = 1
U_NEIGHBOR_DELTA  = max(16, (UNITS_MAX - UNITS_MIN)//10)
D_NEIGHBOR_DELTA  = 0.15

# ---- 활성화·초기화 전수 후보 ----
#   규칙: ReLU/LeakyReLU → He,  Sigmoid/Tanh → Xavier
ACTIVATION_CAND = ["relu", "leakyrelu", "sigmoid", "tanh"]
USE_ALT_INIT    = False

# ---- (PSO가 탐색할) 옵티마이저 범주 후보 ----
OPTIM_CAND      = ["Adam", "AdamW", "SGD"]

# ---- PSO(빠른 설정): 옵티마이저+lr+wd 동시 탐색 ----
PSO_PARTICLES   = 5
PSO_ITERS       = 6
PSO_EARLY_PATI  = 2
PSO_IMPROVE_TOL = 5e-4
PSO_LR_BOUNDS   = (1e-5, 1e-1)
PSO_WD_BOUNDS   = (1e-6, 1e-2)
PSO_VEL_TOL     = 1e-3
PSO_DIV_TOL     = 5e-3
PSO_INERTIA     = 0.7
PSO_C1          = 1.4
PSO_C2          = 1.4
PSO_INERTIA_DEC = 0.95
PSO_KICK_ON     = True
PSO_KICK_SCALE  = 2e-3
PSO_TARGET_VAL  = None
PSO_TIME_LIMIT  = None
P_ADOPT_PBEST   = 0.5
P_ADOPT_GBEST   = 0.3
P_MUTATE_OPT    = 0.10

# ---- 플롯 출력 파일명 ----
SAVE_LOSS_PNG   = f"learning_curves_{cfg['value']}_{tss}.png"
SAVE_LOSS_SVG   = f"learning_curves_{cfg['value']}_{tss}.svg"
SAVE_ACC_PNG    = f"learning_curves_acc_{cfg['value']}_{tss}.png"
SAVE_ACC_SVG    = f"learning_curves_acc_{cfg['value']}_{tss}.svg"

# ---- SIH 프록시 랜덤 설정 (L/U/D 점수 평가 시만 사용) ----
PROXY_RAND_ACTOPT   = True   # 활성화/초기화/옵티마이저를 랜덤 샘플링해 평가
PROXY_REPEATS       = 1      # 같은 (L,U,D)에 대해 랜덤 조합을 여러 번 시도해 최고값 사용
PROXY_FIXED_LR      = 1e-3   # 프록시 평가는 lr 고정
PROXY_FIXED_WD      = 1e-4   # 프록시 평가는 weight_decay 고정
PROXY_SEEDED_BY_CFG = False  # True면 (L,U,D)별 고정 난수시드 → 재현성 ↑

# ============================================================
# [B] 로거/시드/디바이스 (+ 배너 유틸 추가)
# ============================================================

def _init_log():
    with open(LOG_PATH, "w", encoding="utf-8") as f:
        f.write("=== Hybrid HParam Search Run Log (SIH version) ===\n")
        f.write(f"start_time: {datetime.datetime.now().isoformat()}\n\n")

def _writeln(msg: str):
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(msg + "\n")

def log(*args, **kwargs):
    s = " ".join(str(a) for a in args)
    if VERBOSE:
        print(*args, **kwargs)
    _writeln(s)

def banner(title: str, ch: str = "=", width: int = 72):
    line = ch * width
    msg = f"\n{line}\n{title}\n{line}"
    log(msg)

def log_kv(title: str, d: Dict[str, Any], width_key: int = 14):
    log(f"{title}")
    for k in sorted(d.keys()):
        log(f"  {k:>{width_key}s} : {d[k]}")

def log_hp_table(stage_title: str, hp: Dict[str, Any]):
    banner(stage_title, "=", 80)
    log_kv("【SELECTED HYPERPARAMETERS】", hp, width_key=16)
    log("="*80)

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# [C] 과적합 판정 (단일 신뢰수준, Loss + Accuracy 동시)
#      + 갭 결합 트리거(signed + abs 게이트드) 추가
# ============================================================

def _norm_ppf(p):
    """Acklam 근사: 표준정규분포 역 CDF."""
    if not (0.0 < p < 1.0):
        if p == 0.0: return -np.inf
        if p == 1.0: return  np.inf
        raise ValueError("p must be in (0,1)")
    a = [-3.969683028665376e+01,  2.209460984245205e+02,
         -2.759285104469687e+02,  1.383577518672690e+02,
         -3.066479806614716e+01,  2.506628277459239e+00]
    b = [-5.447609879822406e+01,  1.615858368580409e+02,
         -1.556989798598866e+02,  6.680131188771972e+01,
         -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e+00, -2.549732539343734e+00,
          4.374664141464968e+00,  2.938163982698783e+00]
    d = [ 7.784695709041462e-03,  3.224671290700398e-01,
          2.445134137142996e+00,  3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = np.sqrt(-2*np.log(p))
        return (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
               ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1.0)
    if p > phigh:
        q = np.sqrt(-2*np.log(1.0 - p))
        return -(((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
                 ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1.0)
    q = p - 0.5
    r = q*q
    return (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5]) * q / \
           (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1.0)

def _conf_to_z_two_sided(conf_percent):
    c = float(conf_percent)
    if not (0.0 < c < 100.0):
        raise ValueError("conf_level은 (0,100)이어야 함")
    p = 0.5 + c / 200.0
    return float(_norm_ppf(p))

def _robust_std(x):
    x = np.asarray(x, dtype=float)
    mad = np.median(np.abs(x - np.median(x)))
    s = 1.4826 * mad
    if s == 0:
        s = max(1e-12, 1e-3 * (np.median(np.abs(x)) + 1e-12))
    return s

# ----- 추가: EMA & 갭 결합 트리거 -----
def _ema(x, alpha=0.6):
    x = np.asarray(x, dtype=float).flatten()
    if x.size == 0:
        return x
    y = np.empty_like(x)
    y[0] = x[0]
    for i in range(1, x.size):
        y[i] = alpha * x[i] + (1 - alpha) * y[i-1]
    return y

def gap_triggers_combined(train_seq, val_seq, zthr,
                          tail_k=3, use_ema=True, ema_alpha=0.6,
                          abs_gate_eps=1e-6, mode="gated_or", zthr_abs_offset=0.0):
    """
    train_seq, val_seq: 같은 길이의 시계열 (loss 또는 acc에 맞춰서 넘김)
    zthr: 임계 Z 점수(예: conf=95 → ~1.96)
    반환: dict(flag, signed_z, abs_z, gate, mode)
    - signed_z: 최근 평균 gap(=val-train 또는 train-val)이 base 중앙값 대비 ↑된 정도(Z)
    - abs_z   : |gap|의 최근 평균이 base 중앙값 대비 ↑된 정도(Z), 단 gate가 True일 때만 의미
    """
    tr = np.asarray(train_seq, dtype=float).flatten()
    va = np.asarray(val_seq,   dtype=float).flatten()
    n = tr.size
    if n < 4:
        return dict(flag=False, signed_z=0.0, abs_z=0.0, gate=False, mode=mode)

    gap = va - tr                      # loss용 정의(정확도용은 호출부에서 부호 바꿔서 넘김)
    base_k = max(3, n // 3)
    tail_k = int(max(2, min(tail_k, n)))
    base = gap[:base_k]
    recent_seq = gap[-tail_k:]

    mu_gap = float(np.median(base))
    sigma_gap = _robust_std(base)
    recent_gap = float(_ema(recent_seq, ema_alpha)[-1]) if use_ema else float(np.mean(recent_seq))
    gap_delta = recent_gap - mu_gap
    signed_z = float(gap_delta / sigma_gap) if gap_delta > 0 else 0.0
    flag_signed = (signed_z >= zthr)

    abs_base = np.abs(base)
    mu_agap = float(np.median(abs_base))
    sigma_agap = _robust_std(abs_base)
    recent_agap = float(_ema(np.abs(recent_seq), ema_alpha)[-1]) if use_ema else float(np.mean(np.abs(recent_seq)))
    abs_delta = recent_agap - mu_agap
    abs_zthr = zthr + float(zthr_abs_offset)
    abs_z = float(abs_delta / sigma_agap) if abs_delta > 0 else 0.0

    gate = (recent_gap >= abs_gate_eps)  # 최근 평균이 과적합(+방향)일 때만 abs 인정
    flag_abs = gate and (abs_z >= abs_zthr)

    if mode == "gated_or":
        flag = bool(flag_signed or flag_abs)
    elif mode == "and":
        flag = bool(flag_signed and flag_abs)
    elif mode == "z_fuse":
        z_fused = 0.6 * signed_z + 0.4 * (abs_z if gate else 0.0)
        flag = bool(z_fused >= zthr)
    else:
        raise ValueError("mode must be one of ['gated_or','and','z_fuse']")

    return dict(flag=flag, signed_z=signed_z, abs_z=abs_z, gate=gate, mode=mode)

def _overfit_by_loss(train_loss, val_loss, zthr,
                     gap_mode="gated_or", zthr_abs_offset=0.0,
                     tail_k=3, use_ema=True, ema_alpha=0.6) -> bool:
    tr = np.asarray(train_loss, dtype=float).flatten()
    va = np.asarray(val_loss, dtype=float).flatten()
    n = tr.size
    if n < 4:
        return False

    d_tr = np.diff(tr)          # 기대: train_loss ↓
    d_va = np.diff(va)          # 기대: val_loss ↓
    base_len = max(5, n // 2)
    base_d_va = d_va[:max(1, base_len-1)]
    sigma_d_va = _robust_std(base_d_va)

    spike_mask = (d_tr < 0) & (d_va > 0)  # train ↓, val ↑
    spike_z = float(np.max(d_va[spike_mask] / sigma_d_va)) if np.any(spike_mask) else 0.0

    past_min = np.min(va[:-1])
    rebound_delta = va[-1] - past_min
    rebound_z = float(rebound_delta / sigma_d_va) if rebound_delta > 0 else 0.0

    # ----- 갭 결합 트리거(signed + abs 게이트드)
    combo = gap_triggers_combined(
        train_seq=tr, val_seq=va, zthr=zthr,
        tail_k=tail_k, use_ema=use_ema, ema_alpha=ema_alpha,
        abs_gate_eps=1e-6, mode=gap_mode, zthr_abs_offset=zthr_abs_offset
    )
    gap_flag = combo["flag"]

    return bool((spike_z >= zthr) or (rebound_z >= zthr) or gap_flag)

def _overfit_by_acc(train_acc, val_acc, zthr,
                    gap_mode="gated_or", zthr_abs_offset=0.0,
                    tail_k=3, use_ema=True, ema_alpha=0.6) -> bool:
    if train_acc is None or val_acc is None:
        return False
    tr = np.asarray(train_acc, dtype=float).flatten()
    va = np.asarray(val_acc, dtype=float).flatten()
    n = tr.size
    if n < 4 or va.size != tr.size:
        return False

    d_tr = np.diff(tr)          # 기대: train_acc ↑
    d_va = np.diff(va)          # 기대: val_acc ↑
    base_len = max(5, n // 2)
    base_d_va = d_va[:max(1, base_len-1)]
    sigma_d_va = _robust_std(base_d_va)

    spike_mask = (d_tr > 0) & (d_va < 0)
    spike_z = float(np.max((-d_va[spike_mask]) / sigma_d_va)) if np.any(spike_mask) else 0.0

    past_max = np.max(va[:-1])
    rebound_delta = past_max - va[-1]  # 과거 최고 − 현재 (양수면 악화)
    rebound_z = float(rebound_delta / sigma_d_va) if rebound_delta > 0 else 0.0

    # ----- 정확도용 갭 결합 트리거
    # 정확도에서는 과적합 방향이 (train_acc - val_acc) ↑ 이므로
    # gap_triggers_combined의 정의(gap=va-tr)를 재사용하려면 부호를 뒤집어 넘기면 됨.
    # 즉 train_seq' = val_acc, val_seq' = train_acc 로 넘겨서 (val'-train') = (train_val_gap) 효과.
    combo = gap_triggers_combined(
        train_seq=va,  # 바꿔치기
        val_seq=tr,    # 바꿔치기
        zthr=zthr,
        tail_k=tail_k, use_ema=use_ema, ema_alpha=ema_alpha,
        abs_gate_eps=1e-6, mode=gap_mode, zthr_abs_offset=zthr_abs_offset
    )
    gap_flag = combo["flag"]

    return bool((spike_z >= zthr) or (rebound_z >= zthr) or gap_flag)

def overfit_checking(train_loss, val_loss,
                     train_acc=None, val_acc=None,
                     conf_level=DEFAULT_OF_CONF,
                     gap_mode="gated_or",
                     zthr_abs_offset=0.0,
                     tail_k=3, use_ema=True, ema_alpha=0.6):
    """
    손실(loss)과 정확도(accuracy) 모두에 대해 동일한 통계적 징후(스파이크/리바운드/갭결합)를 점검.
    둘 중 하나라도 임계 Z 점수를 넘으면 오버피팅으로 간주.
    gap_mode: "gated_or"(기본) | "and" | "z_fuse"
    zthr_abs_offset: abs-gap 트리거만 약간 더 엄격히(+)
    """
    trL = np.asarray(train_loss, dtype=float).flatten()
    vaL = np.asarray(val_loss, dtype=float).flatten()
    if trL.size != vaL.size:
        raise ValueError("train_loss와 val_loss 길이가 달라요.")
    if trL.size < 4:
        return False

    zthr = _conf_to_z_two_sided(conf_level)

    loss_flag = _overfit_by_loss(trL, vaL, zthr,
                                 gap_mode=gap_mode, zthr_abs_offset=zthr_abs_offset,
                                 tail_k=tail_k, use_ema=use_ema, ema_alpha=ema_alpha)

    acc_flag  = _overfit_by_acc(train_acc, val_acc, zthr,
                                gap_mode=gap_mode, zthr_abs_offset=zthr_abs_offset,
                                tail_k=tail_k, use_ema=use_ema, ema_alpha=ema_alpha) \
                if (train_acc is not None and val_acc is not None) else False

    return bool(loss_flag or acc_flag)

# ============================================================
# [D] 데이터셋 & 데이터로더
# ============================================================

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                         std=[0.2470, 0.2435, 0.2616]),
    transforms.Lambda(lambda x: torch.flatten(x))  # (3,32,32)->(3072,)
])

# ============================================================
# [E] 모델 & 초기화 규칙
# ============================================================

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hid_dim, n_layer, act, dropout, use_bn=True, w_initial="he"):
        super().__init__()
        self.act_name = act.lower()
        if   self.act_name == "relu":      activation = nn.ReLU()
        elif self.act_name == "leakyrelu": activation = nn.LeakyReLU(0.01)
        elif self.act_name == "sigmoid":   activation = nn.Sigmoid()
        elif self.act_name == "tanh":      activation = nn.Tanh()
        else: raise ValueError(f"Unknown act: {act}")
        Drop = nn.Dropout

        layers = [nn.Linear(in_dim, hid_dim)]
        if use_bn: layers.append(nn.BatchNorm1d(hid_dim))
        layers += [activation, Drop(p=dropout)]
        for _ in range(n_layer - 1):
            layers.append(nn.Linear(hid_dim, hid_dim))
            if use_bn: layers.append(nn.BatchNorm1d(hid_dim))
            layers += [activation, Drop(p=dropout)]
        layers.append(nn.Linear(hid_dim, out_dim))
        self.net = nn.Sequential(*layers)
        self.w_initial = w_initial

    def forward(self, x):
        return self.net(x)

ACT_DEFAULT_INIT = {
    "relu":      "he",
    "leakyrelu": "he",
    "sigmoid":   "xavier",
    "tanh":      "xavier",
}
ACT_ALT_INIT = ACT_DEFAULT_INIT

def init_weights(m: nn.Module, w_init: str):
    if isinstance(m, nn.Linear):
        scheme = (w_init or "xavier").lower()
        if scheme == "he":
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        elif scheme == "xavier":
            nn.init.xavier_normal_(m.weight)
        else:
            nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

# ============================================================
# [F] 학습 루프 공통
# ============================================================

def build_model(hp: Dict[str, Any]) -> nn.Module:
    model = MLP(IN_DIM, OUT_DIM, hp["units"], hp["layers"], hp["act"], hp["dropout"], True, hp["w_init"])
    model.apply(lambda mm: init_weights(mm, hp["w_init"]))
    return model.to(device)

def build_optimizer(model: nn.Module, hp: Dict[str, Any]):
    name, lr, wd = hp["optim"].lower(), hp["lr"], hp["weight_decay"]
    if   name == "adam":  return optim.Adam (model.parameters(), lr=lr, weight_decay=wd)
    elif name == "adamw": return optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    elif name == "sgd":   return optim.SGD  (model.parameters(), lr=lr, weight_decay=wd, momentum=0.9, nesterov=True)
    else: raise ValueError(f"Unknown optimizer: {name}")

class EarlyStopper:
    def __init__(self, patience=3, min_delta=1e-4):
        self.patience, self.min_delta = patience, min_delta
        self.best, self.bad_epochs = -float('inf'), 0
    def step(self, metric):
        if metric > self.best + self.min_delta:
            self.best, self.bad_epochs = metric, 0
        else:
            self.bad_epochs += 1
        return self.bad_epochs >= self.patience

# [CHANGED] run_train_eval: 오버피팅 감지 여부를 함께 반환(was_overfit)
def run_train_eval(hp: Dict[str, Any], max_epochs=GLOBAL_EPOCHS, of_conf=DEFAULT_OF_CONF):
    log(f"[train] start: L={hp['layers']} U={hp['units']} D={hp['dropout']:.3f} "
        f"act={hp['act']}/{hp['w_init']} opt={hp['optim']} lr={hp['lr']:.2e} wd={hp['weight_decay']:.2e} "
        f"| overfit_conf={of_conf}%")

    criterion = nn.CrossEntropyLoss()
    model = build_model(hp)
    optimizer = build_optimizer(model, hp)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

    stopper = EarlyStopper(patience=3, min_delta=1e-4)

    best_val_acc, best_state = -1.0, None
    hist = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    was_overfit = False  # [CHANGED]

    for epoch in range(1, max_epochs+1):
        # Train
        model.train()
        tot, corr, tot_loss = 0, 0, 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            tot_loss += loss.item() * yb.size(0)
            corr += (logits.argmax(1) == yb).sum().item()
            tot  += yb.size(0)
        train_loss, train_acc = tot_loss/max(1,tot), corr/max(1,tot)

        # Valid
        model.eval()
        with torch.no_grad():
            tot, corr, tot_loss = 0, 0, 0.0
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                tot_loss += loss.item() * yb.size(0)
                corr += (logits.argmax(1) == yb).sum().item()
                tot  += yb.size(0)
            val_loss, val_acc = tot_loss/max(1,tot), corr/max(1,tot)

        hist["train_loss"].append(train_loss); hist["val_loss"].append(val_loss)
        hist["train_acc"].append(train_acc);   hist["val_acc"].append(val_acc)

        scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {"model": copy.deepcopy(model.state_dict()),
                          "epoch": epoch, "val_acc": val_acc,
                          "val_loss": val_loss, "train_loss": train_loss, "train_acc": train_acc}

        # 에폭 로그
        log(f"  [epoch {epoch:02d}/{max_epochs:02d}] "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"train_acc={train_acc:.3f} val_acc={val_acc:.3f} (best={best_val_acc:.3f})")

        # 조기종료/과적합
        stop_patience = stopper.step(val_acc)
        stop_overfit  = False
        if len(hist["train_loss"]) >= 4:
            try:
                stop_overfit = overfit_checking(
                    hist["train_loss"], hist["val_loss"],
                    train_acc=hist["train_acc"], val_acc=hist["val_acc"],
                    conf_level=of_conf,
                    gap_mode="gated_or",         # 필요시 "and"/"z_fuse"로 변경 가능
                    zthr_abs_offset=0.0,         # abs-gap만 더 엄격히 하려면 +0.2~0.5
                    tail_k=3, use_ema=True, ema_alpha=0.6
                )
                if stop_overfit:
                    was_overfit = True
                    log("  [early] overfit_checking=True → stop (and EXCLUDE this candidate)")
            except Exception as e:
                log(f"  [warn] overfit_checking error: {e}")
                stop_overfit = False

        if stop_patience:
            log("  [early] patience reached → stop")
        if stop_patience or stop_overfit:
            break

    # [CHANGED] 오버피팅이면 점수를 제외(-inf 등가) 처리하도록 신호 반환
    if best_state is None:
        best_state = {"model": copy.deepcopy(model.state_dict()),
                      "epoch": 0, "val_acc": -1.0, "val_loss": float("inf"),
                      "train_loss": float("inf"), "train_acc": 0.0}
    log(f"[train] done: best_val_acc={best_val_acc:.4f} (epoch={best_state['epoch']}) | was_overfit={was_overfit}")
    return best_val_acc, best_state, hist, was_overfit  # [CHANGED: 4개 반환]

# ============================================================
# [G] PSO (옵티마이저+lr+wd 동시)
# ============================================================

class PSO:
    def __init__(self, n_particles=PSO_PARTICLES, n_iters=PSO_ITERS,
                 lr_bounds=PSO_LR_BOUNDS, wd_bounds=PSO_WD_BOUNDS,
                 optim_cand=OPTIM_CAND, seed=SEED,
                 improve_tol=PSO_IMPROVE_TOL, early_patience=PSO_EARLY_PATI,
                 vel_tol=PSO_VEL_TOL, div_tol=PSO_DIV_TOL,
                 target_val=PSO_TARGET_VAL, time_limit_sec=PSO_TIME_LIMIT,
                 inertia=PSO_INERTIA, c1=PSO_C1, c2=PSO_C2, inertia_decay=PSO_INERTIA_DEC,
                 kick_on_stall=PSO_KICK_ON, kick_scale=PSO_KICK_SCALE,
                 p_adopt_pbest=P_ADOPT_PBEST, p_adopt_gbest=P_ADOPT_GBEST, p_mutate=P_MUTATE_OPT):
        self.np = n_particles
        self.ni = n_iters
        self.lr_lo, self.lr_hi = lr_bounds
        self.wd_lo, self.wd_hi = wd_bounds
        self.optim_cand = list(optim_cand)
        self.improve_tol = improve_tol
        self.early_patience = early_patience
        self.vel_tol = vel_tol
        self.div_tol = div_tol
        self.target_val = target_val
        self.time_limit_sec = time_limit_sec
        self.w0, self.c1, self.c2 = inertia, c1, c2
        self.inertia_decay = inertia_decay
        self.kick_on_stall = kick_on_stall
        self.kick_scale = kick_scale
        self.p_adopt_pbest = p_adopt_pbest
        self.p_adopt_gbest  = p_adopt_gbest
        self.p_mutate       = p_mutate
        random.seed(seed); np.random.seed(seed)

    def _rand_log(self, lo, hi):
        a, b = math.log10(lo), math.log10(hi)
        return 10 ** np.random.uniform(a, b)

    def _rand_optim(self):
        return random.choice(self.optim_cand)

    def optimize(self, base_hp: Dict[str, Any], evaluator):
        t0 = time.perf_counter()
        log(f"[PSO-OPT] start base (L={base_hp['layers']} U={base_hp['units']} D={base_hp['dropout']:.3f} "
            f"act={base_hp['act']}/{base_hp['w_init']})")

        # 초기 스웜
        swarm = []
        for _ in range(self.np):
            p = {
                "optim": self._rand_optim(),
                "lr": self._rand_log(self.lr_lo, self.lr_hi),
                "weight_decay": self._rand_log(self.wd_lo, self.wd_hi),
                "v_lr": 0.0, "v_wd": 0.0
            }
            swarm.append(p)

        pbest = [None] * self.np
        pbest_val = [float("-inf")] * self.np

        gbest = None
        gbest_val = float("-inf")

        no_improve = 0
        last_gbest_val = float("-inf")

        for it in range(1, self.ni + 1):
            it_best = float("-inf")

            # 1) 평가
            for i, p in enumerate(swarm):
                hp = copy.deepcopy(base_hp)
                hp["optim"] = p["optim"]
                hp["lr"] = float(max(self.lr_lo, min(self.lr_hi, p["lr"])))
                hp["weight_decay"] = float(max(self.wd_lo, min(self.wd_hi, p["weight_decay"])))

                val, _state, _hist = evaluator(hp)  # 오버피팅이면 -inf가 들어옴
                if val > pbest_val[i]:
                    pbest_val[i] = val
                    pbest[i] = copy.deepcopy(p)
                if val > gbest_val:
                    gbest_val = val
                    gbest = copy.deepcopy(p)
                it_best = max(it_best, val)

            # 모든 개체가 -inf라면 즉시 중단(기본값 반환)
            if not np.isfinite(gbest_val):
                log("  [PSO-OPT] all candidates overfit in this iter → stop and fallback")
                elapsed = time.perf_counter() - t0
                fallback = {
                    "optim": base_hp.get("optim", "Adam"),
                    "lr": float(base_hp.get("lr", 1e-3)),
                    "weight_decay": float(base_hp.get("weight_decay", 1e-4)),
                }
                log(f"[PSO-OPT] done: gbest=-inf (no feasible). time={elapsed:.2f}s")
                return fallback, float("-inf"), elapsed

            # 2) 로그
            log(f"  [PSO-OPT] iter {it}/{self.ni}: iter_best={it_best:.4f} | "
                f"gbest={gbest_val:.4f} (opt={gbest['optim']}, lr={gbest['lr']:.2e}, wd={gbest['weight_decay']:.2e})")

            # 3) 조기종료 조건
            if (self.target_val is not None) and (gbest_val >= self.target_val):
                log("  [PSO-OPT] stop: target_val reached")
                break

            if gbest_val <= last_gbest_val + self.improve_tol:
                no_improve += 1
            else:
                no_improve = 0
                last_gbest_val = gbest_val

            # 속도/다양성 체크(로그 스케일)
            log_lrs = np.array([math.log10(max(self.lr_lo, min(self.lr_hi, p["lr"]))) for p in swarm])
            log_wds = np.array([math.log10(max(self.wd_lo, min(self.wd_hi, p["weight_decay"]))) for p in swarm])
            vel_small = all(abs(p["v_lr"]) < PSO_VEL_TOL and abs(p["v_wd"]) < PSO_VEL_TOL for p in swarm)
            diversity_small = (log_lrs.std() < PSO_DIV_TOL) and (log_wds.std() < PSO_DIV_TOL)

            if no_improve >= self.early_patience and (vel_small or diversity_small):
                log(f"  [PSO-OPT] stop: stagnation (no_improve={no_improve}, vel_small={vel_small}, div_small={diversity_small})")
                break

            # 4) 속도/위치 업데이트
            for i, p in enumerate(swarm):
                r1, r2 = np.random.rand(), np.random.rand()

                log_lr = math.log10(max(self.lr_lo, min(self.lr_hi, p["lr"])))
                log_wd = math.log10(max(self.wd_lo, min(self.wd_hi, p["weight_decay"])))

                pbest_eff = pbest[i] if pbest[i] is not None else p
                gbest_eff = gbest     if gbest     is not None else p

                pbest_log_lr = math.log10(max(self.lr_lo, min(self.lr_hi, pbest_eff["lr"])))
                pbest_log_wd = math.log10(max(self.wd_lo, min(self.wd_hi, pbest_eff["weight_decay"])))
                gbest_log_lr = math.log10(max(self.lr_lo, min(self.lr_hi, gbest_eff["lr"])))
                gbest_log_wd = math.log10(max(self.wd_lo, min(self.wd_hi, gbest_eff["weight_decay"])))

                p["v_lr"] = PSO_INERTIA * p["v_lr"] + PSO_C1 * r1 * (pbest_log_lr - log_lr) + PSO_C2 * r2 * (gbest_log_lr - log_lr)
                p["v_wd"] = PSO_INERTIA * p["v_wd"] + PSO_C1 * r1 * (pbest_log_wd - log_wd) + PSO_C2 * r2 * (gbest_log_wd - log_wd)

                if PSO_KICK_ON and no_improve >= max(1, PSO_EARLY_PATI // 1):
                    p["v_lr"] += np.random.randn() * PSO_KICK_SCALE
                    p["v_wd"] += np.random.randn() * PSO_KICK_SCALE

                new_log_lr = log_lr + p["v_lr"]
                new_log_wd = log_wd + p["v_wd"]
                p["lr"] = float(max(self.lr_lo, min(self.lr_hi, 10 ** new_log_lr)))
                p["weight_decay"] = float(max(self.wd_lo, min(self.wd_hi, 10 ** new_log_wd)))

                if np.random.rand() < P_ADOPT_PBEST and (pbest[i] is not None):
                    p["optim"] = pbest[i]["optim"]
                if np.random.rand() < P_ADOPT_GBEST and (gbest is not None):
                    p["optim"] = gbest["optim"]
                if np.random.rand() < P_MUTATE_OPT:
                    p["optim"] = self._rand_optim()

        elapsed = time.perf_counter() - t0
        log(f"[PSO-OPT] done: gbest={gbest_val:.4f} (opt={gbest['optim']}, lr={gbest['lr']:.2e}, wd={gbest['weight_decay']:.2e}) | time={elapsed:.2f}s")
        return gbest, gbest_val, elapsed

# ============================================================
# [H] SIH 유틸 & 프록시 평가 (개선 정체 종료 포함)
# ============================================================

def _clip_int(v, lo, hi):
    return int(max(lo, min(hi, int(round(v)))))

def _clip_float(v, lo, hi):
    return float(max(lo, min(hi, float(v))))

def _mid(a, b):
    return (a + b) / 2.0

def _int_left_right_mids(lo: int, hi: int):
    c  = (lo + hi) / 2.0
    left  = max(lo,  int((lo + c) // 2))
    right = min(hi,  int(math.ceil((c + hi) / 2)))
    if left == right and hi - lo >= 2:
        left  = int(math.floor((lo + c) / 2))
        right = int(math.ceil((c + hi) / 2))
    return left, right

def _make_point(bounds, kind="center"):
    (Llo, Lhi), (Ulo, Uhi), (Dlo, Dhi) = bounds["L"], bounds["U"], bounds["D"]
    if kind == "center":
        L = _clip_int(_mid(Llo, Lhi), Llo, Lhi)
        U = _clip_int(_mid(Ulo, Uhi), Ulo, Uhi)
        D = _clip_float(_mid(Dlo, Dhi), Dlo, Dhi)
    elif kind == "random":
        L = _clip_int(np.random.uniform(Llo, Lhi), Llo, Lhi)
        U = _clip_int(np.random.uniform(Ulo, Uhi), Ulo, Uhi)
        D = _clip_float(np.random.uniform(Dlo, Dhi), Dlo, Dhi)
    else:
        raise ValueError("unknown kind")
    return {"layers": L, "units": U, "dropout": D}

def _neighbors_1d(bounds, base_point, dim):
    (Llo, Lhi), (Ulo, Uhi), (Dlo, Dhi) = bounds["L"], bounds["U"], bounds["D"]
    pts = []
    if dim == "layers":
        left, right = _int_left_right_mids(Llo, Lhi)
        lp = dict(base_point); lp["layers"] = _clip_int(left, Llo, Lhi);  pts.append(lp)
        rp = dict(base_point); rp["layers"] = _clip_int(right, Llo, Lhi); pts.append(rp)
    elif dim == "units":
        left, right = _int_left_right_mids(Ulo, Uhi)
        lp = dict(base_point); lp["units"] = _clip_int(left, Ulo, Uhi);   pts.append(lp)
        rp = dict(base_point); rp["units"] = _clip_int(right, Ulo, Uhi);  pts.append(rp)
    elif dim == "dropout":
        Dlo_mid = _mid(Dlo, _mid(Dlo, Dhi))
        Dhi_mid = _mid(_mid(Dlo, Dhi), Dhi)
        lp = dict(base_point); lp["dropout"] = _clip_float(Dlo_mid, Dlo, Dhi); pts.append(lp)
        rp = dict(base_point); rp["dropout"] = _clip_float(Dhi_mid, Dlo, Dhi); pts.append(rp)
    else:
        raise ValueError("unknown dim")
    return pts

# ---- SIH 프록시 랜덤 샘플링 유틸 ----
def _sample_act_init():
    act = random.choice(ACTIVATION_CAND)
    w_init = ACT_DEFAULT_INIT[act]  # 규칙대로 자동 선택
    return act, w_init

def _sample_optimizer():
    return random.choice(OPTIM_CAND)

# ---- 프록시 HP 생성 (명시 인자) ----
def _proxy_hp(layers: int, units: int, dropout: float,
              act="relu", w_init=None, optim="Adam",
              lr=1e-3, weight_decay=1e-4):
    if w_init is None:
        w_init = ACT_DEFAULT_INIT.get(act, "xavier")
    return dict(
        layers=int(layers),
        units=int(units),
        dropout=float(np.clip(dropout, DROPOUT_MIN, DROPOUT_MAX)),
        act=act,
        w_init=w_init,
        optim=optim,
        lr=float(lr),
        weight_decay=float(weight_decay),
    )

# [CHANGED] proxy_eval: SIH에서 (L,U,D) 평가 시 랜덤 act/init/optimizer 사용
PROXY_FIXED_LR      = 1e-3
PROXY_FIXED_WD      = 1e-4

def proxy_eval(layers: int, units: int, dropout: float) -> float:
    if not PROXY_RAND_ACTOPT:
        hp = _proxy_hp(layers, units, dropout, act="relu",
                       w_init=ACT_DEFAULT_INIT["relu"],
                       optim="Adam",
                       lr=PROXY_FIXED_LR, weight_decay=PROXY_FIXED_WD)
        val, _state, _hist, was_overfit = run_train_eval(hp, max_epochs=GLOBAL_EPOCHS, of_conf=DEFAULT_OF_CONF)
        if was_overfit:
            log("  [proxy_eval] overfit=True → score = -inf (excluded)")
            return float("-inf")
        return float(val)

    best = float("-inf")
    if PROXY_SEEDED_BY_CFG:
        seed_val = (int(layers)*73856093) ^ (int(units)*19349663) ^ (int(dropout*1e6)*83492791) ^ SEED
        rnd_state = random.getstate(); np_state = np.random.getstate()
        random.seed(seed_val); np.random.seed(seed_val & 0xffffffff)

    for t in range(PROXY_REPEATS):
        act, w_init = _sample_act_init()
        optim_name  = _sample_optimizer()
        hp = _proxy_hp(layers, units, dropout,
                       act=act, w_init=w_init, optim=optim_name,
                       lr=PROXY_FIXED_LR, weight_decay=PROXY_FIXED_WD)
        val, _state, _hist, was_overfit = run_train_eval(hp, max_epochs=GLOBAL_EPOCHS, of_conf=DEFAULT_OF_CONF)
        log(f"  [proxy_eval/RAND] try={t+1}/{PROXY_REPEATS} | act={act}/{w_init}, opt={optim_name} → "
            f"val={('-inf' if was_overfit else f'{val:.4f}')}")
        if was_overfit:
            val = float("-inf")
        best = max(best, float(val))

    if PROXY_SEEDED_BY_CFG:
        random.setstate(rnd_state); np.random.setstate(np_state)

    if not np.isfinite(best):
        log("  [proxy_eval/RAND] all tries overfit → score = -inf")
    return best

def sih_three_models():
    t0 = time.perf_counter()
    bounds = {"L": (LAYER_MIN, LAYER_MAX),
              "U": (UNITS_MIN, UNITS_MAX),
              "D": (DROPOUT_MIN, DROPOUT_MAX)}
    best = {"cfg": None, "acc": float("-inf")}
    gbest_val = float("-inf")
    no_improve = 0

    for rnd in range(1, SIH_ROUNDS+1):
        log(f"[SIH] use_random_act/opt={PROXY_RAND_ACTOPT}, repeats={PROXY_REPEATS}")
        center = _make_point(bounds, "center")
        cands = [center]
        for dim in ["layers", "units", "dropout"]:
            cands += _neighbors_1d(bounds, center, dim)
        for _ in range(SIH_RAND_PER_ROUND):
            cands.append(_make_point(bounds, "random"))

        scored = []
        for cfgp in cands:
            acc = proxy_eval(cfgp["layers"], cfgp["units"], cfgp["dropout"])
            scored.append((acc, cfgp))
            if acc > best["acc"]:
                best = {"cfg": cfgp, "acc": acc}

        valid = [(a,c) for (a,c) in scored if np.isfinite(a)]
        valid.sort(key=lambda x: x[0], reverse=True)
        top = valid[:min(SIH_TOP_K, len(valid))]

        log(f"\n[SIH] round {rnd}/{SIH_ROUNDS} | bounds L={bounds['L']} U={bounds['U']} D={bounds['D']}")
        if top:
            for i,(acc,cfgp) in enumerate(top):
                log(f"  top{i+1}: acc={acc:.4f}, cfg={cfgp}")
        else:
            log("  [warn] all candidates overfit in this round. widening randomness next round?")

        round_best = top[0][0] if top else float("-inf")
        if gbest_val == float("-inf"):
            gbest_val = round_best
            no_improve = 0
        else:
            if not np.isfinite(round_best) or round_best <= gbest_val + SIH_IMPROVE_TOL:
                no_improve += 1
            else:
                no_improve = 0
                gbest_val = round_best

        log(f"[SIH] gbest_val={gbest_val:.4f}, round_best={round_best:.4f}, no_improve={no_improve}")

        if no_improve >= SIH_NO_IMPROVE_PATIENCE:
            log(f"[SIH] early stop: stagnation for {no_improve} rounds (tol={SIH_IMPROVE_TOL})")
            break

        if not top:
            continue

        Llo, Lhi = bounds["L"]; Ulo, Uhi = bounds["U"]; Dlo, Dhi = bounds["D"]
        # layers
        Lpivot = int(np.median([t[1]["layers"] for t in top]))
        Lc = int(round((Llo + Lhi) / 2.0))
        if Lpivot >= Lc:
            Llo = _clip_int((Lc + Lhi)/2.0, LAYER_MIN, LAYER_MAX)
        else:
            Lhi = _clip_int((Llo + Lc)/2.0, LAYER_MIN, LAYER_MAX)
        if Lhi < Llo: Llo, Lhi = Lhi, Llo
        # units
        Upivot = int(np.median([t[1]["units"] for t in top]))
        Uc = int(round((Ulo + Uhi) / 2.0))
        if Upivot >= Uc:
            Ulo = _clip_int((Uc + Uhi)/2.0, UNITS_MIN, UNITS_MAX)
        else:
            Uhi = _clip_int((Ulo + Uc)/2.0, UNITS_MIN, UNITS_MAX)
        if Uhi < Ulo: Ulo, Uhi = Uhi, Ulo
        # dropout
        Dpivot = float(np.median([t[1]["dropout"] for t in top]))
        Dc = (Dlo + Dhi) / 2.0
        if Dpivot >= Dc:
            Dlo = _clip_float((Dc + Dhi)/2.0, DROPOUT_MIN, DROPOUT_MAX)
        else:
            Dhi = _clip_float((Dlo + Dc)/2.0, DROPOUT_MIN, DROPOUT_MAX)
        if Dhi < Dlo: Dlo, Dhi = Dhi, Dlo

        bounds["L"] = (Llo, Lhi); bounds["U"] = (Ulo, Uhi); bounds["D"] = (Dlo, Dhi)

    if best["cfg"] is None or not np.isfinite(best["acc"]):
        fallback = _make_point({"L":(LAYER_MIN,LAYER_MAX),"U":(UNITS_MIN,UNITS_MAX),"D":(DROPOUT_MIN,DROPOUT_MAX)}, "center")
        L_star, U_star, D_star = fallback["layers"], fallback["units"], fallback["dropout"]
    else:
        L_star = best["cfg"]["layers"]
        U_star = best["cfg"]["units"]
        D_star = best["cfg"]["dropout"]

    L_prev = _clip_int(L_star - L_NEIGHBOR_DELTA, LAYER_MIN, LAYER_MAX)
    L_next = _clip_int(L_star + L_NEIGHBOR_DELTA, LAYER_MIN, LAYER_MAX)
    U_prev = _clip_int(U_star - U_NEIGHBOR_DELTA, UNITS_MIN, UNITS_MAX)
    U_next = _clip_int(U_star + U_NEIGHBOR_DELTA, UNITS_MIN, UNITS_MAX)
    D_prev = _clip_float(D_star - D_NEIGHBOR_DELTA, DROPOUT_MIN, DROPOUT_MAX)
    D_next = _clip_float(D_star + D_NEIGHBOR_DELTA, DROPOUT_MIN, DROPOUT_MAX)

    model1 = _proxy_hp(L_prev, U_prev, D_prev, act="relu", w_init=ACT_DEFAULT_INIT["relu"])
    model2 = _proxy_hp(L_star, U_star, D_star, act="relu", w_init=ACT_DEFAULT_INIT["relu"])
    model3 = _proxy_hp(L_next, U_next, D_next, act="relu", w_init=ACT_DEFAULT_INIT["relu"])

    elapsed = time.perf_counter() - t0
    log(f"\n[SIH] done in {elapsed:.2f}s")
    log(f"[SIH] Star:  L={L_star} U={U_star} D={D_star:.3f}")
    log(f"[SIH] Prev:  L={L_prev} U={U_prev} D={D_prev:.3f}")
    log(f"[SIH] Next:  L={L_next} U={U_next} D={D_next:.3f}")

    stage1_hp = {"L*": L_star, "U*": U_star, "D*": round(D_star, 3)}
    banner("【STAGE 1】 SIH 전역 축소 → 최종 추정점 및 3개 베이스 생성")
    log_kv("〈Star (center)〉", model2, width_key=16)
    log_kv("〈Prev〉", model1, width_key=16)
    log_kv("〈Next〉", model3, width_key=16)
    log_kv("〈Star Summary〉", stage1_hp, width_key=16)
    log("="*80)

    return [model1, model2, model3], elapsed

def act_init_candidates():
    cands = []
    for a in ACTIVATION_CAND:
        cands.append((a, ACT_DEFAULT_INIT[a]))
        if USE_ALT_INIT:
            cands.append((a, ACT_ALT_INIT[a]))
    return cands

# ============================================================
# [I] 파이프라인: SIH → 3개 베이스 → 각 베이스 정밀화
# ============================================================

def exhaustive_act_init(base_hp: Dict[str, Any]) -> Tuple[Dict[str, Any], float, float]:
    t0 = time.perf_counter()
    best_hp, best_val = None, float("-inf")
    log(f"\n[AI] Exhaustive (activation/init) for base: L={base_hp['layers']} U={base_hp['units']} D={base_hp['dropout']:.3f}")
    for act, w_init in act_init_candidates():
        hp = copy.deepcopy(base_hp)
        hp.update({"act":act, "w_init":w_init})
        val, _s, _h, was_overfit = run_train_eval(hp, max_epochs=GLOBAL_EPOCHS, of_conf=DEFAULT_OF_CONF)
        if was_overfit:
            log(f"  [AI] act={act:>9s} / init={w_init:<6s} → overfit=True (excluded)")
            continue
        log(f"  [AI] act={act:>9s} / init={w_init:<6s} → val={val:.4f}")
        if val > best_val:
            best_val, best_hp = val, copy.deepcopy(hp)
    elapsed = time.perf_counter() - t0
    if best_hp is None:
        best_hp = copy.deepcopy(base_hp)
        log(f"[AI] all overfit → keep base (will likely be excluded). time={elapsed:.2f}s")
    else:
        log(f"[AI] best: act={best_hp['act']}/{best_hp['w_init']} → val={best_val:.4f} | time={elapsed:.2f}s")

    log_hp_table("【STAGE 2】 (Base) 활성화·초기화 전수 결과 - 선택 조합", best_hp)
    return best_hp, best_val, elapsed

def optimizer_pso_joint(best_ai_hp: Dict[str, Any]) -> Tuple[Dict[str, Any], float, float]:
    def evaluator(hp_eval):
        hp_use = copy.deepcopy(best_ai_hp)
        hp_use.update({"optim":hp_eval["optim"], "lr":hp_eval["lr"], "weight_decay":hp_eval["weight_decay"]})
        val, _s, _h, was_overfit = run_train_eval(hp_use, max_epochs=GLOBAL_EPOCHS, of_conf=DEFAULT_OF_CONF)
        if was_overfit:
            log("  [OPT+PSO] evaluator overfit=True → score=-inf")
            return float("-inf"), None, None
        return val, None, None

    pso = PSO()
    base = copy.deepcopy(best_ai_hp)
    base.update({"optim":"Adam", "lr":1e-3, "weight_decay":1e-4})
    gbest, gval, t_pso = pso.optimize(base, evaluator)

    final_hp = copy.deepcopy(best_ai_hp)
    if np.isfinite(gval):
        final_hp.update({"optim":gbest["optim"], "lr":gbest["lr"], "weight_decay":gbest["weight_decay"]})
    else:
        log("[OPT+PSO] all PSO candidates overfit → keep AI-best hyperparams")

    val, _s, _h, was_overfit = run_train_eval(final_hp, max_epochs=GLOBAL_EPOCHS, of_conf=DEFAULT_OF_CONF)
    if was_overfit:
        log(f"[OPT+PSO] final overfit=True → val=-inf")
        val = float("-inf")
    else:
        log(f"[OPT+PSO] final: opt={final_hp['optim']}, lr={final_hp['lr']:.2e}, wd={final_hp['weight_decay']:.2e} → val={val:.4f}")

    log_hp_table("【STAGE 3】 (Base) PSO(optim+lr+wd) 결과 - 선택 조합", final_hp)
    return final_hp, val, t_pso

def pipeline_search_once() -> Tuple[Dict[str, Any], float, Dict[str, list], Dict[str, Any], Dict[str, float]]:
    t_round0 = time.perf_counter()
    bases, t_sih = sih_three_models()
    for i, b in enumerate(bases, 1):
        log(f"[SNAPSHOT] Base-{i} after SIH: {b}")
    timings = {"sih": t_sih}

    candidates = []
    for idx, base_hp in enumerate(bases, start=1):
        banner(f"【STAGE 2/3】 Base-{idx} 정밀화 시작")
        best_ai_hp, _val_ai, t_ai = exhaustive_act_init(base_hp)
        log(f"[SNAPSHOT] Base-{idx} after AI: {best_ai_hp}")
        timings[f"ai_base{idx}"] = t_ai

        best_full_hp, best_full_val, t_pso = optimizer_pso_joint(best_ai_hp)
        log(f"[SNAPSHOT] Base-{idx} after PSO: {best_full_hp}")
        timings[f"pso_base{idx}"] = t_pso

        if np.isfinite(best_full_val):
            log(f"[Stage] Base-{idx} result: val={best_full_val:.4f} | "
                f"L={best_full_hp['layers']} U={best_full_hp['units']} D={best_full_hp['dropout']:.3f} "
                f"act={best_full_hp['act']}/{best_full_hp['w_init']} "
                f"opt={best_full_hp['optim']} lr={best_full_hp['lr']:.2e} wd={best_full_hp['weight_decay']:.2e}")
            candidates.append((best_full_val, best_full_hp))
        else:
            log(f"[Stage] Base-{idx} result: overfit → excluded")
        log("="*80)

    if not candidates:
        log("[FINAL-SELECT] all candidates excluded by overfitting → pick first base as fallback")
        candidates = [(float("-inf"), bases[0])]

    candidates.sort(key=lambda x: x[0], reverse=True)
    final_val, final_hp = candidates[0]

    log_hp_table("【STAGE 4】 세 베이스 중 최종 선택(VAL 최고)", final_hp)

    t_final0 = time.perf_counter()
    val2, state2, hist2, _over_final = run_train_eval(final_hp, max_epochs=FINAL_EPOCHS, of_conf=DEFAULT_OF_CONF)
    t_final = time.perf_counter() - t_final0
    timings["final_train"] = t_final

    log("\n[HISTORY] Final model epoch-wise metrics:")
    for ep, (trL, vaL, trA, vaA) in enumerate(zip(hist2["train_loss"], hist2["val_loss"], hist2["train_acc"], hist2["val_acc"]), 1):
        log(f"  ep={ep:02d} | tr_loss={trL:.4f} va_loss={vaL:.4f} tr_acc={trA:.4f} va_acc={vaA:.4f}")

    t_round = time.perf_counter() - t_round0
    timings["round_total"] = t_round

    log(
        f"\n[TIMINGS] sih={timings.get('sih', 0.0):.2f}s, "
        f"AI(b1/b2/b3)=[{timings.get('ai_base1',0.0):.2f}, {timings.get('ai_base2',0.0):.2f}, {timings.get('ai_base3',0.0):.2f}]s, "
        f"PSO(b1/b2/b3)=[{timings.get('pso_base1',0.0):.2f}, {timings.get('pso_base2',0.0):.2f}, {timings.get('pso_base3',0.0):.2f}]s, "
        f"final_train={timings.get('final_train', 0.0):.2f}s, total={timings.get('round_total', 0.0):.2f}s"
    )

    return final_hp, val2, hist2, state2, timings

# ============================================================
# [J] 출력/시각화
# ============================================================

def print_hyperparams(title: str, hp: Dict[str, Any]):
    msg_lines = [f"\n===== {title} ====="]
    for k in sorted(hp.keys()):
        msg_lines.append(f"{k:>16s} : {hp[k]}")
    msg = "\n".join(msg_lines)
    print(msg)
    _writeln(msg)

def plot_curves(hist: Dict[str, list], test_acc: float = None,
                save_png=SAVE_LOSS_PNG, save_svg=SAVE_LOSS_SVG):
    epochs = range(1, len(hist["train_loss"])+1)
    # Loss
    plt.figure(figsize=(7,5))
    plt.plot(epochs, hist["train_loss"], label="Train Loss")
    plt.plot(epochs, hist["val_loss"],   label="Val Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Loss Curves")
    plt.legend(); plt.grid(True, linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(save_png, dpi=400); plt.savefig(save_svg)
    # Acc
    plt.figure(figsize=(7,5))
    plt.plot(epochs, hist["train_acc"], label="Train Acc")
    plt.plot(epochs, hist["val_acc"],   label="Val Acc")
    if test_acc is not None:
        plt.scatter([epochs[-1]], [test_acc], marker="o", s=60, label=f"Test Acc ({test_acc:.3f})")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.title("Accuracy Curves")
    plt.legend(); plt.grid(True, linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(SAVE_ACC_PNG, dpi=400); plt.savefig(SAVE_ACC_SVG)
    print("END")

# ============================================================
# [K] 메인
# ============================================================

if __name__ == "__main__":
    _init_log()
    log(f"[info] device = {device}")
    log("[info] downloading CIFAR-10 ...")

    # 데이터 준비 시간 측정
    t_data0 = time.perf_counter()
    train_full = torchvision.datasets.CIFAR10(root=DATA_ROOT, train=True,  download=True, transform=transform)
    test_set   = torchvision.datasets.CIFAR10(root=DATA_ROOT, train=False, download=True, transform=transform)
    val_size   = int(len(train_full) * VAL_RATIO)
    train_size = len(train_full) - val_size
    train_set, val_set = random_split(train_full, [train_size, val_size],
                                      generator=torch.Generator().manual_seed(SEED))
    train_loader = DataLoader(train_set, batch_size=BATCH_TRAIN, shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_set,   batch_size=BATCH_TEST,  shuffle=False, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_set,  batch_size=BATCH_TEST,  shuffle=False, num_workers=2, pin_memory=True)
    t_data = time.perf_counter() - t_data0
    log(f"[TIMING] data_load={t_data:.2f}s")

    t0 = time.perf_counter()

    # 전체 파이프라인 (1 라운드)
    final_hp, final_val, hist2, state2, timings = pipeline_search_once()

    # Test 평가
    model = build_model(final_hp)
    model.load_state_dict(state2["model"])
    model.to(device).eval()
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        tot, corr, tot_loss = 0, 0, 0.0
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            tot_loss += loss.item() * yb.size(0)
            corr += (logits.argmax(1) == yb).sum().item()
            tot  += yb.size(0)
        test_acc = corr / max(1, tot)
        test_loss = tot_loss / max(1, tot)

    elapsed = time.perf_counter() - t0
    summary = (f"\n[RESULT] Final(val@{FINAL_EPOCHS})={final_val:.4f} "
               f"| Test Acc={test_acc:.4f} (loss={test_loss:.4f}) "
               f"| round_time={elapsed/60:.1f} min")
    print(summary)
    _writeln(summary)

    banner("【STAGE 5】 최종 학습 & 테스트 성능")
    log_kv("〈Test Result〉", {"test_acc": round(test_acc, 4), "test_loss": round(test_loss, 4)}, width_key=16)
    log_hp_table("【FINAL SELECTED HYPERPARAMETERS】", final_hp)

    _writeln("\n[TIMINGS SUMMARY]")
    _writeln(f"data_load        : {t_data:.2f}s")
    _writeln(f"sih              : {timings.get('sih', 0.0):.2f}s")
    _writeln(f"ai_base1         : {timings.get('ai_base1', 0.0):.2f}s")
    _writeln(f"ai_base2         : {timings.get('ai_base2', 0.0):.2f}s")
    _writeln(f"ai_base3         : {timings.get('ai_base3', 0.0):.2f}s")
    _writeln(f"pso_base1        : {timings.get('pso_base1', 0.0):.2f}s")
    _writeln(f"pso_base2        : {timings.get('pso_base2', 0.0):.2f}s")
    _writeln(f"pso_base3        : {timings.get('pso_base3', 0.0):.2f}s")
    _writeln(f"final_train      : {timings.get('final_train', 0.0):.2f}s")
    _writeln(f"round_total      : {timings.get('round_total', 0.0):.2f}s")

    plot_curves(hist2, test_acc=test_acc, save_png=SAVE_LOSS_PNG, save_svg=SAVE_LOSS_SVG)
