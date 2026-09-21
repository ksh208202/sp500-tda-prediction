# ==========================================================================
# [팀 공통] TDA 결합 실험 공통 코드  v2.1
#   출처: 금융_TDA_김소현_v2.ipynb §8 (셀 36~40) — 배선은 그대로, 모델 자리만 비움
#   변경점: ⑤변동성 대조군 / ⑥저중복 TDA / test 평가 / 스케일링·시드·회귀 지원
#
# ▶ evaluate() 는 이 코드의 정의를 쓰세요 (건너뛰지 말 것)
#   전처리 §14와 결과 컬럼이 완전히 같고, task 인자 하나만 추가된 버전입니다.
#   분류(소진·서연·소현)는 §14와 소수점까지 동일한 값이 나옵니다.
#   다영은 task="reg" 를 넘겨야 예측수익률을 0 기준으로 자릅니다 (run_matrix가 자동 처리).
#   ※ 평가 함수가 두 개 돌면 결과표를 나란히 못 붙이니, 자기 노트북의 옛 정의는
#     이 셀 아래에서 덮어쓰이도록 순서만 지켜 주세요.
# ==========================================================================

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, balanced_accuracy_score

# ==========================================================================
# 0. 준비 — 소현이 배포하는 파일 3개를 같은 폴더(또는 드라이브)에 두기
#
#    GSPC_arrays_20050101_20260101.npz  ← ★ 각자 전처리를 다시 돌리지 말고 이 파일을 쓰세요
#    tda_features_w50.csv               ← TDA 피처 8종
#    cross_index_features.csv           ← 교차지수 피처 12종
#
#    ★ 이유: 서연·다영 노트북은 END_DATE=2025-01-01 이라 test 구간이
#      2022-01-12~2024-12-31(745행)이고, 소진은 2026-01-01 이라 2022-11-14~
#      2025-12-30(783행)입니다. 서로 다른 구간이라 숫자를 나란히 못 놓습니다.
#      배포된 npz(END_DATE=2026-01-01)를 전원이 그대로 로드하면 이 문제가 사라집니다.
#    ★ 폴더에 기간 없는 옛 GSPC_arrays.npz 가 있으면 지우거나 이름을 바꿔 주세요.
# ==========================================================================
NPZ_PATH   = "GSPC_arrays_20050101_20260101.npz"
TDA_PATH   = "tda_features_w50.csv"
CROSS_PATH = "cross_index_features.csv"

# ==========================================================================
# 1. 로드 (TDA 노트북 §8과 동일)
# ==========================================================================
d = np.load(NPZ_PATH, allow_pickle=True)
feat_cols = list(d["feature_columns"])

def restore(split):
    idx = pd.to_datetime(d[f"idx_{split}"])
    X = pd.DataFrame(d[f"X_{split}"], index=idx, columns=feat_cols)
    y = pd.Series(d[f"y_{split}_cls"], index=idx, name="y_updown").astype(int)
    yr = pd.Series(d[f"y_{split}_reg"], index=idx, name="y_reg")   # 회귀용(다영)
    return X, y, yr

X_train, y_train, yr_train = restore("train")
X_valid, y_valid, yr_valid = restore("valid")
X_test,  y_test,  yr_test  = restore("test")
print("npz 로드:", X_train.shape, X_valid.shape, X_test.shape, "| meta:", list(d["meta"]))
assert list(d["meta"])[2] == "2026-01-01", "END_DATE가 2026-01-01이 아닙니다 — 배포본을 쓰세요"

tda_features   = pd.read_csv(TDA_PATH,   encoding="utf-8-sig", index_col="Date", parse_dates=True)
cross_features = pd.read_csv(CROSS_PATH, encoding="utf-8-sig", index_col="Date", parse_dates=True)
tda_cols   = list(tda_features.columns)     # TDA_L1_H0 … TDA_L1_H1_std20 (8개)
cross_cols = list(cross_features.columns)   # X_exc1d_DJI … X_lead_RUT_1d (12개)
print("TDA", len(tda_cols), "| 교차지수", len(cross_cols))

# ==========================================================================
# 2. 팀 공통 evaluate() — 전처리 §14 + task 인자
#    임계값을 값 범위로 추측하지 않고 task 로 선언받습니다
# ==========================================================================
def evaluate(y_true_cls, score, name="model", task="clf",
             y_true_reg=None, pred_reg=None):
    score = np.asarray(score, dtype=float)
    y_true_cls = np.asarray(y_true_cls, dtype=int)
    thr = 0.5 if task == "clf" else 0.0
    y_pred = (score > thr).astype(int)
    out = {"experiment": name,
           "accuracy": accuracy_score(y_true_cls, y_pred),
           "f1": f1_score(y_true_cls, y_pred, zero_division=0),
           "roc_auc": roc_auc_score(y_true_cls, score),
           "n": len(y_true_cls)}
    if y_true_reg is not None and pred_reg is not None:
        from sklearn.metrics import mean_squared_error, r2_score
        out["rmse"] = float(np.sqrt(mean_squared_error(y_true_reg, pred_reg)))
        out["r2"] = float(r2_score(y_true_reg, pred_reg))
    return out

print("evaluate 정의 완료")

# ==========================================================================
# 3. 병합 — 교차·TDA를 먼저 합친 뒤 join → 모든 실험이 '동일 표본'
# ==========================================================================
extra = cross_features.join(tda_features, how="inner")

def attach(X, y, yr):
    Xt = X.join(extra, how="inner").dropna(subset=cross_cols + tda_cols)
    return Xt, y.loc[Xt.index], yr.loc[Xt.index]

X_train_t, y_train_t, yr_train_t = attach(X_train, y_train, yr_train)
X_valid_t, y_valid_t, yr_valid_t = attach(X_valid, y_valid, yr_valid)
X_test_t,  y_test_t,  yr_test_t  = attach(X_test,  y_test,  yr_test)

print(f"train {len(X_train)} → {len(X_train_t)} (워밍업 제거, 앞부분만)")
print(f"valid {len(X_valid)} → {len(X_valid_t)} | test {len(X_test)} → {len(X_test_t)}")
assert len(X_valid_t) == len(X_valid) and len(X_test_t) == len(X_test), \
    "valid/test 행이 줄었음 — 날짜 정합 확인 필요!"

# ==========================================================================
# 4. baseline 기준 — ★ 0.5 아님. train 다수 클래스를 항상 찍는 모델의 정확도
#    (사전적 기준: 학습 시점에 알 수 있는 정보만 사용)
#    현재 데이터에서는 valid 0.5409 / test 0.5568 이 나옵니다.
# ==========================================================================
maj = int(round(y_train_t.mean()))
BASE_ACC = {"valid": float((y_valid_t == maj).mean()),
            "test":  float((y_test_t == maj).mean())}
print("다수 클래스 baseline:", {k: round(v, 4) for k, v in BASE_ACC.items()})

# ==========================================================================
# 5. 실험 정의 ①~⑥
#    ⑤ 변동성만 : TDA norm 이 기존 변동성 피처와 같은 걸 재는지 가르는 대조군
#    ⑥ 저중복 TDA : TDA 8개 중 Volatility_20 과 |상관| 0.5 미만인 5개만
#       (소현 train 기준 — L1_H0 0.81 / L2_H0 0.81 / L1_H1_std20 0.64 는 제외)
#       "④가 ⑤보다 낮은 건 중복 피처가 신호를 희석했기 때문"이라는 가설 검증용
# ==========================================================================
vol_cols = [c for c in feat_cols if c.startswith("Volatility_")]
lowcorr_tda = ["TDA_L1_H1", "TDA_L2_H1", "TDA_PE_H1", "TDA_PE_H0", "TDA_L1_H1_diff1"]

EXPERIMENTS = {
    "① 베이스라인 (22)":    feat_cols,
    "② +교차지수 (22+12)":  feat_cols + cross_cols,
    "③ +TDA (22+8)":       feat_cols + tda_cols,
    "②+③ 전부 (22+20)":    feat_cols + cross_cols + tda_cols,
    "④ TDA만 (8)":         tda_cols,
    "④' 교차지수만 (12)":   cross_cols,
    "⑤ 변동성만 (4)":       vol_cols,          # ← ④와 짝지어 보기
    "⑥ 저중복 TDA (5)":     lowcorr_tda,       # ← ④와 짝지어 보기
}

# 사전 진단: TDA 피처가 기존 변동성과 얼마나 겹치는가
_diag = pd.concat([X_train_t[tda_cols], X_train_t[vol_cols]], axis=1).corr()
print("\nTDA × Volatility_20 상관:")
print(_diag.loc[tda_cols, "Volatility_20"].round(3).to_string())
print("→ 0.8 이상이면 그 피처는 사실상 변동성의 재탕. ⑥은 이 값이 낮은 것만 모은 실험입니다.")

# ==========================================================================
# 6. 실험 매트릭스 실행 — ★ 이 셀은 그대로 두세요 (네 명 공용)
#    make_model 은 "모델 만드는 함수를 받을 빈 자리"입니다. 실제 모델은
#    §8에서 run_matrix(xgb, ...) 처럼 넘기면 그때 채워집니다.
# ==========================================================================
from sklearn.preprocessing import StandardScaler

def run_matrix(make_model, name="model", split="valid",
               use_scaled=False, task="clf", seeds=(42,)):
    """
    make_model : 시드를 받아 모델을 돌려주는 함수. 예) lambda s: XGBClassifier(random_state=s)
                 → 함수 정의를 고치는 게 아니라, 호출할 때 넘기는 값입니다.
    split : "valid" (기본, 튜닝·비교용) / "test" (최종 1회만)
    task  : "clf" 분류 / "reg" 회귀(Ridge·Lasso — 예측수익률 부호로 방향 판정)
    seeds : 여러 개 주면 예측값 평균 (Bagging처럼 시드 변동이 큰 모델용)
    """
    Xte_t = {"valid": X_valid_t, "test": X_test_t}[split]
    yte   = {"valid": y_valid_t, "test": y_test_t}[split]
    yr_te = {"valid": yr_valid_t, "test": yr_test_t}[split]
    ytr = yr_train_t if task == "reg" else y_train_t

    rows, probs = [], {}
    for exp_name, cols in EXPERIMENTS.items():
        Xtr, Xte = X_train_t[cols], Xte_t[cols]
        if use_scaled:
            sc = StandardScaler().fit(Xtr)          # ★ train 에만 fit
            Xtr = pd.DataFrame(sc.transform(Xtr), index=Xtr.index, columns=cols)
            Xte = pd.DataFrame(sc.transform(Xte), index=Xte.index, columns=cols)

        preds, seed_accs = [], []
        for s in seeds:
            m = make_model(s)
            m.fit(Xtr, ytr)
            if task == "clf" and hasattr(m, "predict_proba"):
                q = m.predict_proba(Xte)[:, 1]
            else:
                q = m.predict(Xte)
            preds.append(q)
            thr_q = 0.5 if task == "clf" else 0.0
            seed_accs.append(accuracy_score(yte, (q > thr_q).astype(int)))
        p = np.mean(preds, axis=0)
        probs[exp_name] = p

        if task == "reg":
            res = evaluate(yte, p, name=exp_name, task=task,
                           y_true_reg=yr_te, pred_reg=p)
        else:
            res = evaluate(yte, p, name=exp_name, task=task)

        # balanced_acc — 0.5 근처면 '한쪽 클래스로만 찍는 중'이라는 뜻.
        #   임계값 기반 지표(accuracy·f1·gap_pp)가 정보를 못 담는 상황을 즉시 알려줌.
        thr_p = 0.5 if task == "clf" else 0.0
        res["balanced_acc"] = balanced_accuracy_score(
            np.asarray(yte, dtype=int), (p > thr_p).astype(int))
        res["model"] = name                      # 모델명 (실험명과 분리 → 표가 안 깨짐)
        res["n_feat"] = len(cols)
        res["split"] = split
        res["baseline_acc"] = BASE_ACC[split]
        res["gap_pp"] = (res["accuracy"] - BASE_ACC[split]) * 100
        if len(seeds) > 1:
            res["acc_std_seed"] = float(np.std(seed_accs, ddof=1))
        rows.append(res)

    df = pd.DataFrame(rows).round(4)
    front = ["model", "experiment", "n_feat", "accuracy", "baseline_acc", "gap_pp",
             "balanced_acc", "roc_auc", "f1", "n"]
    cols_order = front + [c for c in df.columns if c not in front]
    return df[cols_order], probs

# ==========================================================================
# 7. 핵심 비교의 ΔAUC 부트스트랩 CI (TDA 노트북 §8과 동일)
# ==========================================================================
def boot_auc_diff(y, p_a, p_b, n_boot=2000, seed=42):
    rng = np.random.default_rng(seed)
    y = np.asarray(y, dtype=int); p_a = np.asarray(p_a); p_b = np.asarray(p_b)
    diffs = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) < 2:
            continue
        diffs.append(roc_auc_score(y[i], p_b[i]) - roc_auc_score(y[i], p_a[i]))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(np.mean(diffs)), float(lo), float(hi)


COMPARISONS = [
    ("① 베이스라인 (22)",   "② +교차지수 (22+12)"),
    ("① 베이스라인 (22)",   "③ +TDA (22+8)"),
    ("② +교차지수 (22+12)", "②+③ 전부 (22+20)"),
    ("⑤ 변동성만 (4)",      "④ TDA만 (8)"),        # TDA가 변동성을 넘어서는가
    ("④ TDA만 (8)",        "⑥ 저중복 TDA (5)"),    # 희석 가설 검증
    ("⑤ 변동성만 (4)",      "⑥ 저중복 TDA (5)"),    # 저중복 TDA vs 변동성 정면 비교
]

def compare_table(probs, split="valid"):
    y = {"valid": y_valid_t, "test": y_test_t}[split]
    rows = []
    for a, b in COMPARISONS:
        m, lo, hi = boot_auc_diff(y, probs[a], probs[b])
        rows.append({"비교": f"{a} → {b}", "ΔAUC": m,
                     "95% CI 하한": lo, "95% CI 상한": hi,
                     "0 포함": "예" if lo <= 0 <= hi else "아니오"})
    print("※ CI가 0을 포함하면 '이 표본에서 차이가 확인되지 않음'이 정확한 표현.")
    return pd.DataFrame(rows).round(4)

# ==========================================================================
# 8. ★ 여기만 각자 것으로 — 아래 자기 블록의 주석만 풀면 끝
#    모델을 lambda 로 만들어 run_matrix 의 첫 인자로 넘깁니다.
#
#  ▸ 결과 읽는 법 (중요)
#    · 공통 축은 ROC-AUC 입니다. 네 모델 모두 AUC 로 비교합니다.
#    · accuracy 는 보조 지표이고, XGBoost 에서만 사용 불가입니다.
#      (트리 6그루라 예측확률이 0.53~0.56에 갇혀 전부 '상승'으로 분류 →
#       accuracy·f1·gap_pp 가 다수 클래스 값에 고정. balanced_acc 0.5가 그 신호)
#    · Ridge/Lasso 는 R² 를 함께 봅니다.
#    · 비교는 '모델 간'이 아니라 '실험 간'입니다. 자기 모델 안에서 ①vs③, ④vs⑤,
#      ④vs⑥ 을 보고, 네 사람의 판정 방향이 일치하는지를 8/22에 확인합니다.
# ==========================================================================

# ── 소진 (Bagging / GBM) ────────────────────────────────────────────────
# from sklearn.ensemble import BaggingClassifier, GradientBoostingClassifier
# from sklearn.tree import DecisionTreeClassifier
#
# bag = lambda s: BaggingClassifier(DecisionTreeClassifier(),
#                                   n_estimators=50, max_samples=0.8, random_state=s)
# gbm = lambda s: GradientBoostingClassifier(n_estimators=100, learning_rate=0.1,
#                                            max_depth=3, random_state=s)
# SEEDS = (0, 1, 2, 3, 4, 42, 100)          # 4차 회의에서 쓰신 7개 그대로
# res_bag, prob_bag = run_matrix(bag, "Bagging", seeds=SEEDS)
# res_gbm, prob_gbm = run_matrix(gbm, "GBM",     seeds=SEEDS)
# display(pd.concat([res_bag, res_gbm], ignore_index=True))
# display(compare_table(prob_gbm))
#
#  ▸ 시드 7개 평균으로 보는 이유: Bagging의 시드 표준편차가 0.0200인데
#    TDA 추가 효과는 커야 0.01~0.02 수준이라, 단일 실행으로는 판정 불가.
#    결과표의 acc_std_seed 컬럼을 gap_pp 와 반드시 같이 읽어 주세요.
#  ▸ baseline 은 0.5가 아니라 BASE_ACC(valid 0.5409 / test 0.5568)입니다.
#    4차 회의 표를 이 기준으로 다시 계산하면 GBM(test) 0.5160 → -4.1%p 가 됩니다.


# ── 서연 (XGBoost) ──────────────────────────────────────────────────────
# from xgboost import XGBClassifier
#
# xgb = lambda s: XGBClassifier(objective="binary:logistic",
#                               n_estimators=6, max_depth=2, learning_rate=0.03,
#                               min_child_weight=5, reg_lambda=5.0,
#                               subsample=0.8, colsample_bytree=0.8,
#                               tree_method="hist", random_state=s,
#                               n_jobs=-1, verbosity=0)
# res_xgb, prob_xgb = run_matrix(xgb, "XGBoost", seeds=(42, 202, 777, 1234, 20250815))
# display(res_xgb); display(compare_table(prob_xgb))
#
#  ▸ 전처리 재실행 불필요 — 배포된 npz를 쓰면 END_DATE 2026-01-01로 자동 통일됩니다.
#    (기존 노트북 §1~§16을 건너뛰고 이 공통 코드부터 실행)
#  ▸ 하이퍼파라미터는 ①에서 찾은 값을 ①~⑥에 그대로 고정합니다. 조합별 재탐색은
#    안 하는 쪽을 권합니다 — 1-SE 이내가 36/36이었으니 재탐색해도 같은 결론이 나오고,
#    실험마다 다른 설정을 쓰면 "TDA 효과"와 "튜닝 효과"가 섞입니다.
#    (현재 설정은 2025년 데이터로 고른 값이라는 점만 발표에 명시하면 충분합니다)
#  ▸ 여유가 되면 ③에서만 기존 TimeSeriesSplit 탐색을 한 번 더 돌려
#    1-SE 이내 조합이 36/36에서 줄어드는지 확인 → 줄면 "선택 가능한 신호가 생겼다"는 증거.


# ── 다영 (Ridge / Lasso) ────────────────────────────────────────────────
# from sklearn.linear_model import Ridge, Lasso
#
# ridge = lambda s: Ridge(alpha=1.0)
# lasso = lambda s: Lasso(alpha=1e-4, max_iter=10000)
# res_r, prob_r = run_matrix(ridge, "Ridge", use_scaled=True, task="reg")
# res_l, prob_l = run_matrix(lasso, "Lasso", use_scaled=True, task="reg")
# display(pd.concat([res_r, res_l], ignore_index=True))
#
#  ▸ task="reg" → 예측수익률의 부호로 방향을 판정하고, AUC는 예측수익률을 점수로 씁니다.
#    rmse·r2 컬럼도 같이 나오니 기존 회귀 지표 서사를 그대로 이어갈 수 있습니다.
#  ▸ Lasso 계수표를 꼭 같이 뽑아 주세요 (아래). 22개 중 20개를 0으로 만드셨는데,
#    TDA 피처가 0이 아닌 채로 살아남는지가 "선형이 못 잡는 정보가 TDA에 있는가"의
#    가장 직접적인 답입니다.
#
# sc = StandardScaler().fit(X_train_t[feat_cols + tda_cols])
# lz = Lasso(alpha=1e-4, max_iter=10000).fit(
#         sc.transform(X_train_t[feat_cols + tda_cols]), yr_train_t)
# coef = pd.Series(lz.coef_, index=feat_cols + tda_cols)
# print("0이 아닌 계수:", int((coef != 0).sum()), "/", len(coef))
# display(coef[coef != 0].sort_values(key=abs, ascending=False).round(6))


# ── 소현 (TDA / RandomForest) ───────────────────────────────────────────
# from sklearn.ensemble import RandomForestClassifier
#
# rf = lambda s: RandomForestClassifier(n_estimators=300, min_samples_leaf=20,
#                                       random_state=s, n_jobs=-1)
# res_rf, prob_rf = run_matrix(rf, "RandomForest")
# display(res_rf); display(compare_table(prob_rf))
#
#  ▸ 본 노트북 §8이 원본입니다. §8의 EXPERIMENTS 에 "⑤ 변동성만"·"⑥ 저중복 TDA",
#    COMPARISONS 에 ⑤→④ / ④→⑥ / ⑤→⑥ 을 추가해 두면 팀 전체와 표가 맞습니다.

# ==========================================================================
# 9. 제출 형식 (5차 회의 8/22)
#    - 위 결과표를 CSV로 저장해서 노션에 올려 주세요. 파일명: matrix_{이름}.csv
#    - split="valid" 결과를 기본으로 냅니다. test는 방향 확정 후 마지막에 1회만.
# ==========================================================================
# res_gbm.to_csv("matrix_소진.csv", index=False, encoding="utf-8-sig")
