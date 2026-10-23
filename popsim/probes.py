"""Linear probes and latent twin embeddings on JEV-9B hidden activations.

    uv run python -m popsim.probes

Tests whether JEV-9B's hidden representations at `[decision]:` contain individual-level
predictive signal that its final answers fail to express:
  1. Evaluates regularized ridge probes on the 18 polling items (`uses.ITEMS`) using the
     300-person training split (with internal 5-fold CV for layer and alpha selection) and
     scores on the 1,758 disjoint test participants.
  2. Compares against demographic baselines (rounded & continuous), zero-shot JEV-9B (`J`,
     `P`, and expected value `E[Y]`), supervised predictors on JEV output probabilities and
     log-probabilities, and a persona TF-IDF baseline.
  3. Runs token-position (`[state]` vs `[decision]:`), layer, PCA rank, head-architecture,
     and person-dependence (within-stratum & global shuffle) ablations, plus 5,000-draw
     paired bootstrap confidence intervals.
  4. Evaluates the reusable Layer 23 `[decision]:` person embedding across all 78 signal
     questions (`score.signal_columns`) and interprets its top principal components against
     Wave 1-3 human survey traits.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from popsim import baselines, score, twin, uses

POLLING_MAP = {
    **{f"False Cons. self _{i}": f"QID287#{i - 1}" for i in range(1, 11)},
    **{f"nonseparabilty bene _{i}": f"QID288#{i - 1}" for i in range(1, 5)},
    **{f"nonseparability ris _{i}": f"QID289#{i - 1}" for i in range(1, 5)},
}
ALPHAS_HIGH_D = (10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0, 10000.0, 30000.0, 100000.0)


def dual_ridge_cv(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    alphas: tuple[float, ...] = ALPHAS_HIGH_D,
    n_splits: int = 5,
    standardize: bool = True,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Exact ridge regression with unpenalized intercept solved via the n x n Gram matrix.
    All feature centering/scaling and CV selection use only `X_train` and `y_train`."""
    X_tr = np.asarray(X_train, dtype=np.float64)
    X_te = np.asarray(X_test, dtype=np.float64)
    y_tr = np.asarray(y_train, dtype=np.float64)

    mean_x = X_tr.mean(axis=0, keepdims=True)
    if standardize:
        std_x = X_tr.std(axis=0, keepdims=True)
        std_x = np.where(std_x < 1e-8, 1.0, std_x)
        Z_tr = (X_tr - mean_x) / std_x
        Z_te = (X_te - mean_x) / std_x
    else:
        Z_tr = X_tr - mean_x
        Z_te = X_te - mean_x

    n = len(Z_tr)
    rng = np.random.default_rng(0)
    folds = np.array_split(rng.permutation(n), n_splits)
    G_full = Z_tr @ Z_tr.T

    best_alpha = alphas[0]
    best_mse = float("inf")
    best_oof = np.zeros(n, dtype=np.float64)

    for alpha in alphas:
        oof = np.zeros(n, dtype=np.float64)
        for val_idx in folds:
            tr_idx = np.setdiff1d(np.arange(n), val_idx, assume_unique=True)
            z_tr_f = Z_tr[tr_idx]
            z_val_f = Z_tr[val_idx]
            m_f = z_tr_f.mean(axis=0, keepdims=True)
            z_tr_c = z_tr_f - m_f
            z_val_c = z_val_f - m_f

            y_tr_f = y_tr[tr_idx]
            y_mean_f = float(y_tr_f.mean())
            y_tr_c = y_tr_f - y_mean_f

            K_tr = z_tr_c @ z_tr_c.T
            K_val = z_val_c @ z_tr_c.T
            dual_w = np.linalg.solve(K_tr + alpha * np.eye(len(tr_idx)), y_tr_c)
            oof[val_idx] = y_mean_f + K_val @ dual_w
        cv_mse = float(np.mean((oof - y_tr) ** 2))
        if cv_mse < best_mse:
            best_mse = cv_mse
            best_alpha = alpha
            best_oof = oof

    y_mean = float(y_tr.mean())
    dual_w = np.linalg.solve(G_full + best_alpha * np.eye(n), y_tr - y_mean)
    pred_test = y_mean + (Z_te @ Z_tr.T) @ dual_w
    return pred_test, best_oof, best_alpha, best_mse


def build_tfidf_features(texts: pd.Series, train_pids: pd.Index, max_vocab: int = 2500) -> pd.DataFrame:
    """TF-IDF bag-of-words features fit strictly on training participants."""
    token_re = re.compile(r"[a-z0-9_']+")
    tokenized = {pid: token_re.findall(text.lower()) for pid, text in texts.items()}
    df_counts: Counter[str] = Counter()
    tf_totals: Counter[str] = Counter()
    for pid in train_pids:
        toks = tokenized[pid]
        tf_totals.update(toks)
        df_counts.update(set(toks))

    n_train = len(train_pids)
    candidates = [w for w, _ in tf_totals.most_common() if 3 <= df_counts[w] <= int(0.98 * n_train)][:max_vocab]
    vocab = {w: i for i, w in enumerate(candidates)}
    idf = np.array([np.log((1 + n_train) / (1 + df_counts[w])) + 1.0 for w in candidates])

    mat = np.zeros((len(texts), len(candidates)), dtype=float)
    for row_idx, (_, toks) in enumerate(tokenized.items()):
        counts = Counter(t for t in toks if t in vocab)
        if not counts:
            continue
        total = sum(counts.values())
        for w, c in counts.items():
            mat[row_idx, vocab[w]] = (c / total) * idf[vocab[w]]
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    mat = mat / np.where(norms < 1e-8, 1.0, norms)
    return pd.DataFrame(mat, index=texts.index)


def load_captured_chunks(out_dir: Path | None = None) -> dict[str, np.ndarray]:
    """Load captured JEV-9B activation chunks from `cache/probes/`."""
    out_dir = out_dir or (twin.ROOT / "cache" / "probes")
    files = sorted(out_dir.glob("jev9b_polling_chunk_*.npz"))
    if not files:
        raise FileNotFoundError(f"No activation chunks found in {out_dir}")
    pids, state_acts, decision_acts, probs, logprobs = [], [], [], [], []
    layers, keys = None, None
    for f in files:
        d = np.load(f)
        pids.append(d["pids"])
        state_acts.append(d["state_acts"])
        decision_acts.append(d["decision_acts"])
        probs.append(d["probs"])
        logprobs.append(d["logprobs"])
        if layers is None:
            layers = d["layers"]
            keys = d["keys"]
    return {
        "pids": np.concatenate(pids, axis=0),
        "layers": layers,
        "keys": keys,
        "state_acts": np.concatenate(state_acts, axis=0),
        "decision_acts": np.concatenate(decision_acts, axis=0),
        "probs": np.concatenate(probs, axis=0),
        "logprobs": np.concatenate(logprobs, axis=0),
    }


def within_stratum_shuffle_indices(demo_df: pd.DataFrame, pids: pd.Index, seed: int = 0) -> np.ndarray:
    """Permute participants only within identical demographic strata (party x ideology x gender x age)."""
    rng = np.random.default_rng(seed)
    sub = demo_df.loc[pids, ["QID20", "QID22", "QID12", "QID13"]].astype(str)
    strata = sub.apply(lambda r: "|".join(r.values), axis=1)
    perm = np.arange(len(pids))
    for _, group in strata.groupby(strata):
        idxs = [pids.get_loc(pid) for pid in group.index]
        if len(idxs) > 1:
            perm[idxs] = rng.permutation(idxs)
    return perm


def paired_bootstrap_ci(
    r_a: pd.Series, r_b: pd.Series, n_boot: int = 5000, seed: int = 0
) -> tuple[float, float, float, int]:
    """Paired bootstrap over questions for mean(r_a - r_b)."""
    valid = r_a.index.intersection(r_b.index)
    diff = (r_a.loc[valid] - r_b.loc[valid]).to_numpy(dtype=float)
    diff = diff[~np.isnan(diff)]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(n_boot, len(diff)))
    draws = diff[idx].mean(axis=1)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return float(diff.mean()), float(lo), float(hi), int(np.sum(diff > 0))


def per_item_corrs(pred_df: pd.DataFrame, target_df: pd.DataFrame, items: list[str]) -> pd.Series:
    out = {}
    for col in items:
        p = pred_df[col].astype(float)
        y = target_df[col].astype(float)
        ok = p.notna() & y.notna()
        out[col] = float(np.corrcoef(p[ok], y[ok])[0, 1]) if p[ok].std() > 1e-8 else 0.0
    return pd.Series(out)


def eval_polling_predictions(
    preds: dict[str, pd.DataFrame],
    w4: pd.DataFrame,
    w13: pd.DataFrame,
    demo_df: pd.DataFrame,
    test_pids: pd.Index,
    demo_pred: pd.DataFrame,
) -> pd.DataFrame:
    """Evaluate correlation (vs w13 and w4), partial correlation controlling for demo_pred,
    similarity (1 - normalized MAE), topline error, and subgroup gap metrics on test_pids."""
    items = list(uses.ITEMS.keys())
    clear_poll = uses.clear_gaps(w4, demo_df)
    rows = {}
    for name, df in preds.items():
        sub = df.loc[test_pids, items]
        r_w13, r_w4, r_part_w13, r_part_w4, sims = [], [], [], [], []
        for col, (lo, hi) in uses.ITEMS.items():
            p = sub[col].astype(float)
            y13 = w13.loc[test_pids, col].astype(float)
            y4 = w4.loc[test_pids, col].astype(float)
            d_p = demo_pred.loc[test_pids, col].astype(float)

            ok13 = p.notna() & y13.notna()
            r_w13.append(float(np.corrcoef(p[ok13], y13[ok13])[0, 1]) if p[ok13].std() > 1e-8 else 0.0)

            ok4 = p.notna() & y4.notna()
            r_w4.append(float(np.corrcoef(p[ok4], y4[ok4])[0, 1]) if p[ok4].std() > 1e-8 else 0.0)

            for y_target, bucket in ((y13, r_part_w13), (y4, r_part_w4)):
                ok = p.notna() & y_target.notna() & d_p.notna()
                dp_v = d_p[ok].to_numpy()
                y_v = y_target[ok].to_numpy()
                p_v = p[ok].to_numpy()
                X_d = np.column_stack([np.ones(len(dp_v)), dp_v])
                y_res = y_v - X_d @ np.linalg.lstsq(X_d, y_v, rcond=None)[0]
                p_res = p_v - X_d @ np.linalg.lstsq(X_d, p_v, rcond=None)[0]
                if np.std(p_res) > 1e-8 and np.std(y_res) > 1e-8:
                    bucket.append(float(np.corrcoef(p_res, y_res)[0, 1]))
                else:
                    bucket.append(0.0)

            sims.append(float(1.0 - np.abs(p[ok13] - y13[ok13]).mean() / (hi - lo)))

        poll_stats = uses.polling(sub, w4.loc[test_pids], demo_df, clear_poll)
        rows[name] = {
            "corr_w1_3": float(np.mean(r_w13)),
            "corr_w4": float(np.mean(r_w4)),
            "partial_r_w1_3|demo": float(np.mean(r_part_w13)),
            "partial_r_w4|demo": float(np.mean(r_part_w4)),
            "sim_w1_3": float(np.mean(sims)),
            "topline_err": poll_stats["topline error"],
            "gap_err_SD": poll_stats["gap error (SD)"],
            "gap_ratio": poll_stats["gap ratio"],
        }
    return pd.DataFrame.from_dict(rows, orient="index")


def interpret_layer23_pcs(
    H_person: np.ndarray,
    tr_rows: np.ndarray,
    te_rows: np.ndarray,
    test_pids: pd.Index,
    w4: pd.DataFrame,
    jev_ev: pd.DataFrame,
    demo_df: pd.DataFrame,
    n_pcs: int = 8,
) -> list[dict]:
    """Correlate the top principal components of the Layer 23 person embedding with Wave 1-3
    survey traits and compare Human vs JEV E[Y] polling loadings."""
    H_c = H_person - H_person[tr_rows].mean(axis=0, keepdims=True)
    _, S, Vt = np.linalg.svd(H_c[tr_rows], full_matrices=False)
    var_exp = (S**2) / np.sum(S**2)
    PC_te = (H_c @ Vt[:n_pcs].T)[te_rows]

    w13_raw = pd.read_csv(
        twin.dataset_file("question_catalog_and_human_response_csv/wave1_3_response.csv")
    ).set_index("pid").loc[test_pids]
    with open(twin.dataset_file("question_catalog_and_human_response_csv/question_catalog.json")) as f:
        cat = json.load(f)

    col_desc = {}
    for q in cat:
        qid = q.get("QuestionID", "")
        bname = q.get("BlockName", "").strip()
        qtext = " ".join(str(q.get("QuestionText", "")).split())[:80]
        if q.get("QuestionType") == "Matrix" and "Rows" in q:
            for idx, row_text in enumerate(q["Rows"], 1):
                r_clean = " ".join(str(row_text).split())[:70]
                col_desc[f"{qid}_{idx}"] = f"[{bname}] {r_clean}"
        else:
            col_desc[qid] = f"[{bname}] {qtext}"

    demo_dummies = pd.get_dummies(demo_df.loc[test_pids].astype("string").fillna("missing"), dtype=float)
    X_d = np.column_stack([np.ones(len(test_pids)), demo_dummies.to_numpy()])

    pol_h = w4.loc[test_pids, [f"False Cons. self _{i}" for i in range(1, 8)]].astype(float).mean(axis=1)
    pol_j = jev_ev.loc[test_pids, [f"False Cons. self _{i}" for i in range(1, 8)]].mean(axis=1)
    ben_h = w4.loc[test_pids, [f"nonseparabilty bene _{i}" for i in range(1, 5)]].astype(float).mean(axis=1)
    ben_j = jev_ev.loc[test_pids, [f"nonseparabilty bene _{i}" for i in range(1, 5)]].mean(axis=1)
    ris_h = w4.loc[test_pids, [f"nonseparability ris _{i}" for i in range(1, 5)]].astype(float).mean(axis=1)
    ris_j = jev_ev.loc[test_pids, [f"nonseparability ris _{i}" for i in range(1, 5)]].mean(axis=1)

    out = []
    for k in range(n_pcs):
        pc = PC_te[:, k]
        beta = np.linalg.lstsq(X_d, pc, rcond=None)[0]
        pc_hat = X_d @ beta
        r2_demo = float(np.var(pc_hat) / np.var(pc))
        pc_res = pc - pc_hat

        corrs_raw, corrs_res = [], []
        for c in w13_raw.columns:
            s = pd.to_numeric(w13_raw[c], errors="coerce")
            ok = s.notna().to_numpy()
            if ok.sum() < 500 or s[ok].std() < 1e-6:
                continue
            v = s.to_numpy()[ok]
            r_raw = float(np.corrcoef(pc[ok], v)[0, 1])
            r_res = float(np.corrcoef(pc_res[ok], v)[0, 1])
            corrs_raw.append((abs(r_raw), r_raw, c))
            if c not in {"QID12", "QID13", "QID14", "QID15", "QID20", "QID21", "QID22"}:
                corrs_res.append((abs(r_res), r_res, c))
        corrs_raw.sort(reverse=True)
        corrs_res.sort(reverse=True)

        out.append({
            "pc": f"PC{k + 1}",
            "var_explained": round(float(var_exp[k]), 4),
            "demo_r2": round(r2_demo, 4),
            "non_demo_share": round(1.0 - r2_demo, 4),
            "loadings_human": {
                "policy": round(float(np.corrcoef(pc, pol_h)[0, 1]), 4),
                "tech_benefit": round(float(np.corrcoef(pc, ben_h)[0, 1]), 4),
                "tech_risk": round(float(np.corrcoef(pc, ris_h)[0, 1]), 4),
            },
            "loadings_jev_ey": {
                "policy": round(float(np.corrcoef(pc, pol_j)[0, 1]), 4),
                "tech_benefit": round(float(np.corrcoef(pc, ben_j)[0, 1]), 4),
                "tech_risk": round(float(np.corrcoef(pc, ris_j)[0, 1]), 4),
            },
            "top_correlates": [
                {"col": c, "r": round(r_v, 4), "desc": col_desc.get(c, c)} for _, r_v, c in corrs_raw[:3]
            ],
            "top_beyond_demo_correlates": [
                {"col": c, "partial_r": round(r_v, 4), "desc": col_desc.get(c, c)} for _, r_v, c in corrs_res[:3]
            ],
        })
    return out


def main() -> None:
    people = twin.participants()
    train_df, test_df = baselines.split(people, n_train=300)
    train_pids, test_pids = pd.Index(train_df.pid), pd.Index(test_df.pid)

    data = load_captured_chunks()
    pids = pd.Index(data["pids"])
    layers = [int(l) for l in data["layers"]]
    keys = [str(k) for k in data["keys"]]
    pid_to_row = {int(pid): i for i, pid in enumerate(pids)}
    tr_rows = np.array([pid_to_row[int(p)] for p in train_pids], dtype=np.int64)
    te_rows = np.array([pid_to_row[int(p)] for p in test_pids], dtype=np.int64)

    csv_dir = twin.ROOT / "results" / "summary-n2058" / "J" / "csv"
    w4 = uses.formatted(csv_dir / "responses_wave4_formatted.csv")
    w13 = uses.formatted(csv_dir / "responses_wave1_3_formatted.csv")
    jev_j = uses.formatted(csv_dir / "responses_llm_imputed_formatted.csv")
    jev_p = uses.formatted(twin.ROOT / "results" / "summary-n2058" / "P" / "csv" / "responses_llm_imputed_formatted.csv")
    demo_df = uses.demographics()
    demo_dummies = pd.get_dummies(demo_df.astype("string").fillna("missing"), dtype=float)

    items = list(uses.ITEMS.keys())
    key_to_qi = {k: i for i, k in enumerate(keys)}

    # 1. Baseline controls
    poll_ranges = {col: uses.ITEMS[col] for col in items}
    demo_fixed = baselines.demographic_predictions(w4.loc[train_pids], demo_df, test_pids, poll_ranges)
    demo_cont = pd.DataFrame(index=test_pids, columns=items, dtype=float)
    jev_ev_all = pd.DataFrame(index=pids, columns=items, dtype=float)
    prob_ridge = pd.DataFrame(index=test_pids, columns=items, dtype=float)
    logprob_ridge = pd.DataFrame(index=test_pids, columns=items, dtype=float)
    demo_plus_probs = pd.DataFrame(index=test_pids, columns=items, dtype=float)

    X_demo_tr = demo_dummies.loc[train_pids].to_numpy()
    X_demo_te = demo_dummies.loc[test_pids].to_numpy()

    for col in items:
        qi = key_to_qi[POLLING_MAP[col]]
        lo, hi = uses.ITEMS[col]
        k_len = hi - lo + 1
        P = data["probs"][:, qi, :k_len]
        LP = data["logprobs"][:, qi, :k_len]
        LP = LP - LP.mean(axis=1, keepdims=True)
        ev = P @ np.arange(1, k_len + 1, dtype=float)
        jev_ev_all[col] = ev

        y_tr = w4.loc[train_pids, col].astype(float).to_numpy()
        demo_cont[col], _, _, _ = dual_ridge_cv(
            X_demo_tr, y_tr, X_demo_te, alphas=(0.1, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0), standardize=False
        )
        prob_ridge[col], _, _, _ = dual_ridge_cv(
            P[tr_rows], y_tr, P[te_rows], alphas=(0.01, 0.1, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0)
        )
        logprob_ridge[col], _, _, _ = dual_ridge_cv(
            LP[tr_rows], y_tr, LP[te_rows], alphas=(0.01, 0.1, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0)
        )
        demo_plus_probs[col], _, _, _ = dual_ridge_cv(
            np.hstack([X_demo_tr, P[tr_rows]]),
            y_tr,
            np.hstack([X_demo_te, P[te_rows]]),
            alphas=(1.0, 3.0, 10.0, 30.0, 100.0, 300.0),
        )

    tfidf_df = build_tfidf_features(people.set_index("pid")["persona_summary"], train_pids, max_vocab=2500)
    tfidf_ridge = pd.DataFrame(index=test_pids, columns=items, dtype=float)
    for col in items:
        y_tr = w4.loc[train_pids, col].astype(float).to_numpy()
        tfidf_ridge[col], _, _, _ = dual_ridge_cv(
            tfidf_df.loc[train_pids].to_numpy(),
            y_tr,
            tfidf_df.loc[test_pids].to_numpy(),
            alphas=(0.1, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0),
            standardize=False,
        )

    # 2. Layer sweep ([state] vs [decision]:) & Layer 23 ablations
    state_preds = {l: pd.DataFrame(index=test_pids, columns=items, dtype=float) for l in layers}
    decision_preds = {l: pd.DataFrame(index=test_pids, columns=items, dtype=float) for l in layers}
    for col in items:
        qi = key_to_qi[POLLING_MAP[col]]
        y_tr = w4.loc[train_pids, col].astype(float).to_numpy()
        for li, l in enumerate(layers):
            X_s = data["state_acts"][:, li, :]
            state_preds[l][col], _, _, _ = dual_ridge_cv(X_s[tr_rows], y_tr, X_s[te_rows])
            X_d = data["decision_acts"][:, qi, li, :]
            decision_preds[l][col], _, _, _ = dual_ridge_cv(X_d[tr_rows], y_tr, X_d[te_rows])

    l23_idx = layers.index(23)
    H_l23 = data["decision_acts"][:, :, l23_idx, :].astype(np.float64)
    H_l23_c = (H_l23 - H_l23[tr_rows].mean(axis=0, keepdims=True)) / np.maximum(
        H_l23[tr_rows].std(axis=0, keepdims=True), 1e-6
    )
    H_person = H_l23_c.mean(axis=1)  # (2058, 4096) reusable person embedding

    pca128_preds = pd.DataFrame(index=test_pids, columns=items, dtype=float)
    residual_preds = pd.DataFrame(index=test_pids, columns=items, dtype=float)
    loqo_person_preds = pd.DataFrame(index=test_pids, columns=items, dtype=float)
    q_delta_preds = pd.DataFrame(index=test_pids, columns=items, dtype=float)

    for col in items:
        qi = key_to_qi[POLLING_MAP[col]]
        y_tr = w4.loc[train_pids, col].astype(float).to_numpy()
        X_tr, X_te = H_l23_c[tr_rows, qi, :], H_l23_c[te_rows, qi, :]

        _, _, Vt = np.linalg.svd(X_tr, full_matrices=False)
        V128 = Vt[:128].T
        pca128_preds[col], _, _, _ = dual_ridge_cv(
            X_tr @ V128, y_tr, X_te @ V128, alphas=(1.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0), standardize=False
        )

        ev_tr = jev_ev_all.loc[train_pids, col].to_numpy()
        ev_te = jev_ev_all.loc[test_pids, col].to_numpy()
        res_p, _, _, _ = dual_ridge_cv(X_tr, y_tr - ev_tr, X_te, standardize=False)
        residual_preds[col] = ev_te + res_p

        other_qis = [j for j in range(18) if j != qi]
        H_p_loqo = H_l23_c[:, other_qis, :].mean(axis=1)
        H_delta = H_l23_c[:, qi, :] - H_p_loqo
        loqo_person_preds[col], _, _, _ = dual_ridge_cv(H_p_loqo[tr_rows], y_tr, H_p_loqo[te_rows])
        q_delta_preds[col], _, _, _ = dual_ridge_cv(H_delta[tr_rows], y_tr, H_delta[te_rows])

    strat_perm = within_stratum_shuffle_indices(demo_df, test_pids, seed=0)
    global_perm = np.random.default_rng(0).permutation(len(test_pids))
    probe_stratum_shuffled = pd.DataFrame(decision_preds[23].to_numpy()[strat_perm], index=test_pids, columns=items)
    probe_global_shuffled = pd.DataFrame(decision_preds[23].to_numpy()[global_perm], index=test_pids, columns=items)

    arms = {
        "Demographic ridge (rounded, lam=10)": demo_fixed,
        "Demographic ridge (continuous, CV)": demo_cont,
        "Persona TF-IDF ridge (300 train)": tfidf_ridge,
        "JEV-9B P (sampled, zero-shot)": jev_p.loc[test_pids],
        "JEV-9B J (argmax, zero-shot)": jev_j.loc[test_pids],
        "JEV-9B E[Y] (prob expectation, zero-shot)": jev_ev_all.loc[test_pids],
        "JEV-9B probs ridge (300 train)": prob_ridge,
        "JEV-9B logprobs ridge (300 train)": logprob_ridge,
        "Demographics + JEV probs (concat ridge)": demo_plus_probs,
        "Probe [state] Layer 19 (best [state])": state_preds[19],
        **{f"Probe [decision] Layer {l}": decision_preds[l] for l in layers},
        "Probe [decision] Layer 23 (PCA top-128)": pca128_preds,
        "Residual Probe (JEV E[Y] + Layer 23 probe)": residual_preds,
        "Decomp A: Mean [decision] from other 17 Qs": loqo_person_preds,
        "Decomp B: Question-specific [decision] delta": q_delta_preds,
        "Shuffle Check: Within-Demo-Stratum Permute": probe_stratum_shuffled,
        "Shuffle Check: Global Persona Permute": probe_global_shuffled,
        "Humans, retest (wave 1-3 ceiling)": w13.loc[test_pids],
    }

    polling_table = eval_polling_predictions(arms, w4, w13, demo_df, test_pids, demo_cont)
    print("\n=== 18 Polling Items Probing & Controls (300 Train -> 1,758 Disjoint Test) ===")
    print(polling_table.round(4).to_string())

    # 3. Paired bootstrap CIs
    r_l23 = per_item_corrs(decision_preds[23], w13.loc[test_pids], items)
    r_pca = per_item_corrs(pca128_preds, w13.loc[test_pids], items)
    r_j = per_item_corrs(jev_j.loc[test_pids], w13.loc[test_pids], items)
    r_ev = per_item_corrs(jev_ev_all.loc[test_pids], w13.loc[test_pids], items)
    r_pr = per_item_corrs(prob_ridge, w13.loc[test_pids], items)
    r_dm = per_item_corrs(demo_cont, w13.loc[test_pids], items)
    r_dp = per_item_corrs(demo_plus_probs, w13.loc[test_pids], items)

    boot_cis = {}
    print("\n=== Paired Bootstrap 95% CIs Over the 18 Polling Items (corr_w1_3) ===")
    for label, a, b in [
        ("Probe L23 vs JEV-9B J (argmax)", r_l23, r_j),
        ("Probe L23 vs JEV-9B E[Y] (zero-shot)", r_l23, r_ev),
        ("Probe L23 vs JEV-9B probs ridge", r_l23, r_pr),
        ("Probe L23 vs Demographic ridge (continuous)", r_l23, r_dm),
        ("Probe L23 vs Demographics + JEV probs", r_l23, r_dp),
        ("Probe L23 PCA-128 vs JEV-9B probs ridge", r_pca, r_pr),
    ]:
        mean_d, lo, hi, wins = paired_bootstrap_ci(a, b)
        boot_cis[label] = {"delta": round(mean_d, 4), "ci95": [round(lo, 4), round(hi, 4)], "wins_of_18": wins}
        print(f"  {label:46s}: {mean_d:+.4f} [{lo:+.4f}, {hi:+.4f}] (wins {wins}/18)")

    # 4. Evaluate Reusable Layer 23 Person Embedding across all 78 Signal Items
    col_map = {c.upper(): c for c in w4.columns}
    sig_ranges = {
        c: r
        for c, r in score.question_ranges().items()
        if c not in {"Q164", "Q166", "Q168", "Q170"} and not c.endswith("_Q295")
    }
    r_all_j, r_all_demo, r_all_probe = [], [], []
    dis_all_j, dis_all_demo, dis_all_probe = [], [], []
    for c_up in sig_ranges:
        col = col_map[c_up]
        y_tr = pd.to_numeric(w4.loc[train_pids, col], errors="coerce").to_numpy(dtype=float)
        ok_tr = ~np.isnan(y_tr)
        pred_d, _, _, _ = dual_ridge_cv(
            X_demo_tr[ok_tr], y_tr[ok_tr], X_demo_te, alphas=(1.0, 10.0, 30.0, 100.0, 300.0), standardize=False
        )
        pred_p, _, _, _ = dual_ridge_cv(H_person[tr_rows][ok_tr], y_tr[ok_tr], H_person[te_rows])

        y4_te = pd.to_numeric(w4.loc[test_pids, col], errors="coerce")
        y13_te = pd.to_numeric(w13.loc[test_pids, col], errors="coerce")
        pj_te = pd.to_numeric(jev_j.loc[test_pids, col], errors="coerce")
        ok_te = (y13_te.notna() & pj_te.notna()).to_numpy()

        rj = float(np.corrcoef(pj_te.to_numpy()[ok_te], y13_te.to_numpy()[ok_te])[0, 1]) if pj_te[ok_te].std() > 1e-8 else 0.0
        rd = float(np.corrcoef(pred_d[ok_te], y13_te.to_numpy()[ok_te])[0, 1]) if np.std(pred_d[ok_te]) > 1e-8 else 0.0
        rp = float(np.corrcoef(pred_p[ok_te], y13_te.to_numpy()[ok_te])[0, 1]) if np.std(pred_p[ok_te]) > 1e-8 else 0.0
        r_all_j.append(rj)
        r_all_demo.append(rd)
        r_all_probe.append(rp)

        ok_rel = (y4_te.notna() & y13_te.notna()).to_numpy()
        rel = float(np.corrcoef(y4_te.to_numpy()[ok_rel], y13_te.to_numpy()[ok_rel])[0, 1])
        if rel > 0:
            dis_all_j.append(rj / rel)
            dis_all_demo.append(rd / rel)
            dis_all_probe.append(rp / rel)

    all78_summary = {
        "JEV-9B J (argmax)": {"person_signal": round(float(np.mean(r_all_j)), 4), "disattenuated": round(float(np.mean(dis_all_j)), 4)},
        "Demographic ridge (continuous, CV)": {"person_signal": round(float(np.mean(r_all_demo)), 4), "disattenuated": round(float(np.mean(dis_all_demo)), 4)},
        "Reusable Layer 23 Person Embedding Probe": {"person_signal": round(float(np.mean(r_all_probe)), 4), "disattenuated": round(float(np.mean(dis_all_probe)), 4)},
    }
    print("\n=== All 78 Signal Items Evaluation (1,758 Disjoint Test Participants) ===")
    print(pd.DataFrame(all78_summary).T.to_string())

    # 5. Interpret top Layer 23 PCs
    pc_interp = interpret_layer23_pcs(H_person, tr_rows, te_rows, test_pids, w4, jev_ev_all, demo_df, n_pcs=8)

    out_path = twin.ROOT / "results" / "probes.json"
    out_path.write_text(
        json.dumps(
            {
                "n_train": len(train_pids),
                "n_test": len(test_pids),
                "layers": layers,
                "polling_metrics": polling_table.round(4).to_dict(orient="index"),
                "bootstrap_ci95_corr_w1_3": boot_cis,
                "all_78_signal_items": all78_summary,
                "layer23_pc_interpretation": pc_interp,
            },
            indent=2,
        )
    )
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
