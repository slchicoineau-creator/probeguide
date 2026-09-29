"""Teacher-strength ladder (pre-registered in the project work log, 2026-09-27 17:36 CDT; design from the judge gate).
Teachers: HF scrollprize/PHerc.1667-iteration-1..5 (rungs), iteration-0 = reference only (never primary).
Inputs: wt_points.npz (iteration-0) and wt_points_iter{1..5}.npz from `weakteacher.py --model ... --out ...`.

  python ladder_stats.py --select   # T alone only: direction eligibility, strength s_k, leak flag -> ladder_primary.json
  python ladder_stats.py --test     # requires ladder_primary.json: H1-H4 on the primary rung (one-sided exact Wilcoxon,
                                    # Holm, family alpha 0.05) + descriptive rows for all rungs.
Primary metric for "DINO adds/removes" = WITHIN-CHUNK AUPRC (chunk qualifies with >= 50 points of each class; mean over a
segment's qualifying chunks; paired per segment, n = 8)."""
import _env  # path setup (VILLA_SRC, PROBEGUIDE_ASSETS, PROBEGUIDE_DATA)
import glob, hashlib, json, sys, numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score as AP, roc_auc_score as AUC, precision_recall_curve

MINC = 50
RUNGS = {0: "wt_points.npz", **{k: f"wt_points_iter{k}.npz" for k in range(1, 6)}}
base = np.load("wt_points.npz", allow_pickle=True)
seg_pt, ch, xyz, y = base["seg"].astype(str), base["chunk"].astype(int), base["xyz"], base["ink"].astype(int)
segs = sorted(set(seg_pt)); folds = [segs[i::4] for i in range(4)]
R = {}
for k, f in RUNGS.items():
    d = np.load(f, allow_pickle=True)
    for key in ("seg", "rr", "cc"):
        assert np.array_equal(d[key], base[key]), f"rung {k}: {key} not paired with wt_points.npz"
    R[k] = d


def seg_auc(sc, s):
    m = seg_pt == s; return AUC(y[m], sc[m])


def select():
    out = {"rungs": {}}
    for k, d in R.items():
        folds_info, eligible = [], True
        picks = []
        for fi, test in enumerate(folds):
            tr = ~np.isin(seg_pt, test)
            a = {t: float(AUC(y[tr], d[t][tr])) for t in ("t_fwd", "t_rev")}
            pick = max(a, key=a.get); picks.append(pick)
            if abs(a["t_fwd"] - a["t_rev"]) < 0.03:
                eligible = False
            folds_info.append(dict(fold=fi, auroc=a, pick=pick))
        if len(set(picks)) > 1:
            eligible = False
        s_k = float(np.mean([fo["auroc"][fo["pick"]] for fo in folds_info]))
        T = np.zeros(len(y))
        for fi, test in enumerate(folds):
            te = np.isin(seg_pt, test); T[te] = d[folds_info[fi]["pick"]][te]
        held = {s: float(seg_auc(T, s)) for s in segs}
        leak = [s for s in segs if held[s] - np.median([held[o] for o in segs if o != s]) > 0.10]
        band = "noise" if s_k < 0.58 else ("weak" if s_k < 0.80 else "strong")
        out["rungs"][k] = dict(s_k=s_k, band=band, eligible=bool(eligible and k != 0), picks=picks, folds=folds_info,
                               heldout_auroc=held, leak_flag=leak, predicted_rev=(k != 0))
        print(f"rung {k}: s_k={s_k:.3f} band={band} picks={picks} eligible={eligible and k != 0} leak={leak}")
    cand = {k: v for k, v in out["rungs"].items() if v["eligible"]}
    weak = {k: v for k, v in cand.items() if 0.58 <= v["s_k"] < 0.80}
    strong = {k: v for k, v in cand.items() if v["s_k"] >= 0.80}
    if weak:
        prim = min(weak, key=lambda k: weak[k]["s_k"]); outcome = "weak rung"
    elif strong:
        prim = min(strong, key=lambda k: strong[k]["s_k"]); outcome = "Kill A"
    else:
        prim = None; outcome = "Kill N"
    out.update(primary=prim, outcome=outcome)
    js = json.dumps(out, indent=1, default=float)
    open("ladder_primary.json", "w").write(js)
    print("PRIMARY:", prim, "| outcome:", outcome, "| sha256(ladder_primary.json) =", hashlib.sha256(js.encode()).hexdigest())


# ------------------------------------------------------------------ test phase
def build_features():
    ref = np.load(f"{_env.ASSETS}/avg_ref_embedding.npy").astype(np.float32); ref /= np.linalg.norm(ref)
    files = sorted(glob.glob("emb/chunk_*.npz"))
    tok_id = np.zeros(len(y), int); feats, sup_rows, off = [], [], 0; s_proto = np.zeros(len(y))
    for n, f in enumerate(files):
        d = np.load(f); m = ch == n
        E = d["emb"].astype(np.float32); E /= np.linalg.norm(E, axis=-1, keepdims=True) + 1e-9
        cos = E @ ref; inc = (cos - cos.min()) / max(cos.max() - cos.min(), 1e-6)
        tk = np.clip(((xyz[m] - np.asarray(d["origin_xyz"])) // 8).astype(int), 0, 31)
        pts = np.stack([tk[:, 2], tk[:, 1], tk[:, 0]], 1) if m.any() else np.zeros((0, 3), int)
        s_proto[m] = inc[pts[:, 0], pts[:, 1], pts[:, 2]]
        sup = np.argwhere(d["sup"] > 0)
        allk = np.unique(np.concatenate([sup, pts]), axis=0); key = {tuple(k): i for i, k in enumerate(allk)}
        feats.append(E[allk[:, 0], allk[:, 1], allk[:, 2]])
        if m.any():
            tok_id[m] = off + np.array([key[tuple(k)] for k in pts])
        for k in sup:
            sup_rows.append((off + key[tuple(k)], int(d["ink"][tuple(k)]), str(d["seg"])))
        off += len(allk)
    F = np.concatenate(feats)
    sup_idx = np.array([r[0] for r in sup_rows]); sup_y = np.array([r[1] for r in sup_rows]); sup_seg = np.array([r[2] for r in sup_rows])
    return F, tok_id, s_proto, sup_idx, sup_y, sup_seg


lg = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))


def within(s, met):
    out = []
    for g in segs:
        v = [met(y[(seg_pt == g) & (ch == c)], s[(seg_pt == g) & (ch == c)]) for c in np.unique(ch[seg_pt == g])
             if (y[(seg_pt == g) & (ch == c)] == 1).sum() >= MINC and (y[(seg_pt == g) & (ch == c)] == 0).sum() >= MINC]
        out.append(np.mean(v))
    return np.array(out)


def per_seg(s, met):
    return np.array([met(y[seg_pt == g], s[seg_pt == g]) for g in segs])


def prf(lab, g):
    tp = (lab & (g == 1)).sum(); P = tp / max(lab.sum(), 1); Rr = tp / max((g == 1).sum(), 1)
    return P, Rr, 2 * P * Rr / max(P + Rr, 1e-9)


def run_rung(k, sel, F, tok_id, s_proto, sup_idx, sup_y, sup_seg):
    d = R[k]; picks = sel["rungs"][str(k)]["picks"]
    probe = lambda tr_segs: LogisticRegression(C=0.5, max_iter=3000, class_weight="balanced").fit(
        F[sup_idx[np.isin(sup_seg, tr_segs)]], sup_y[np.isin(sup_seg, tr_segs)])
    T = np.zeros(len(y)); S = np.zeros(len(y)); stk = np.zeros(len(y)); tm = np.zeros(len(y), bool)

    def dm(X, mask):
        X = X.copy()
        for n in np.unique(ch[mask]):
            mm = (ch == n) & mask; X[mm] -= X[mm].mean(0)
        return X
    for fi, test in enumerate(folds):
        train = [s for s in segs if s not in test]; te = np.isin(seg_pt, test); tr = ~te
        Tf = d[picks[fi]].astype(np.float64); T[te] = Tf[te]
        pr = probe(train); S[te] = pr.predict_proba(F[tok_id[te]])[:, 1]
        Sin = np.zeros(len(y))
        for s in train:
            mm = seg_pt == s; Sin[mm] = probe([t for t in train if t != s]).predict_proba(F[tok_id[mm]])[:, 1]
        Xall_tr = dm(np.c_[lg(Tf), lg(Sin)], tr); Xall_te = dm(np.c_[lg(Tf), lg(S)], te)
        st = LogisticRegression(max_iter=1000).fit(Xall_tr[tr], y[tr]); stk[te] = st.decision_function(Xall_te[te])
        g_tr = (Tf[tr] * s_proto[tr]) > 0.5; tau = np.quantile(Tf[tr], 1 - g_tr.mean()); tm[te] = Tf[te] > tau
    res = {"pooled": {a: dict(auprc=float(AP(y, v)), auroc=float(AUC(y, v))) for a, v in
                      {"T": T, "T*S_proto": T * s_proto, "T*S_probe": T * S, "stack_dm": stk}.items()}}
    wT = within(T, AP)
    res["H1"] = within(T * s_proto, AP) - wT
    res["H2"] = within(T * S, AP) - wT
    res["H3"] = within(stk, AP) - wT
    dom, rows = [], {"T>0.5": [], "G_proto": [], "Tmatch": []}
    for g in segs:
        m = seg_pt == g; yy = y[m]; t = T[m]; G = t * s_proto[m] > 0.5
        Pc, Rc, _ = precision_recall_curve(yy, t); Pg, Rg, _ = prf(G, yy)
        dom.append(Pc[np.nonzero(Rc >= Rg)[0].max()] - Pg)
        for key_, lab in (("T>0.5", t > 0.5), ("G_proto", G), ("Tmatch", tm[m])):
            rows[key_].append(prf(lab, yy))
    res["H4"] = np.array(dom)
    res["secondary"] = {
        "per_seg_AUPRC": {a: (per_seg(v, AP) - per_seg(T, AP)).round(4).tolist() for a, v in
                          {"T*S_proto": T * s_proto, "T*S_probe": T * S, "stack_dm": stk}.items()},
        "within_AUROC": {a: (within(v, AUC) - within(T, AUC)).round(4).tolist() for a, v in
                         {"T*S_proto": T * s_proto, "T*S_probe": T * S, "stack_dm": stk}.items()},
        "within_AUPRC_T": wT.round(4).tolist(),
        "op_point_PRF": {kk: np.array(v).mean(0).round(4).tolist() for kk, v in rows.items()},
        "Tmatch_minus_gate_dF1": (np.array(rows["Tmatch"])[:, 2] - np.array(rows["G_proto"])[:, 2]).round(4).tolist()}
    return res


def one_sided(d, direction):
    x = d if direction == ">" else -d
    return float(stats.wilcoxon(x, alternative="greater", method="exact").pvalue), int((x > 0).sum())


def test_phase():
    sel = json.load(open("ladder_primary.json"))
    print("selection:", sel["outcome"], "primary rung", sel["primary"])
    feats = build_features()
    report = {"selection": {"primary": sel["primary"], "outcome": sel["outcome"]}, "rungs": {}}
    for k in R:
        report["rungs"][k] = r = run_rung(k, sel, *feats)
        print(f"\nrung {k} (s_k {sel['rungs'][str(k)]['s_k']:.3f}): pooled", {a: (round(v['auprc'], 3), round(v['auroc'], 3)) for a, v in r["pooled"].items()})
        for h in ("H1", "H2", "H3", "H4"):
            print(f"  {h} within-chunk (H4: dominance) mean {r[h].mean():+.4f} per-seg {np.round(r[h], 3).tolist()}")
        print("  op point P/R/F1:", r["secondary"]["op_point_PRF"])
    if sel["primary"] is not None:
        r = report["rungs"][sel["primary"]]
        spec = {"H1": ("<", -0.02), "H2": (">", 0.01), "H3": (">", 0.01), "H4": (">", None)}
        ps = {h: one_sided(r[h], spec[h][0]) for h in spec}
        order = sorted(ps, key=lambda h: ps[h][0]); alpha = [0.05 / 4, 0.05 / 3, 0.05 / 2, 0.05]
        holm, stop = {}, False
        for i, h in enumerate(order):
            ok = (not stop) and ps[h][0] < alpha[i]
            mean_ok = spec[h][1] is None or (r[h].mean() <= spec[h][1] if spec[h][0] == "<" else r[h].mean() >= spec[h][1])
            holm[h] = dict(p=ps[h][0], n_dir=ps[h][1], mean=float(r[h].mean()), holm_alpha=alpha[i], pass_=bool(ok and mean_ok))
            if not ok:
                stop = True
        report["confirmatory"] = holm
        print("\nCONFIRMATORY (primary rung", sel["primary"], "):")
        for h in ("H1", "H2", "H3", "H4"):
            v = holm[h]; print(f"  {h}: mean {v['mean']:+.4f}  one-sided exact Wilcoxon p={v['p']:.4f}  {v['n_dir']}/8 in direction  Holm alpha {v['holm_alpha']:.4f}  -> {'PASS' if v['pass_'] else 'fail'}")
    json.dump(report, open("ladder_results.json", "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o))


if __name__ == "__main__":
    select() if "--select" in sys.argv else test_phase()
