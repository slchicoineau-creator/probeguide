"""Judge gate checks (b) and (c) on the Section 3c points (wt_points.npz, teacher = iteration-0, direction picked per fold
on training segments). Runs in $PROBEGUIDE_DATA.
(b) T*S_proto - T, T*S_probe - T, and the frozen soft stack (agent 08 / 01 LO2: LR on [logit T, logit S_probe], S cross-fit
    on training segments) vs T: per-segment AUPRC/AUROC and WITHIN-CHUNK (rule fixed before looking: chunk counts if it has
    >= 50 points of each class). Controls: stack on [logit T, chunk ct_mean] (label-free, no DINO) and on [logit T, rich flag]
    (design-leak control, not deployable).
(c) operating point at 0.5: P/R/F1, TP-kept/FP-kept, oracle dominance margin (max P of T at recall >= gated recall on the
    TEST segment), non-max version, deployable Tmatch (train-fold threshold matching the gated POSITIVE RATE, label-free),
    Tmatch-by-recall (uses train labels), Jaccard(gate, Tmatch), Platt(T) at 0.5 fitted on training pixels."""
import _env  # path setup (VILLA_SRC, PROBEGUIDE_ASSETS, PROBEGUIDE_DATA); written by the pre-build judge gate
import glob, numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score as AP, roc_auc_score as AUC, precision_recall_curve

MINC = 50
W = np.load("wt_points.npz", allow_pickle=True)
seg_pt, ch, xyz, y = W["seg"].astype(str), W["chunk"].astype(int), W["xyz"], W["ink"].astype(int)
ref = np.load(f"{_env.ASSETS}/avg_ref_embedding.npy").astype(np.float32); ref /= np.linalg.norm(ref)
files = sorted(glob.glob("emb/chunk_*.npz"))
tok_id = np.zeros(len(y), int); feats = []; sup_rows = []; off = 0; s_proto = np.zeros(len(y)); ctm = np.zeros(len(y))
rich = np.zeros(len(y)); segc = []
for n, f in enumerate(files):
    d = np.load(f); segc.append(str(d["seg"])); m = ch == n
    E = d["emb"].astype(np.float32); E /= np.linalg.norm(E, axis=-1, keepdims=True) + 1e-9
    cos = E @ ref; inc = (cos - cos.min()) / max(cos.max() - cos.min(), 1e-6)
    tk = np.clip(((xyz[m] - np.asarray(d["origin_xyz"])) // 8).astype(int), 0, 31)
    pts = np.stack([tk[:, 2], tk[:, 1], tk[:, 0]], 1) if m.any() else np.zeros((0, 3), int)
    s_proto[m] = inc[pts[:, 0], pts[:, 1], pts[:, 2]]; ctm[m] = float(d["ct_mean"]); rich[m] = float(str(d["kind"]) == "rich")
    sup = np.argwhere(d["sup"] > 0)
    allk = np.unique(np.concatenate([sup, pts]), axis=0); key = {tuple(k): i for i, k in enumerate(allk)}
    feats.append(E[allk[:, 0], allk[:, 1], allk[:, 2]])
    if m.any():
        tok_id[m] = off + np.array([key[tuple(k)] for k in pts])
    for k in sup:
        sup_rows.append((off + key[tuple(k)], int(d["ink"][tuple(k)]), str(d["seg"])))
    off += len(allk)
F = np.concatenate(feats); del feats
sup_idx = np.array([r[0] for r in sup_rows]); sup_y = np.array([r[1] for r in sup_rows]); sup_seg = np.array([r[2] for r in sup_rows])
segs = sorted(set(segc)); folds = [segs[i::4] for i in range(4)]
lg = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))


def probe(train_segs):
    mm = np.isin(sup_seg, train_segs)
    return LogisticRegression(C=0.5, max_iter=3000, class_weight="balanced").fit(F[sup_idx[mm]], sup_y[mm])


T = np.zeros(len(y)); S = np.zeros(len(y)); stk = np.zeros(len(y)); c_ct = np.zeros(len(y)); c_rich = np.zeros(len(y))
platt = np.zeros(len(y)); tm_rate = np.zeros(len(y), bool); tm_rec = np.zeros(len(y), bool); tmp_rate = np.zeros(len(y), bool)
info = []
for fi, test in enumerate(folds):
    train = [s for s in segs if s not in test]; te = np.isin(seg_pt, test); tr = ~te
    a = {k: AUC(y[tr], W[k][tr]) for k in ("t_fwd", "t_rev")}; pick = max(a, key=a.get)
    Tf = W[pick].astype(np.float64); T[te] = Tf[te]
    Sin = np.zeros(len(y))
    for s in train:
        mm = seg_pt == s; Sin[mm] = probe([t for t in train if t != s]).predict_proba(F[tok_id[mm]])[:, 1]
    S[te] = probe(train).predict_proba(F[tok_id[te]])[:, 1]
    st = LogisticRegression(max_iter=1000).fit(np.c_[lg(Tf[tr]), lg(Sin[tr])], y[tr])
    stk[te] = st.decision_function(np.c_[lg(Tf[te]), lg(S[te])])
    c1 = LogisticRegression(max_iter=1000).fit(np.c_[lg(Tf[tr]), ctm[tr]], y[tr]); c_ct[te] = c1.decision_function(np.c_[lg(Tf[te]), ctm[te]])
    c2 = LogisticRegression(max_iter=1000).fit(np.c_[lg(Tf[tr]), rich[tr]], y[tr]); c_rich[te] = c2.decision_function(np.c_[lg(Tf[te]), rich[te]])
    pl = LogisticRegression(max_iter=1000).fit(lg(Tf[tr])[:, None], y[tr]); platt[te] = pl.predict_proba(lg(Tf[te])[:, None])[:, 1]
    # deployable Tmatch: threshold on T whose TRAIN positive rate equals the gated train positive rate (label-free)
    g_tr = (Tf[tr] * s_proto[tr]) > 0.5
    tau = np.quantile(Tf[tr], 1 - g_tr.mean()); tm_rate[te] = Tf[te] > tau
    rec_g = g_tr[y[tr] == 1].mean(); tau_r = np.quantile(Tf[tr][y[tr] == 1], 1 - rec_g); tm_rec[te] = Tf[te] > tau_r
    # probe gate on training segs uses cross-fit Sin
    gp_tr = (Tf[tr] * Sin[tr]) > 0.5; tau_p = np.quantile(Tf[tr], 1 - gp_tr.mean()); tmp_rate[te] = Tf[te] > tau_p
    info.append((fi, pick, round(a[pick], 3), round(tau, 3), round(tau_r, 3), round(tau_p, 3), round(st.coef_[0][1] / st.coef_[0][0], 2)))
print("fold, pick, train AUROC, tau_rate(proto), tau_recall(proto), tau_rate(probe), stack gamma:", info)


def per_seg(s, met):
    return np.array([met(y[seg_pt == g], s[seg_pt == g]) for g in segs])


def within(s, met):
    out, nch = [], []
    for g in segs:
        v = [met(y[(seg_pt == g) & (ch == c)], s[(seg_pt == g) & (ch == c)]) for c in np.unique(ch[seg_pt == g])
             if (y[(seg_pt == g) & (ch == c)] == 1).sum() >= MINC and (y[(seg_pt == g) & (ch == c)] == 0).sum() >= MINC]
        out.append(np.mean(v)); nch.append(len(v))
    return np.array(out), nch


def test(d):
    pos = int((d > 0).sum())
    return f"mean {d.mean():+.3f} [{', '.join(f'{x:+.3f}' for x in d)}] t p={stats.ttest_1samp(d, 0).pvalue:.3f} W p={stats.wilcoxon(d).pvalue:.3f} {pos}/8"


arms = {"T*S_proto": T * s_proto, "T*S_probe": T * S, "stack[T,S_probe]": stk, "ctrl[T,ct_mean]": c_ct, "ctrl[T,richflag]": c_rich}
print("pooled T AUPRC/AUROC", round(AP(y, T), 3), round(AUC(y, T), 3), "| within-chunk qualifying chunks:", within(T, AUC)[1])
for k, s in arms.items():
    for mn, met in (("AUPRC", AP), ("AUROC", AUC)):
        print(f"{k:17s} - T {mn} per-seg : {test(per_seg(s, met) - per_seg(T, met))}")
        print(f"{k:17s} - T {mn} within  : {test(within(s, met)[0] - within(T, met)[0])}")

# ---------- (c) operating point ----------
def prf(lab, g):
    tp = (lab & (g == 1)).sum(); P = tp / max(lab.sum(), 1); R = tp / max((g == 1).sum(), 1)
    return P, R, 2 * P * R / max(P + R, 1e-9)


rows = {k: [] for k in ("T>0.5", "G_proto", "G_probe", "Tmatch_rate(proto)", "Tmatch_recall(proto)", "Tmatch_rate(probe)", "Platt>0.5")}
dom_or, dom_nm, dom_or_p, keep, jac, jacp = [], [], [], [], [], []
for g in segs:
    m = seg_pt == g; yy = y[m]; t = T[m]
    lab = {"T>0.5": t > 0.5, "G_proto": t * s_proto[m] > 0.5, "G_probe": t * S[m] > 0.5, "Tmatch_rate(proto)": tm_rate[m],
           "Tmatch_recall(proto)": tm_rec[m], "Tmatch_rate(probe)": tmp_rate[m], "Platt>0.5": platt[m] > 0.5}
    for k, v in lab.items():
        rows[k].append(prf(v, yy) + (v.mean(),))
    Pc, Rc, th = precision_recall_curve(yy, t)
    for G, lst_or, lst_nm in ((lab["G_proto"], dom_or, dom_nm), (lab["G_probe"], dom_or_p, None)):
        Pg, Rg, _ = prf(G, yy); ok = Rc >= Rg
        lst_or.append(Pc[ok].max() - Pg)
        if lst_nm is not None:
            lst_nm.append(Pc[np.nonzero(ok)[0].max()] - Pg)  # highest threshold that still reaches the gated recall
    tpm = (t > 0.5) & (yy == 1); fpm = (t > 0.5) & (yy == 0); G = lab["G_proto"]
    keep.append((G[tpm].mean(), G[fpm].mean()))
    A, B = lab["G_proto"], lab["Tmatch_rate(proto)"]; jac.append((A & B).sum() / max((A | B).sum(), 1))
    A, B = lab["G_probe"], lab["Tmatch_rate(probe)"]; jacp.append((A & B).sum() / max((A | B).sum(), 1))
print("\nper-segment mean P / R / F1 / positive rate")
for k, v in rows.items():
    v = np.array(v); print(f"  {k:22s} " + " / ".join(f"{x:.3f}" for x in v.mean(0)))
keep = np.array(keep); print("proto gate keeps TP / FP of T>0.5 (seg means):", keep.mean(0).round(3), "TP-FP kept", test(keep[:, 0] - keep[:, 1]))
print("oracle dominance proto  :", test(np.array(dom_or)))
print("non-max dominance proto :", test(np.array(dom_nm)))
print("oracle dominance probe  :", test(np.array(dom_or_p)))
R = {k: np.array(v) for k, v in rows.items()}
for a, b in (("Tmatch_rate(proto)", "G_proto"), ("Tmatch_recall(proto)", "G_proto"), ("Tmatch_rate(probe)", "G_probe"),
             ("Platt>0.5", "G_proto"), ("Platt>0.5", "T>0.5"), ("G_proto", "T>0.5")):
    for j, mn in ((0, "P"), (1, "R"), (2, "F1")):
        print(f"  {a} - {b} d{mn}: {test(R[a][:, j] - R[b][:, j])}")
print("Jaccard(G_proto, Tmatch_rate):", np.round(jac, 3).tolist(), "mean", round(float(np.mean(jac)), 3))
print("Jaccard(G_probe, Tmatch_rate(probe)):", np.round(jacp, 3).tolist(), "mean", round(float(np.mean(jacp)), 3))
