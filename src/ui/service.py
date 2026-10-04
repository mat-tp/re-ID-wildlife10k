"""
Backend for the UI (no streamlit imports here, so it can be unit-tested and reused by a future API).

Query -> embed -> shortlist in the gallery -> (calibrate + fuse extra branches) -> top-k DISTINCT identities -> verdict.
Local (ALIKED) evidence is not wired into the UI yet (needs images + lightglue at query time) - see TODO in app.py.
"""
import json, os, pickle
import numpy as np
from src.retrieval.similarity import candidate_cosine
from src.retrieval.search import rank_by_embedding, Ranked, distinct_identities
from src.features.fusion import fuse, normalise_weights
from src.data_processing.splits import NEW


class ReIDService:
    def __init__(self, df, feats, results_dir="results", root="data", model_loader=None):
        self.df, self.feats, self.root = df, feats, root
        self.model_loader, self._models = model_loader, {}
        self.g_rows = df.index[df.split == "train"].to_numpy()          # the 'known animals' database
        self.g_ids = df.identity.to_numpy()[self.g_rows]
        j = os.path.join(results_dir, "thresholds.json"); c = os.path.join(results_dir, "calibrators.pkl")
        self.card = json.load(open(j)) if os.path.isfile(j) else {}
        self.cal = pickle.load(open(c, "rb")) if os.path.isfile(c) else {}

    def configs(self):
        return [tuple(k.split("+")) for k in self.card] or [(b,) for b in self.feats]

    def settings(self, branches):
        s = self.card.get("+".join(branches), {})
        return normalise_weights(s["weights"]) if "weights" in s else {b: 1 / len(branches) for b in branches}, s.get("tau")

    def embed_upload(self, image, branch):
        """Embed a PIL image with the live model of `branch` (needs torch + weights)."""
        if self.model_loader is None:
            raise RuntimeError("No model loader configured: pick an image from the dataset, or install torch and enable models.")
        import torch
        if branch not in self._models:
            self._models[branch] = self.model_loader(branch).eval()
        m = self._models[branch]
        with torch.no_grad():
            return m.embed(m.eval_transform()(np.asarray(image.convert("RGB"))).unsqueeze(0)).cpu().numpy()

    def search(self, query_vecs, branches, k=5, K=50, tau=None, exclude_row=None):
        """query_vecs: {branch: (1,d)}. Returns dict(verdict, best_score, hits[(identity, score, gallery_row)])."""
        branches = tuple(branches); w, tau_saved = self.settings(branches)
        tau = tau_saved if tau is None else tau
        sl = branches[0]
        r = rank_by_embedding(query_vecs[sl], self.feats[sl][self.g_rows], K + 1)
        if exclude_row is not None:                                      # query image itself lives in the gallery in 'pick' mode
            keep = self.g_rows[r.cand[0]] != exclude_row
            r = Ranked(r.cand[:, keep][:, :K], r.score[:, keep][:, :K])
        raw = {sl: r.score}
        for b in branches[1:]:
            raw[b] = candidate_cosine(query_vecs[b], self.feats[b][self.g_rows], r.cand)
        cal = {b: (self.cal[b].transform(raw[b]) if b in self.cal else raw[b]) for b in branches}
        fused = Ranked(r.cand, fuse(cal, w)); fused = fused.resort(fused.score)
        hits = [(i, s, int(self.g_rows[p])) for i, s, p in distinct_identities(fused, self.g_ids, k)[0]]
        # tau was tuned on the fused (calibrated) score scale; with a single branch that is still the raw-cosine scale only
        # if no calibrator is stored - the stored tau always refers to the same scale used in run_config().
        best = float(fused.score[0, 0])
        verdict = NEW if (tau is not None and best < tau) else hits[0][0]
        return {"verdict": verdict, "best_score": best, "tau": tau, "hits": hits, "weights": w}
