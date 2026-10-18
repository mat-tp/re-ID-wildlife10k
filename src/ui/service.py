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
        self.local_scorer = None

    def configs(self):
        saved = [tuple(k.split("+")) for k in self.card]
        options = list(saved)
        # Ensure multi-branch fusion options including local are available
        all_options = [
            ("global", "complementary", "local"),
            ("global", "complementary"),
            ("global", "local"),
            ("global",),
            ("complementary",),
            ("baseline",)
        ]
        for opt in all_options:
            if all(b == "local" or b in self.feats for b in opt) and opt not in options:
                options.append(opt)
        return options or [(b,) for b in self.feats]

    def settings(self, branches):
        s = self.card.get("+".join(branches), {})
        if "weights" in s:
            return normalise_weights(s["weights"]), s.get("tau")
        if tuple(branches) == ("global", "complementary", "local"):
            return {"global": 0.45, "complementary": 0.35, "local": 0.20}, s.get("tau", 0.45)
        if tuple(branches) == ("global", "local"):
            return {"global": 0.70, "local": 0.30}, s.get("tau", 0.50)
        return {b: 1.0 / len(branches) for b in branches}, s.get("tau")

    def _get_local_scorer(self):
        if self.local_scorer is None:
            from src.retrieval.similarity import LocalScorer
            from src.data_processing.dataloader import convert_to_rgb_image
            self.local_scorer = LocalScorer(lambda r: convert_to_rgb_image(self.df.path.iloc[r], self.root), use_ransac=False)
        return self.local_scorer

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

    def search(self, query_vecs, branches, k=5, K=50, tau=None, exclude_row=None, query_img=None, query_row=None):
        """query_vecs: {branch: (1,d)}. Returns dict(verdict, best_score, hits[(identity, score, gallery_row)])."""
        branches = tuple(branches)
        w, tau_saved = self.settings(branches)
        tau = tau_saved if tau is None else tau

        vec_branches = [b for b in branches if b != "local"]
        sl = vec_branches[0] if vec_branches else ("global" if "global" in self.feats else next(iter(self.feats)))
        
        K_search = max(K, 20) if "local" in branches else K
        r = rank_by_embedding(query_vecs[sl], self.feats[sl][self.g_rows], K_search + 1)
        if exclude_row is not None:
            keep = self.g_rows[r.cand[0]] != exclude_row
            r = Ranked(r.cand[:, keep][:, :K_search], r.score[:, keep][:, :K_search])

        raw = {sl: r.score}
        for b in vec_branches:
            if b != sl:
                raw[b] = candidate_cosine(query_vecs[b], self.feats[b][self.g_rows], r.cand)

        # Local branch re-ranking on the top K_local candidate shortlist
        if "local" in branches:
            k_local = min(10, r.cand.shape[1])
            cand_eval = r.cand[:, :k_local]
            for b in vec_branches:
                raw[b] = raw[b][:, :k_local]

            scorer = self._get_local_scorer()
            if query_row is not None:
                q_feats = scorer._feats(int(query_row))
            elif query_img is not None:
                q_feats = scorer.extract_image(query_img)
            else:
                raise ValueError("Local branch requires query_img or query_row.")

            raw_local = np.empty((1, k_local), dtype=np.float32)
            for j in range(k_local):
                gr = self.g_rows[cand_eval[0, j]]
                cand_feats = scorer._feats(int(gr))
                raw_local[0, j] = scorer.score_features_pair(q_feats, cand_feats)
            raw["local"] = raw_local
        else:
            cand_eval = r.cand

        # Calibrate raw scores into probability scale
        cal = {}
        for b in branches:
            if b in self.cal:
                cal[b] = self.cal[b].transform(raw[b])
            elif b == "local":
                # Normalize raw match counts to [0, 1] pseudo-probabilities
                cal[b] = np.clip(raw[b] / 30.0, 0.0, 1.0)
            else:
                cal[b] = np.clip(raw[b], 0.0, 1.0)

        fused = Ranked(cand_eval, fuse(cal, w))
        fused = fused.resort(fused.score)
        hits = [(i, s, int(self.g_rows[p])) for i, s, p in distinct_identities(fused, self.g_ids, k)[0]]
        best = float(fused.score[0, 0])
        verdict = NEW if (tau is not None and best < tau) else hits[0][0]
        return {"verdict": verdict, "best_score": best, "tau": tau, "hits": hits, "weights": w}
