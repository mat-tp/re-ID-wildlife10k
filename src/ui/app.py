"""
WildlifeReID UI  -  run from the project root:  streamlit run src/ui/app.py
Tabs: Re-identify (search + show results) | Results (ablation tables) | Train (TODO)
Works without a GPU/images in DEMO mode (synthetic embeddings on the real metadata) so the layout can be developed anywhere.
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))   # project root, so `import src...` works
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

import src.config as C
from src.data_processing.splits import get_frames, NEW
from src.ui.service import ReIDService

st.set_page_config(page_title="Wildlife Re-ID", layout="wide")


@st.cache_resource(show_spinner="Loading metadata and features ...")
def load_service(mode, cache_root, results_dir):
    df = get_frames(C.data)
    if mode == "Demo (synthetic features)":
        from src.demo import make_demo_features
        df = df.sample(frac=0.10, random_state=0).sort_index().reset_index(drop=True)   # small, fast
        feats = make_demo_features(df)
    else:
        from src.features.extraction import load_features
        feats = load_features(cache_root, {"global": "megadescriptor", "complementary": "eva02", "baseline": "dinov2"})
    return ReIDService(df, feats, results_dir, C.data["dataset_root"], model_loader=_model_loader if mode != "Demo (synthetic features)" else None)


def _model_loader(branch):
    from src.models.global_models import GlobalModel
    from src.models.complementary_models import ComplementaryModel
    return GlobalModel(C.models["global"]["variant"], C.models["global"]["init"]) if branch == "global" else ComplementaryModel()


def thumb(root, rel_path, width=160):
    f = os.path.join(root, rel_path)
    if os.path.isfile(f):
        st.image(Image.open(f), width=width)
    else:
        st.caption("(image file not found)")


#  sidebar
with st.sidebar:
    st.header("Setup")
    mode = st.radio("Data source", ["Demo (synthetic features)", "Real cached features"])
    cache_root = st.text_input("Feature cache folder", C.retrieval["feature_cache"])
    results_dir = st.text_input("Results folder", C.retrieval["results_dir"])
    try:
        svc = load_service(mode, cache_root, results_dir)
    except Exception as e:                                      # e.g. no cache yet
        st.error(f"Could not load: {e}"); st.stop()
    cfg = st.selectbox("Method (branches fused)", svc.configs(), format_func=lambda c: " + ".join(c))
    cfg = tuple(b for b in cfg if b in svc.feats)               # local not wired into the UI yet
    if not cfg:
        st.warning("Selected method needs the local branch, which the UI does not support yet."); st.stop()
    w, tau_saved = svc.settings(cfg)
    k = st.slider("Identities to show", 1, 10, 5)
    use_saved = st.checkbox("Use threshold tau tuned on validation", value=tau_saved is not None, disabled=tau_saved is None)
    tau = tau_saved if use_saved else st.slider("Manual tau (raw/fused score)", 0.0, 1.0, 0.5, 0.01)
    st.caption("Weights: " + ", ".join(f"{b}={v:.2f}" for b, v in w.items()))

tab_id, tab_res, tab_train = st.tabs(["Re-identify", "Results", "Train"])

#  re-identify
with tab_id:
    st.subheader("Which individual is this?")
    src = st.radio("Query image", ["Pick from the benchmark test set", "Upload an image"], horizontal=True)
    qvec, qinfo, exclude = None, None, None
    if src.startswith("Pick"):
        test = svc.df[svc.df.split == "test"]
        c1, c2 = st.columns(2)
        ds = c1.selectbox("Dataset", sorted(test.dataset.unique()))
        pool = test[test.dataset == ds]
        row = c2.selectbox("Image", pool.index[:500], format_func=lambda r: f"{svc.df.path[r].split('/')[-1]}  (row {r})")
        qvec = {b: svc.feats[b][[row]] for b in cfg}
        qinfo = svc.df.loc[row]
        left, right = st.columns([1, 3]); 
        with left:
            thumb(svc.root, qinfo.path, 220)
    else:
        up = st.file_uploader("Image of an animal", type=["jpg", "jpeg", "png"])
        if up:
            img = Image.open(up); st.image(img, width=220)
            try:
                qvec = {b: svc.embed_upload(img, b) for b in cfg}
            except Exception as e:
                st.error(f"Cannot embed uploaded image: {e}")
    if qvec is not None and st.button("Search", type="primary"):
        out = svc.search(qvec, cfg, k=k, tau=tau)
        if out["verdict"] == NEW:
            st.warning(f"**New individual** (best score {out['best_score']:.3f} < tau {out['tau']:.3f})")
        else:
            st.success(f"**Predicted: {out['verdict']}**  (best score {out['best_score']:.3f}"
                       + (f", tau {out['tau']:.3f})" if out["tau"] is not None else ")"))
        if qinfo is not None:
            known = qinfo.identity in set(svc.g_ids)
            st.caption(f"Ground truth (for checking): {qinfo.identity if known else 'NEW - not in the database'}")
        cols = st.columns(len(out["hits"]))
        for col, (ident, sc, grow) in zip(cols, out["hits"]):
            with col:
                st.markdown(f"**{ident}**"); st.caption(f"score {sc:.3f}")
                thumb(svc.root, svc.df.path[grow], 150)

#  results
with tab_res:
    st.subheader("Ablation results")
    f = os.path.join(results_dir, "ablation_val.csv")
    if os.path.isfile(f):
        t = pd.read_csv(f); st.dataframe(t, use_container_width=True)
        cols = [c for c in t.columns if c.endswith(("BAKS", "BAUS", "NBA"))]
        st.bar_chart(t.set_index("config")[cols])
    else:
        st.info("No results yet. Run the ablation (see README 'Running experiments') to create results/ablation_val.csv.")

#  train
with tab_train:
    st.subheader("Fine-tune MegaDescriptor")
    
    # 1. The Form
    c1, c2, c3 = st.columns(3)
    variant = c1.selectbox("Backbone Variant", ["T-224", "B-224", "L-384"], index=0)
    init = c2.selectbox("Initialization", ["megadescriptor", "imagenet"], index=1)
    epochs = c3.number_input("Epochs", min_value=1, value=12)
    
    c4, c5, c6 = st.columns(3)
    lr = c4.number_input("Learning Rate", min_value=0.0, value=1e-3, format="%.4f")
    batch_size = c5.number_input("Micro Batch Size", min_value=1, value=16)
    margin = c6.number_input("ArcFace Margin", min_value=0.0, value=0.5)
    
    if st.button("Start training", type="primary"):
        st.info("Training initiated... (Waiting for logs)")
        
        # Placeholders for real-time UI updates
        log_container = st.empty()
        chart_container = st.empty()
        val_container = st.empty()
        
        # Setup threading and queue
        import threading
        import queue
        import contextlib
        import io
        import re
        
        log_queue = queue.Queue()
        
        class QueueWriter(io.StringIO):
            def write(self, s):
                if s.strip():
                    log_queue.put(s.strip())
                return len(s)
                
        def run_training():
            # Dynamically override the config
            C.models["global"]["variant"] = variant
            C.models["global"]["init"] = init
            C.training["epochs"] = epochs
            C.training["lr"] = lr
            C.training["micro_batch"] = batch_size
            C.training["arcface_margin"] = margin
            
            # Run the actual experiment script and intercept prints
            with contextlib.redirect_stdout(QueueWriter()):
                try:
                    from src.models.run_global_experiments import main
                    main("fine_tune", "megadescriptor")
                except Exception as e:
                    log_queue.put(f"ERROR: {str(e)}")
        
        thread = threading.Thread(target=run_training)
        thread.start()
        
        # Real-time listener loop
        logs, losses, vals = [], [], []
        import time
        while thread.is_alive() or not log_queue.empty():
            while not log_queue.empty():
                msg = log_queue.get()
                logs.append(msg)
                
                # Parse Loss
                loss_match = re.search(r"loss (\d+\.\d+)", msg)
                if loss_match:
                    losses.append(float(loss_match.group(1)))
                    chart_container.line_chart(losses)
                
                # Parse Validation
                if "validation score" in msg:
                    val_match = re.search(r"validation score: (\d+\.\d+)", msg)
                    if val_match:
                        vals.append(float(val_match.group(1)))
                        val_container.dataframe(pd.DataFrame({"Validation NBA": vals}))
                
                # Keep logs reasonably small
                log_container.code("\n".join(logs[-10:]))
                
            time.sleep(0.1)
            
        st.success("Training background process concluded.")
