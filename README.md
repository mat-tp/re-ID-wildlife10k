# WildlifeReID-10k Group Project

## Overview

This project investigates whether calibrated fusion of global (MiewID/MegaDescriptor), complementary (EVA02), and local (ALIKED) visual representations provides more reliable wildlife individual re-identification than any single representation alone on the WildlifeReID-10k benchmark.

---

## Group Name: MLM

## Group Members

Please include yourself:

1. Tshephang Matlala
2. Brilliant Mogaladi
3. Kamohelo Lebjane
4. Denzel Zimba

## Objectives

1. Re-Identification task:

- **Classification** asks "what species is this?". **Re-ID** asks "which individual animal is this?" among animals you have already seen.

2. Open/Closed set evaluation:

   **Closed set:** the animal is guaranteed to be in the gallery, so the best match is the answer.
   **Open set:** the animal may be new. We add a threshold **tau**: if the best score is below tau, answer "new individual".

3. How the system works:
   Every image becomes a _score_ against every image in a **gallery** (the database of known animals).
   The best-scoring gallery image gives the predicted identity (1-nearest-neighbour).

```
image -> embeddings (global, EVA02)  --cosine-->  shortlist of K gallery images
                                                      |
              ALIKED + LightGlue on the K_local best only
                                                      v
                     calibrate each score -> weighted sum -> re-rank
                                                      v
              best gallery image: identity + score  ->  score >= tau ? identity : NEW
```

I will fix this diagram to show properly

4. Feature vectors:

Global vector = overall look. EVA02 vector = generic visual features. ALIKED = matching of tiny local
patterns (spots, scars). Their scores live on different scales, so each is **calibrated** to "probability this is the same animal"
and then combined with a weighted sum.

Plain accuracy would be dominated by the big ones. BAKS averages per image -> per animal -> per dataset, so every animal and every dataset counts equally.

## Working with LaTeX (I should actually remove this..)

I've provided a command to compile LaTeX files (`.tex`) and the bibliography (`.bib`):

```bash
latexmk -pdf main.tex
```

## Source code (src)

I've provided the initial skeleton. Packages include

1. data_processing: Dataset loading & preprocessing (WildlifeReID-10k)
2. models: Global, Local, and Complementary models
3. features: Extraction, Calibration and Fusion
4. retrieval: Similarity, Search, and Metrics (BAKS/ BAUS)
5. evaluation: Closed-set & Open-set protocols

## Dependencies

I've recommended some, feel free to add if you found a different one.

```bash
pip install -r requirements.txt
```

I will actually create a docker file, which can make things easier.

## Running experiments (added)

```bash
pip install -r requirements.txt            # + lightglue (pip install git+https://github.com/cvg/LightGlue), streamlit
python -m src.tests.test_global_pipeline            # splits / metrics / PCA   (no torch needed)
python -m src.tests.test_retrieval_pipeline         # calibration / fusion / ablation on synthetic features
python -m src.run_experiments extract global   # then miewid, complementary, baseline  (resumable)
python -m src.run_experiments ablation --no-local
python -m src.run_experiments ablation --test  # ONLY when everything is final
streamlit run src/ui/app.py                       # UI (demo mode works without GPU or images)
```

## References
