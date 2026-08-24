# WildlifeReID-10k Group Project

## Overview

This project investigates whether calibrated fusion of global (MiewID/MegaDescriptor), complementary (EVA02), and local (ALIKED) visual representations provides more reliable wildlife individual re-identification than any single representation alone on the WildlifeReID-10k benchmark.

---

## Group Members

Please include yourself:

1. Tshephang Matlala
2. Brilliant Mogaladi
3. Kamohelo Lebjane
4. \_

## Objectives

1. Re-Identification task
2. Open/Closed set evaluation

---

## Working with LaTeX

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
