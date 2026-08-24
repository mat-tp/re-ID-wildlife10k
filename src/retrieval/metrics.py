"""
This file implements metrics for:
1. Closed-set evaluation:
   - Top-1, Top-5, Top-10 accuracy
   - Mean Average Precision (mAP)
   - Balanced metrics

2. Open-set evaluation:
   - BAKS (Balanced Accuracy for Known Subjects)
   - BAUS (Balanced Accuracy for Unknown Subjects)
   - Open-set classification metrics

Key aspects:
- Balanced metrics account for class imbalance
- BAKS and BAUS from WildlifeReID-10k benchmark
- Some identities have many images, others very few

Reference: WildlifeReID-10k evaluation protocols

TODO: Implement standard retrieval metrics (accuracy, mAP)
TODO: Implement BAKS metric
TODO: Implement BAUS metric
TODO: Implement per-species evaluation
TODO: Implement per-dataset evaluation
TODO: Add confidence intervals for metrics
"""
