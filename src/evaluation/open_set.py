"""
This file implements open-set evaluation where:
- Some test identities are not present in training
- ID_test Not-Subset of ID_train
- System must identify known identities and reject unknown ones

Evaluation process:
1. Build gallery from training/known identities
2. For each query (known or unknown), compute similarity
3. Determine if query matches known identity (threshold-based)
4. Evaluate using BAKS and BAUS metrics

Key aspects:
- Important for real-world wildlife monitoring
- Populations are not static - new individuals appear
- Must reject sufficiently dissimilar queries

TODO: Implement open-set evaluation pipeline
TODO: Implement threshold determination for rejection
TODO: Compute BAKS and BAUS metrics
TODO: Evaluate per-species open-set performance
TODO: Analyze performance on unseen species
"""
