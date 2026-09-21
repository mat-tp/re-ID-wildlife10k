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


# Top-1 acuracy is the standard measure of correctness where a model's single guess is ranked
# the highest probability must exactly match the ground truth label for the prediction to be considered correct.
def top_k_accuracy(predictions, labels, k=1):
      """
      Computes the Top-K accuracy for the given predictions and labels.
      
      Args:
         predictions (torch.Tensor): Model output probabilities or logits of shape (N, C)
         labels (torch.Tensor): Ground truth labels of shape (N,)
         k (int): The 'k' in Top-K accuracy. Default is 1 for Top-1 accuracy.
         
      Returns:
         float: Top-K accuracy as a percentage.
         
      """