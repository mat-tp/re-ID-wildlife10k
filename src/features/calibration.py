"""
Score calibration (WildFusion): raw score -> estimated probability that the pair shows the SAME individual.
Isotonic regression (monotone, step-like) followed by PCHIP interpolation through its knots (monotone, smooth).
Fitted on VALIDATION pairs only (never the test set).

Why: cosine of MegaDescriptor lives in ~[0.2, 0.9], EVA02 cosines are squeezed near 0.8-0.95, ALIKED gives an
integer inlier count 0..1000. A weighted sum of raw values would be dominated by whichever has the widest range.
"""
import numpy as np
from sklearn.isotonic import IsotonicRegression
from scipy.interpolate import PchipInterpolator


class ScoreCalibrator:
    def __init__(self, method="isotonic", tie_break=1e-4):
        assert method in ("isotonic", "pchip")
        self.method, self.tie = method, tie_break

    def fit(self, scores, same):
        x, y = np.asarray(scores, np.float64).ravel(), np.asarray(same, np.float64).ravel()
        iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(x, y)
        self.lo, self.hi = float(x.min()), float(x.max())
        self._iso = iso
        xt, yt = iso.X_thresholds_, iso.y_thresholds_
        self._pchip = PchipInterpolator(xt, yt) if len(xt) >= 2 else None
        self._const = float(yt[0]) if len(xt) < 2 else None
        return self

    def transform(self, scores):
        x = np.clip(np.asarray(scores, np.float64), self.lo, self.hi)
        if self._pchip is None:
            p = np.full(x.shape, self._const)
        elif self.method == "isotonic":
            p = self._iso.predict(x.ravel()).reshape(x.shape)
        else:                                              # isotonic knots + smooth monotone interpolation
            p = np.clip(self._pchip(x), 0.0, 1.0)
        # tiny monotone tilt so plateaus of the step function do not create exact ties in the fused score
        span = max(self.hi - self.lo, 1e-12)
        return (p + self.tie * (x - self.lo) / span).astype(np.float32)

    def reliability(self, scores, same, bins=15):
        """(mean raw score, empirical same-rate, count) per quantile bin - for a calibration plot."""
        s, y = np.asarray(scores).ravel(), np.asarray(same).ravel()
        edges = np.unique(np.quantile(s, np.linspace(0, 1, bins + 1)))
        b = np.clip(np.digitize(s, edges[1:-1]), 0, len(edges) - 2)
        return [(float(s[b == i].mean()), float(y[b == i].mean()), int((b == i).sum()))
                for i in range(len(edges) - 1) if (b == i).any()]


def expected_calibration_error(prob, same, bins=15):
    prob, same = np.asarray(prob).ravel(), np.asarray(same).ravel()
    b = np.clip((prob * bins).astype(int), 0, bins - 1)
    return float(sum(abs(prob[b == i].mean() - same[b == i].mean()) * (b == i).mean() for i in range(bins) if (b == i).any()))
