"""Classifiers and evaluation helpers used by the app.

Each classifier works directly on pandas frames that hold a mix of numeric and
categorical columns, so the Flask routes never deal with encoding details.
To add an algorithm (e.g. Ho-Kashyap): write a class here, then register it in
ALGORITHMS in app.py.
"""

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.special import logsumexp
from sklearn.naive_bayes import CategoricalNB, GaussianNB
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OrdinalEncoder


# --------------------------------------------------------------------------
# Shared handling of mixed numeric + categorical columns
# --------------------------------------------------------------------------
class _MixedFeatures:
    """Missing values are filled with the *training* median / most frequent value,
    and numeric columns are z-scored with training statistics."""

    def __init__(self, numeric, categorical):
        self.numeric = list(numeric)
        self.categorical = list(categorical)

    def _numeric_frame(self, X):
        return X[self.numeric].astype(float).replace([np.inf, -np.inf], np.nan)

    def _categorical_frame(self, X):
        out = {}
        for col in self.categorical:
            series = X[col]
            filled = series.astype(object).where(series.notna(), self.cat_mode_[col])
            out[col] = filled.map(str)
        return pd.DataFrame(out, index=X.index)

    def _fit_features(self, X):
        if self.numeric:
            data = self._numeric_frame(X)
            self.num_median_ = data.median().fillna(0.0)
            data = data.fillna(self.num_median_)
            self.num_mean_ = data.mean()
            self.num_std_ = data.std(ddof=0).replace(0.0, 1.0).fillna(1.0)
        self.cat_mode_ = {}
        for col in self.categorical:
            mode = X[col].dropna().astype(str).mode()
            self.cat_mode_[col] = mode.iloc[0] if len(mode) else "(missing)"

    def _scaled_numeric(self, X):
        data = self._numeric_frame(X).fillna(self.num_median_)
        return ((data - self.num_mean_) / self.num_std_).to_numpy()


# --------------------------------------------------------------------------
# Naive Bayes
# --------------------------------------------------------------------------
class MixedNaiveBayes(_MixedFeatures):
    """Naive Bayes that accepts numeric and categorical columns together.

    * numeric features     -> GaussianNB (z-scored so var_smoothing is scale-free)
    * categorical features -> CategoricalNB (Laplace smoothing, reserved slot for
                              categories never seen in training)

    Both sub-models return a normalised log-posterior that includes the class
    prior. Adding them double-counts the prior, so it is subtracted once:
        log P(y|x) = log P(y|x_num) + log P(y|x_cat) - log P(y)   (up to a constant)
    which is exactly the naive-Bayes factorisation over all features.
    """

    def fit(self, X, y):
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self._fit_features(X)

        if self.numeric:
            self.gnb_ = GaussianNB().fit(self._scaled_numeric(X), y)

        if self.categorical:
            self.encoder_ = OrdinalEncoder(
                handle_unknown="use_encoded_value", unknown_value=-1, dtype=np.int64
            )
            codes = self.encoder_.fit_transform(self._categorical_frame(X))
            self.cardinality_ = [len(levels) for levels in self.encoder_.categories_]
            # One extra slot per feature for categories first seen at prediction time.
            self.cnb_ = CategoricalNB(
                min_categories=[k + 1 for k in self.cardinality_]
            ).fit(codes, y)
        return self

    def _log_posterior(self, X):
        parts = []
        if self.numeric:
            parts.append(self.gnb_.predict_log_proba(self._scaled_numeric(X)))
        if self.categorical:
            codes = self.encoder_.transform(self._categorical_frame(X))
            for j, cardinality in enumerate(self.cardinality_):
                codes[codes[:, j] < 0, j] = cardinality  # unseen category -> reserved slot
            parts.append(self.cnb_.predict_log_proba(codes))

        if len(parts) == 2:
            log_post = parts[0] + parts[1] - np.log(self.gnb_.class_prior_)
        else:
            log_post = parts[0]
        return log_post - logsumexp(log_post, axis=1, keepdims=True)

    def predict_proba(self, X):
        return np.exp(self._log_posterior(X))

    def predict(self, X):
        return self.classes_[np.argmax(self._log_posterior(X), axis=1)]


# --------------------------------------------------------------------------
# Shared distance-space encoding (K-nearest neighbours and Parzen window)
# --------------------------------------------------------------------------
class _DistanceSpace(_MixedFeatures):
    """Encoding for methods that compare records by distance.

    * numeric features     -> median-filled, then z-scored
    * categorical features -> most-frequent-filled, then one-hot encoded. Each
      feature gets one extra all-zero column in training that lights up for a
      category first seen at prediction time, so an unseen category counts as a
      mismatch against every training record.
    """

    def _fit_levels(self, X):
        frame = self._categorical_frame(X)
        self.cat_levels_ = {
            col: {value: i for i, value in enumerate(sorted(frame[col].unique()))}
            for col in self.categorical
        }

    def _encode(self, X):
        blocks = []
        if self.numeric:
            blocks.append(self._scaled_numeric(X))
        if self.categorical:
            frame = self._categorical_frame(X)
            for col in self.categorical:
                levels = self.cat_levels_[col]
                n_levels = len(levels)
                codes = frame[col].map(levels).fillna(n_levels).astype(int).to_numpy()
                blocks.append(np.eye(n_levels + 1)[codes])
        return np.hstack(blocks)


# --------------------------------------------------------------------------
# K-nearest neighbours
# --------------------------------------------------------------------------
class KNNClassifier(_DistanceSpace):
    """K-nearest neighbours on mixed features, with Euclidean distance.

    * numeric features     -> median-filled, then z-scored
    * categorical features -> most-frequent-filled, then one-hot encoded. Each
      feature gets one extra all-zero column in training that lights up for a
      category first seen at prediction time, so an unseen category counts as a
      mismatch against every training record.

    k is not fixed at fit time: fitting only stores the training points, and
    every query chooses its own k. That lets one neighbour search answer many
    values of k (see predict_indices).

    Ties between classes go to the class that sorts first.
    """

    def fit(self, X, y):
        y = np.asarray(y, dtype=str)
        self.classes_ = np.unique(y)
        self.train_class_idx_ = np.searchsorted(self.classes_, y)
        self.n_train_ = len(y)

        self._fit_features(X)
        self._fit_levels(X)
        encoded = self._encode(X)
        self.n_dimensions_ = encoded.shape[1]
        self.nn_ = NearestNeighbors(metric="euclidean").fit(encoded)
        return self

    def predict_indices(self, X, ks):
        """Predicted class index for every row of X, for each k in ks.

        One neighbour search (for the largest k) serves all values of k: the
        neighbours are sorted nearest-first, so the first k of them are the k
        nearest. Returns {k: array of class indices into classes_}.
        """
        wanted = sorted(set(int(k) for k in ks))
        _, neighbours = self.nn_.kneighbors(self._encode(X), n_neighbors=wanted[-1])
        neighbour_class = self.train_class_idx_[neighbours]

        rows = np.arange(len(X))
        votes = np.zeros((len(X), len(self.classes_)), dtype=np.int32)
        out = {}
        for j in range(wanted[-1]):
            votes[rows, neighbour_class[:, j]] += 1
            if (j + 1) in wanted:
                out[j + 1] = votes.argmax(axis=1)  # ties -> lowest class index
        return out

    def vote_counts(self, X, k):
        """How many of the k nearest training records belong to each class,
        for a single-row frame X. Returns an array aligned with classes_."""
        _, neighbours = self.nn_.kneighbors(self._encode(X), n_neighbors=int(k))
        return np.bincount(self.train_class_idx_[neighbours[0]], minlength=len(self.classes_))


# --------------------------------------------------------------------------
# Parzen window
# --------------------------------------------------------------------------
PARZEN_KERNELS = ("gaussian", "epanechnikov", "hypercube")


class ParzenWindowClassifier(_DistanceSpace):
    """Parzen window (kernel density) classifier.

    Each class's density is estimated by placing a window of width h on every
    training record of that class. A record is given to the class with the
    largest posterior, with class frequencies as priors. The prior P(c) = n_c / n
    cancels the 1/n_c in the density estimate, so the rule reduces to:

        choose the class whose training records put the most total window weight
        around the record being classified.

    Windows (u is the distance to a training record divided by the width h):

    * "gaussian"      exp(-u^2 / 2), Euclidean distance. Never exactly zero, and
                      evaluated in the log domain so it cannot underflow.
    * "epanechnikov"  max(0, 1 - u^2), Euclidean distance.
    * "hypercube"     1 inside a box of side h centred on the training record
                      (max-norm), else 0: the classic textbook window.

    The compact windows can be empty: no training record inside the window of
    a record being classified. That record is given the class of its nearest
    training record, and `window_probabilities` reports which records it
    happened to, so the caller can tell the user the window was too narrow.

    With bandwidth=None the width comes from Silverman's rule of thumb for data
    with unit variance (which is what z-scoring produces).
    """

    def __init__(self, numeric, categorical, kernel="gaussian", bandwidth=None):
        super().__init__(numeric, categorical)
        if kernel not in PARZEN_KERNELS:
            raise ValueError(f"kernel must be one of {', '.join(PARZEN_KERNELS)}")
        if bandwidth is not None and not bandwidth > 0:
            raise ValueError("bandwidth must be greater than 0")
        self.kernel = kernel
        self.bandwidth = bandwidth
        self.chunk_cells = 4_000_000  # distance-matrix cells held in memory at once

    def fit(self, X, y):
        y = np.asarray(y, dtype=str)
        self.classes_ = np.unique(y)
        self.train_class_idx_ = np.searchsorted(self.classes_, y)
        self.class_columns_ = [
            np.flatnonzero(self.train_class_idx_ == c) for c in range(len(self.classes_))
        ]

        self._fit_features(X)
        self._fit_levels(X)
        self.train_points_ = self._encode(X)
        self.n_dimensions_ = self.train_points_.shape[1]

        if self.bandwidth is None:
            # the all-zero "unseen category" columns add no spread, so don't count them
            d = max(1, self.n_dimensions_ - len(self.categorical))
            n = len(y)
            self.bandwidth_ = float((4.0 / (d + 2)) ** (1.0 / (d + 4)) * n ** (-1.0 / (d + 4)))
        else:
            self.bandwidth_ = float(self.bandwidth)
        return self

    def _block_probabilities(self, block):
        h = self.bandwidth_
        if self.kernel == "gaussian":
            log_weight = cdist(block, self.train_points_, "sqeuclidean") / (-2.0 * h * h)
            log_score = np.column_stack(
                [logsumexp(log_weight[:, cols], axis=1) for cols in self.class_columns_]
            )
            proba = np.exp(log_score - logsumexp(log_score, axis=1, keepdims=True))
            return proba, np.zeros(len(block), dtype=bool)

        if self.kernel == "epanechnikov":
            distance = cdist(block, self.train_points_, "sqeuclidean")
            weight = np.maximum(0.0, 1.0 - distance / (h * h))
        else:  # hypercube
            distance = cdist(block, self.train_points_, "chebyshev")
            weight = (distance <= h / 2.0).astype(float)

        score = np.column_stack([weight[:, cols].sum(axis=1) for cols in self.class_columns_])
        total = score.sum(axis=1)
        empty = total <= 0.0
        proba = np.zeros_like(score)
        np.divide(score, total[:, None], out=proba, where=~empty[:, None])
        if empty.any():
            rows = np.flatnonzero(empty)
            nearest = self.train_class_idx_[distance[rows].argmin(axis=1)]
            proba[rows, nearest] = 1.0
        return proba, empty

    def window_probabilities(self, X):
        """(posterior, empty_window) for every row of X.

        posterior       rows sum to 1, columns aligned with classes_
        empty_window    True where no training record fell inside the window and
                        the nearest training record decided instead
        """
        points = self._encode(X)
        step = max(1, int(self.chunk_cells // max(1, len(self.train_points_))))
        proba = np.empty((len(points), len(self.classes_)))
        empty = np.zeros(len(points), dtype=bool)
        for start in range(0, len(points), step):
            stop = start + step
            proba[start:stop], empty[start:stop] = self._block_probabilities(points[start:stop])
        return proba, empty

    def predict_proba(self, X):
        return self.window_probabilities(X)[0]

    def predict(self, X):
        return self.classes_[np.argmax(self.window_probabilities(X)[0], axis=1)]


# --------------------------------------------------------------------------
# Ho-Kashyap
# --------------------------------------------------------------------------
# An update that leaves no positive error component (relative to the margins)
# means the iteration has reached its fixed point.
HK_NO_POSITIVE_TOLERANCE = 1e-7


def ho_kashyap(Z, Z_pinv, sign, learning_rate, max_iterations, tolerance):
    """One Ho-Kashyap run (Duda, Hart & Stork, Pattern Classification, ch. 5).

    Z       design matrix, one row [1, x] per training record
    Z_pinv  its pseudo-inverse
    sign    +1 for records of the class being separated, -1 for the rest

    Multiplying each row by its sign gives Y = diag(sign) Z. The goal is a weight
    vector a and margins b > 0 with Y a = b, i.e. every record on the correct
    side of the boundary. Repeat:

        e = Y a - b              error between the boundary and the margins
        b <- b + 2 eta e+        raise margins only where the error is positive
        a <- Y+ b                refit the boundary by least squares

    Why it stopped is reported in `status`:

    * "converged"      every |e| <= tolerance, so Y a > 0: all training records
                       are on the correct side.
    * "not_separable"  e has no positive part yet is not zero. In exact
                       arithmetic that proves no linear boundary exists.
    * "max_iterations" neither happened within max_iterations updates.

    For 0 < eta < 1 the squared error ||e||^2 never increases.
    """
    b = np.ones(len(sign))
    a = Z_pinv @ (sign * b)
    history = []
    iterations = 0
    while True:
        e = sign * (Z @ a) - b
        max_error = float(np.abs(e).max())
        history.append(float(e @ e))
        if max_error <= tolerance:
            status = "converged"
            break
        e_pos = 0.5 * (e + np.abs(e))
        if e_pos.max() <= HK_NO_POSITIVE_TOLERANCE * (1.0 + b.max()):
            status = "not_separable"
            break
        if iterations >= max_iterations:
            status = "max_iterations"
            break
        b = b + 2.0 * learning_rate * e_pos
        a = Z_pinv @ (sign * b)
        iterations += 1
    return {
        "weights": a,
        "margins": b,
        "status": status,
        "iterations": iterations,
        "max_error": max_error,
        "criterion": history[-1],
        "history": history,
    }


class HoKashyapClassifier(_MixedFeatures):
    """Linear classifier trained with the Ho-Kashyap procedure.

    * two classes   -> one discriminant g(x); g(x) > 0 means the second class
                       (in sorted order), otherwise the first
    * more classes  -> one discriminant per class against all the others
                       (one-vs-rest); the class with the highest g(x) wins

    Features are prepared like the other models: missing values filled from the
    training data, numeric columns z-scored, categorical columns one-hot encoded
    (a category not seen in training gets all zeros, i.e. the baseline).

    Ho-Kashyap needs the classes to be linearly separable to converge. When they
    are not, the final weights are still the best compromise the procedure
    found; `runs_` records how each discriminant ended.
    """

    def __init__(self, numeric, categorical, learning_rate=0.5, max_iterations=5000, tolerance=1e-3):
        super().__init__(numeric, categorical)
        self.learning_rate = float(learning_rate)
        self.max_iterations = int(max_iterations)
        self.tolerance = float(tolerance)

    def _design(self, X):
        blocks = [np.ones((len(X), 1))]
        if self.numeric:
            blocks.append(self._scaled_numeric(X))
        if self.categorical:
            frame = self._categorical_frame(X)
            for col in self.categorical:
                levels = self.cat_levels_[col]
                codes = pd.Index(levels).get_indexer(frame[col])  # -1 for a category not seen in training
                onehot = np.zeros((len(X), len(levels)))
                known = codes >= 0
                onehot[np.flatnonzero(known), codes[known]] = 1.0
                blocks.append(onehot)
        return np.hstack(blocks)

    def fit(self, X, y):
        y = np.asarray(y, dtype=str)
        self.classes_ = np.unique(y)
        self._fit_features(X)
        frame = self._categorical_frame(X)
        self.cat_levels_ = {col: sorted(frame[col].unique()) for col in self.categorical}

        Z = self._design(X)
        self.n_dimensions_ = Z.shape[1] - 1  # encoded features, not counting the constant 1
        # Y = diag(sign) Z, and diag(sign) is orthogonal, so Y+ = Z+ diag(sign):
        # one pseudo-inverse serves every discriminant.
        Z_pinv = np.linalg.pinv(Z, rcond=1e-10)

        if len(self.classes_) == 2:
            targets = [(self.classes_[1], str(self.classes_[0]))]
        else:
            targets = [(cls, "all other classes") for cls in self.classes_]

        weights, self.runs_ = [], []
        for cls, versus in targets:
            sign = np.where(y == cls, 1.0, -1.0)
            run = ho_kashyap(Z, Z_pinv, sign, self.learning_rate, self.max_iterations, self.tolerance)
            weights.append(run["weights"])
            self.runs_.append({
                "label": str(cls),
                "versus": versus,
                "status": run["status"],
                "iterations": run["iterations"],
                "max_error": run["max_error"],
                "criterion": run["criterion"],
                # training records this boundary still puts on the wrong side (or exactly on it)
                "train_errors": int((sign * (Z @ run["weights"]) <= 0).sum()),
            })

        self.weights_ = np.column_stack(weights)  # one column per discriminant
        if not np.isfinite(self.weights_).all():
            raise ValueError(
                "The weights became non-finite. Try a smaller learning rate or fewer features."
            )
        return self

    def decision_scores(self, X):
        """Score per class for every row, aligned with classes_. The highest wins.
        For two classes the scores are -g(x) and +g(x)."""
        g = self._design(X) @ self.weights_
        if len(self.classes_) == 2:
            return np.column_stack([-g[:, 0], g[:, 0]])
        return g

    def predict(self, X):
        return self.classes_[np.argmax(self.decision_scores(X), axis=1)]


# --------------------------------------------------------------------------
# Choosing k
# --------------------------------------------------------------------------
def choose_k(ks, error_counts):
    """Find the lowest-error k and the elbow of the error curve.

    ks            recorded values of k, ascending
    error_counts  number of test records misclassified for each k (integers, so
                  ties are exact)

    Returns (optimal_index, elbow_index, tied_indices).

    * optimal: the lowest error; among ties, the smallest k.
    * elbow: look only at the values up to the optimal k, where the error is
      still coming down. Replace each error by the best error seen so far (so
      noise in the curve can't create false bends), then pick the point that
      sits furthest below the straight line joining the first recorded k to the
      optimal k. That is the point after which the improvement flattens out. If
      there is no bend (fewer than three points, or the curve never dips below
      that line) the elbow is the optimal k itself.
    """
    errors = np.asarray(error_counts, dtype=float)
    optimal = int(np.argmin(errors))
    tied = [i for i, e in enumerate(errors) if e == errors[optimal]]

    elbow = optimal
    if optimal >= 2:
        x = np.asarray(ks[: optimal + 1], dtype=float)
        y = np.minimum.accumulate(errors[: optimal + 1])
        # y[0] > y[-1] because `optimal` is the first index holding the minimum
        x_norm = (x - x[0]) / (x[-1] - x[0])
        y_norm = (y - y[-1]) / (y[0] - y[-1])
        gap = (1.0 - x_norm) - y_norm
        candidate = int(np.argmax(gap))
        if gap[candidate] > 1e-12:
            elbow = candidate
    return optimal, elbow, tied


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------
def evaluate_predictions(true_pos, pred_pos, n_classes):
    """Accuracy, weighted precision / recall / F1, per-class scores and the
    confusion matrix, all derived from one confusion matrix.

    true_pos / pred_pos are integer class positions in 0..n_classes-1.
    Classes that are never predicted get precision 0; classes absent from the
    test set carry no weight in the weighted averages.
    """
    true_pos = np.asarray(true_pos)
    pred_pos = np.asarray(pred_pos)
    matrix = np.bincount(true_pos * n_classes + pred_pos, minlength=n_classes ** 2)
    matrix = matrix.reshape(n_classes, n_classes)

    correct = np.diag(matrix).astype(float)
    support = matrix.sum(axis=1).astype(float)
    predicted = matrix.sum(axis=0).astype(float)
    total = support.sum()

    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(predicted > 0, correct / predicted, 0.0)
        recall = np.where(support > 0, correct / support, 0.0)
        denominator = precision + recall
        f1 = np.where(denominator > 0, 2 * precision * recall / denominator, 0.0)

    weights = support / total
    return {
        "errors": int(total - correct.sum()),
        "accuracy": float(correct.sum() / total),
        "precision": float((precision * weights).sum()),
        "recall": float((recall * weights).sum()),
        "f1": float((f1 * weights).sum()),
        "per_class": {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support.astype(int),
        },
        "confusion_matrix": matrix,
    }
