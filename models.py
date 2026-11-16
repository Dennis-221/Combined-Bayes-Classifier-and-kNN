"""Classifiers and evaluation helpers used by the app.

Each classifier works directly on pandas frames that hold a mix of numeric and
categorical columns, so the Flask routes never deal with encoding details.
To add an algorithm (e.g. Ho-Kashyap): write a class here, then register it in
ALGORITHMS in app.py.
"""

import numpy as np
import pandas as pd
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
# K-nearest neighbours
# --------------------------------------------------------------------------
class KNNClassifier(_MixedFeatures):
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
        frame = self._categorical_frame(X)
        self.cat_levels_ = {
            col: {value: i for i, value in enumerate(sorted(frame[col].unique()))}
            for col in self.categorical
        }
        encoded = self._encode(X)
        self.n_dimensions_ = encoded.shape[1]
        self.nn_ = NearestNeighbors(metric="euclidean").fit(encoded)
        return self

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
