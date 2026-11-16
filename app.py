import io
import math

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request
from sklearn.model_selection import train_test_split

from models import HoKashyapClassifier, KNNClassifier, MixedNaiveBayes, choose_k, evaluate_predictions

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024  # 32 MB upload cap

ALLOWED_EXTENSIONS = {"csv"}
SAMPLE_ROWS = 10          # rows shown in the "sample records" view
MAX_CLASSES = 20          # more distinct target values than this is not a classification target
MAX_CATEGORIES = 50       # categorical features with more levels are excluded
MIN_USABLE_ROWS = 10      # fewer labelled rows than this can't be split and evaluated
SMALL_DATASET_ROWS = 50   # below this we warn that metrics will be noisy
IMBALANCE_RATIO = 10      # largest/smallest class ratio that triggers a warning
MAX_K = 40                # KNN: k can be any whole number from 1 to this
SLOW_KNN_ROWS = 50_000    # KNN: warn above this many training records
HK_DEFAULTS = {"learning_rate": 0.5, "max_iterations": 5000, "tolerance": 0.001}
HK_MAX_ITERATIONS = 100_000
HK_MAX_WORK = 2e10        # Ho-Kashyap: refuse settings that would run for minutes
RANDOM_STATE = 42
TEST_SIZES = (0.1, 0.2, 0.25, 0.3)
DEFAULT_TEST_SIZE = 0.2

# Registry of supported algorithms: key -> label shown in the UI. To add one,
# write its class in models.py, add it here, and handle its key in train().
ALGORITHMS = {
    "naive_bayes": "Naive Bayes (Gaussian + Categorical)",
    "knn": "K-nearest neighbors (KNN)",
    "ho_kashyap": "Ho-Kashyap (linear discriminant)",
}
DEFAULT_ALGORITHM = "naive_bayes"

# Server-side state: the single active dataset, target and trained model.
# Fine for local, single-user use. A multi-user deployment would keep this
# per session (Flask session + server-side cache, or a database).
STATE = {"df": None, "filename": None, "target": None, "model": None, "model_counter": 0}


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def plural(n, word):
    return f"{n} {word}{'' if n == 1 else 's'}"


def r4(value):
    return round(float(value), 4)


def new_model_id():
    STATE["model_counter"] += 1
    return STATE["model_counter"]


def read_csv_bytes(raw):
    """Parse CSV bytes, tolerating Excel-style files: non-UTF-8 text and
    semicolon / tab / pipe separators."""

    def parse(**kwargs):
        try:
            return pd.read_csv(io.BytesIO(raw), **kwargs)
        except UnicodeDecodeError:
            # Excel on Windows often saves CSVs in a legacy encoding.
            return pd.read_csv(io.BytesIO(raw), encoding="latin-1", **kwargs)

    df = parse()
    if df.shape[1] == 1:
        header = str(df.columns[0])
        for sep in (";", "\t", "|"):
            if sep in header:
                return parse(sep=sep)
    return df


def format_cell(value):
    """Show whole-number floats (ints that were upcast because of NaN) without '.0'."""
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    return str(value)


def label_of(value):
    """Canonical string label for a target value (1.0 -> '1', True -> 'True')."""
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value))
    return format_cell(value)


def label_sort_key(label):
    """Sort labels numerically when they look like numbers, otherwise alphabetically."""
    try:
        return (0, float(label), label)
    except ValueError:
        return (1, 0.0, label)


def is_numeric_column(series):
    return pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)


def is_integer_sequence(values):
    """True for ID / row-index style columns: distinct integers that fill (almost)
    all of their range, e.g. 1..N with a few gaps left by deleted or filtered rows.
    These carry no signal for a classifier. Real numeric features such as salary
    are far sparser within their range, so they are not caught by the 90% density test."""
    if len(values) < 2 or values.nunique() != len(values):
        return False
    if not (values % 1 == 0).all():
        return False
    span = int(values.max() - values.min()) + 1
    return len(values) / span >= 0.9


def round_sig(value):
    return None if value is None or pd.isna(value) else float(f"{value:.6g}")


def require_dataset():
    if STATE["df"] is None:
        return None, (jsonify({"error": "Upload a CSV file first."}), 400)
    return STATE["df"], None


# --------------------------------------------------------------------------
# Target / feature analysis (shared by /target and /train)
# --------------------------------------------------------------------------
def analyze(df, target):
    """Check whether `target` can be used to train a classifier.

    Returns (summary, prepared). `prepared` is None when there are blocking
    errors, otherwise it holds the X / y / feature lists used for training.
    """
    errors, warnings = [], []

    total = len(df)
    labelled = df[target].notna()
    work = df.loc[labelled]
    usable = len(work)
    dropped = total - usable

    y = work[target].map(label_of)
    counts = y.value_counts()
    n_classes = len(counts)

    # ---- target checks ----------------------------------------------------
    if usable == 0:
        errors.append(f"Target '{target}' has no values to learn from.")
    elif n_classes < 2:
        errors.append(
            f"Target '{target}' has only one class ('{counts.index[0]}'). "
            "A classifier needs at least two."
        )
    elif n_classes > MAX_CLASSES:
        errors.append(
            f"Target '{target}' has {n_classes} distinct values, which looks continuous or "
            f"ID-like. A classifier needs a discrete target with at most {MAX_CLASSES} classes."
        )
    if 0 < usable < MIN_USABLE_ROWS:
        errors.append(
            f"Only {plural(usable, 'record')} {'has' if usable == 1 else 'have'} a target value; "
            f"at least {MIN_USABLE_ROWS} are needed to train and test a model."
        )
    if dropped:
        warnings.append(
            f"{plural(dropped, 'record')} {'has' if dropped == 1 else 'have'} no target value "
            "and will be left out."
        )

    # ---- feature checks ---------------------------------------------------
    numeric, categorical, excluded = [], [], []
    for col in df.columns:
        if col == target:
            continue
        values = work[col].dropna()
        n_unique = values.nunique()
        numeric_col = is_numeric_column(work[col])

        reason = None
        if len(values) == 0:
            reason = "empty (no values)"
        elif n_unique <= 1:
            reason = "constant (one distinct value)"
        elif not numeric_col and n_unique == len(values):
            reason = "ID-like (every value is unique)"
        elif not numeric_col and n_unique > MAX_CATEGORIES:
            reason = f"too many categories ({n_unique})"
        elif numeric_col and is_integer_sequence(values):
            reason = "ID-like (sequential integers)"

        if reason:
            excluded.append({"name": col, "reason": reason})
        elif numeric_col:
            numeric.append(col)
        else:
            categorical.append(col)

    features = [c for c in df.columns if c in numeric or c in categorical]
    if not features:
        if excluded:
            errors.append("No usable feature columns remain; every other column was excluded (see below).")
        else:
            errors.append("The dataset has no feature columns besides the target.")

    rows_with_missing = int(work[features].isna().any(axis=1).sum()) if features else 0
    if rows_with_missing:
        warnings.append(
            f"{plural(rows_with_missing, 'usable record')} {'has' if rows_with_missing == 1 else 'have'} "
            "missing feature values. They are kept: missing numbers are filled with the training "
            "median and missing categories with the most frequent value."
        )

    # ---- data-quality warnings ---------------------------------------------
    if 2 <= n_classes <= MAX_CLASSES:
        ratio = counts.max() / counts.min()
        if ratio > IMBALANCE_RATIO:
            warnings.append(
                f"Classes are imbalanced (the largest is {ratio:.0f}x the smallest). Accuracy can look "
                "high while minority classes are missed, so check per-class recall and F1."
            )
        rare = counts[counts < 2]
        if len(rare):
            names = ", ".join(f"'{label}'" for label in rare.index[:5])
            more = "..." if len(rare) > 5 else ""
            warnings.append(
                f"Class {names}{more} has only one record, which is too few to learn from or "
                "evaluate reliably."
                if len(rare) == 1 else
                f"Classes {names}{more} have only one record each, which is too few to learn "
                "from or evaluate reliably."
            )
    if MIN_USABLE_ROWS <= usable < SMALL_DATASET_ROWS:
        warnings.append(f"Only {usable} usable records, so metrics from the test split will be noisy.")

    # A target with too many distinct values has no meaningful class distribution.
    ordered = sorted(counts.items(), key=lambda kv: (-kv[1], label_sort_key(kv[0])))
    distribution = [] if n_classes > MAX_CLASSES else [
        {"label": label, "count": int(count), "pct": round(100 * count / usable, 1)}
        for label, count in ordered
    ]

    summary = {
        "target": target,
        "total_records": total,
        "usable_records": usable,
        "dropped_records": dropped,
        "n_features": len(features),
        "numeric_features": numeric,
        "categorical_features": categorical,
        "excluded_columns": excluded,
        "rows_with_missing_features": rows_with_missing,
        "n_classes": n_classes,
        "class_distribution": distribution,
        "errors": errors,
        "warnings": warnings,
        "ready": not errors,
    }
    prepared = None
    if not errors:
        prepared = {"X": work[features], "y": y, "numeric": numeric, "categorical": categorical}
    return summary, prepared


def build_schema(X_train, numeric, categorical):
    """Describe each feature so the UI can build the 'predict new instance' form."""
    fields = []
    for col in X_train.columns:
        if col in numeric:
            values = X_train[col].astype(float).replace([np.inf, -np.inf], np.nan).dropna()
            fields.append({
                "name": col,
                "kind": "numeric",
                "min": round_sig(values.min()) if len(values) else None,
                "max": round_sig(values.max()) if len(values) else None,
                "median": round_sig(values.median()) if len(values) else None,
            })
        else:
            levels = sorted(X_train[col].dropna().map(str).unique().tolist())
            fields.append({"name": col, "kind": "categorical", "categories": levels})
    return fields


# --------------------------------------------------------------------------
# Splitting and evaluation (shared by every algorithm)
# --------------------------------------------------------------------------
def prepare_split(df, target_name, test_size):
    """Analyse the target, then split the labelled records into train / test.
    Returns (split, error_response); exactly one of them is None."""
    summary, prepared = analyze(df, target_name)
    if prepared is None:
        return None, (jsonify({"errors": summary["errors"], "warnings": summary["warnings"]}), 400)

    X, y = prepared["X"], prepared["y"]
    warnings = []

    n = len(y)
    n_test = math.ceil(test_size * n)
    n_train = n - n_test
    counts = y.value_counts()
    n_classes = len(counts)

    stratified = bool(counts.min() >= 2 and n_test >= n_classes and n_train >= n_classes)
    if not stratified:
        if counts.min() < 2:
            reason = "a class has only one record"
        else:
            reason = "the test split is smaller than the number of classes"
        warnings.append(f"The split is not stratified because {reason}; a plain random split was used.")

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=RANDOM_STATE,
            stratify=y if stratified else None,
        )
    except ValueError as exc:
        return None, (jsonify({"errors": [f"Could not split the data: {exc}"]}), 400)

    if y_train.nunique() < 2:
        return None, (jsonify({"errors": [
            "The training split contains only one class, so nothing can be learned. "
            "Use a smaller test split or add more records."
        ]}), 400)

    unseen = [c for c in counts.index if c not in set(y_train)]
    if unseen:
        warnings.append(
            "No training record belongs to class " + ", ".join(f"'{c}'" for c in sorted(unseen)) +
            ", so the model can never predict it."
        )

    majority = y_train.value_counts().idxmax()
    return {
        "X": X, "X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test,
        "numeric": prepared["numeric"], "categorical": prepared["categorical"],
        "labels": sorted(counts.index, key=label_sort_key),
        "stratified": stratified,
        "baseline": float((y_test == majority).mean()),
        "warnings": warnings,
    }, None


def split_fields(split, test_size):
    """Response fields every algorithm reports about the split."""
    return {
        "n_train": int(len(split["y_train"])),
        "n_test": int(len(split["y_test"])),
        "n_features": int(split["X"].shape[1]),
        "test_size": test_size,
        "stratified": split["stratified"],
        "random_state": RANDOM_STATE,
        "labels": split["labels"],
        "baseline_accuracy": r4(split["baseline"]),
        "schema": build_schema(split["X_train"], split["numeric"], split["categorical"]),
    }


def metric_fields(ev):
    return {
        "accuracy": r4(ev["accuracy"]),
        "precision": r4(ev["precision"]),
        "recall": r4(ev["recall"]),
        "f1": r4(ev["f1"]),
    }


def per_class_rows(labels, ev):
    pc = ev["per_class"]
    return [
        {
            "label": label,
            "precision": r4(pc["precision"][i]),
            "recall": r4(pc["recall"][i]),
            "f1": r4(pc["f1"][i]),
            "support": int(pc["support"][i]),
        }
        for i, label in enumerate(labels)
    ]


# --------------------------------------------------------------------------
# Naive Bayes
# --------------------------------------------------------------------------
def train_naive_bayes(split, test_size):
    try:
        model = MixedNaiveBayes(split["numeric"], split["categorical"]).fit(
            split["X_train"], split["y_train"]
        )
        y_pred = model.predict(split["X_test"])
    except Exception as exc:  # surface any library error instead of a bare 500
        return jsonify({"errors": [f"Training failed: {exc}"]}), 500

    labels = split["labels"]
    index = pd.Index(labels)
    ev = evaluate_predictions(index.get_indexer(split["y_test"]), index.get_indexer(y_pred), len(labels))

    fields = split_fields(split, test_size)
    model_id = new_model_id()
    STATE["model"] = {
        "algorithm": "naive_bayes",
        "label": ALGORITHMS["naive_bayes"],
        "model_id": model_id,
        "clf": model,
        "features": split["X"].columns.tolist(),
        "numeric": split["numeric"],
        "categorical": split["categorical"],
        "schema": fields["schema"],
    }

    return jsonify({
        **fields,
        "algorithm": ALGORITHMS["naive_bayes"],
        "algorithm_key": "naive_bayes",
        "model_id": model_id,
        "metrics": {**metric_fields(ev), "average": "weighted"},
        "per_class": per_class_rows(labels, ev),
        "confusion_matrix": ev["confusion_matrix"].tolist(),
        "warnings": split["warnings"],
    })


# --------------------------------------------------------------------------
# K-nearest neighbours
# --------------------------------------------------------------------------
def parse_k(value):
    """A whole number from 1 to MAX_K, or None."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        k = value
    elif isinstance(value, float) and value.is_integer():
        k = int(value)
    elif isinstance(value, str) and value.strip().isdigit():
        k = int(value.strip())
    else:
        return None
    return k if 1 <= k <= MAX_K else None


def parse_ks(params):
    raw = params.get("ks") if isinstance(params, dict) else None
    if not isinstance(raw, list) or not raw:
        return None, "Select at least one value of k."
    ks = set()
    for value in raw:
        k = parse_k(value)
        if k is None:
            return None, f"k must be a whole number from 1 to {MAX_K}."
        ks.add(k)
    return sorted(ks), None


def build_knn_bundle(split, target_name, test_size):
    """Fit the neighbour index once; every k is answered from it later."""
    clf = KNNClassifier(split["numeric"], split["categorical"]).fit(split["X_train"], split["y_train"])
    labels = split["labels"]
    index = pd.Index(labels)
    warnings = list(split["warnings"])
    if len(split["y_train"]) > SLOW_KNN_ROWS:
        warnings.append(
            f"KNN compares every test record with {len(split['y_train']):,} training records, "
            "so this may take a while."
        )
    return {
        "algorithm": "knn",
        "label": ALGORITHMS["knn"],
        "model_id": new_model_id(),
        "target": target_name,
        "test_size": test_size,
        "clf": clf,
        "features": split["X"].columns.tolist(),
        "numeric": split["numeric"],
        "categorical": split["categorical"],
        "X_test": split["X_test"],
        "true_pos": index.get_indexer(split["y_test"]),
        "class_to_pos": np.array([labels.index(c) for c in clf.classes_]),
        "n_dimensions": int(clf.n_dimensions_),
        "fields": split_fields(split, test_size),
        "warnings": warnings,
        "records": {},   # k -> evaluation of that k on the test split
    }


def record_knn(bundle, ks):
    """Evaluate every k in `ks` on the test split and add it to the record."""
    n_labels = len(bundle["fields"]["labels"])
    predictions = bundle["clf"].predict_indices(bundle["X_test"], ks)
    for k, class_idx in predictions.items():
        ev = evaluate_predictions(bundle["true_pos"], bundle["class_to_pos"][class_idx], n_labels)
        bundle["records"][k] = ev


def knn_selection(bundle):
    """Recorded ks (ascending) with the optimal and elbow k."""
    ks = sorted(bundle["records"])
    optimal, elbow, tied = choose_k(ks, [bundle["records"][k]["errors"] for k in ks])
    return ks, ks[optimal], ks[elbow], [ks[i] for i in tied]


def knn_payload(bundle):
    labels = bundle["fields"]["labels"]
    n_test = bundle["fields"]["n_test"]
    ks, optimal_k, elbow_k, tied_ks = knn_selection(bundle)
    records = bundle["records"]
    return {
        **bundle["fields"],
        "algorithm": bundle["label"],
        "algorithm_key": "knn",
        "model_id": bundle["model_id"],
        "n_dimensions": bundle["n_dimensions"],
        "warnings": bundle["warnings"],
        "records": [
            {
                "k": k,
                "errors": records[k]["errors"],
                "error_rate": r4(records[k]["errors"] / n_test),
                **metric_fields(records[k]),
            }
            for k in ks
        ],
        "details": {
            str(k): {
                "per_class": per_class_rows(labels, records[k]),
                "confusion_matrix": records[k]["confusion_matrix"].tolist(),
            }
            for k in ks
        },
        "optimal_k": optimal_k,
        "elbow_k": elbow_k,
        "tied_ks": tied_ks,
    }


# --------------------------------------------------------------------------
# Ho-Kashyap
# --------------------------------------------------------------------------
def as_number(value):
    """A finite float from a number or numeric string, otherwise None."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
    elif isinstance(value, str):
        try:
            number = float(value.strip())
        except ValueError:
            return None
    else:
        return None
    return number if math.isfinite(number) else None


def parse_hk_params(params):
    """Validate the Ho-Kashyap settings. Returns (values, problems)."""
    raw = params if isinstance(params, dict) else {}
    values, problems = {}, []

    rate = as_number(raw.get("learning_rate", HK_DEFAULTS["learning_rate"]))
    if rate is None or not 0 < rate < 1:
        problems.append("Learning rate must be a number greater than 0 and less than 1.")
    else:
        values["learning_rate"] = rate

    iterations = as_number(raw.get("max_iterations", HK_DEFAULTS["max_iterations"]))
    if iterations is None or not iterations.is_integer() or not 1 <= iterations <= HK_MAX_ITERATIONS:
        problems.append(f"Maximum iterations must be a whole number from 1 to {HK_MAX_ITERATIONS:,}.")
    else:
        values["max_iterations"] = int(iterations)

    tolerance = as_number(raw.get("tolerance", HK_DEFAULTS["tolerance"]))
    if tolerance is None or not 1e-6 <= tolerance <= 0.5:
        problems.append("Convergence threshold must be a number from 0.000001 to 0.5.")
    else:
        values["tolerance"] = tolerance

    return values, problems


def hk_warnings(runs, params, n_train):
    """Plain-language warnings for boundaries that still misclassify training records."""
    unresolved = [r for r in runs if r["train_errors"] > 0]
    if not unresolved:
        return []
    if len(runs) == 1:
        r = runs[0]
        wrong = f"{r['train_errors']} of {n_train} training records"
        if r["status"] == "not_separable":
            return [
                "Ho-Kashyap shows these two classes cannot be separated by a straight boundary "
                f"({wrong} stay on the wrong side). The classifier uses the best compromise it "
                "found, so expect mistakes."
            ]
        return [
            f"Ho-Kashyap reached the {params['max_iterations']:,}-iteration limit with {wrong} still on the "
            "wrong side. The classes may not be linearly separable; the classifier uses the best boundary "
            "found so far."
        ]
    return [
        f"{len(unresolved)} of {len(runs)} one-vs-rest boundaries still misclassify training records, so "
        "those classes are not (or not yet shown to be) linearly separable from the rest. The classifier "
        "uses the best compromise found for each."
    ]


def train_ho_kashyap(split, test_size, params):
    X_train = split["X_train"]
    n_train = len(X_train)
    train_classes = int(split["y_train"].nunique())
    n_boundaries = 1 if train_classes == 2 else train_classes
    width = 1 + len(split["numeric"]) + sum(int(X_train[c].nunique()) for c in split["categorical"])

    # Each iteration is two matrix-vector products per boundary; refuse settings
    # that would keep the server busy for minutes.
    work = params["max_iterations"] * n_boundaries * n_train * width
    if work > HK_MAX_WORK:
        affordable = int(HK_MAX_WORK // (n_boundaries * n_train * width))
        advice = (
            f"Lower the maximum iterations to about {affordable:,} or less."
            if affordable >= 10 else "Use fewer records or fewer features."
        )
        return jsonify({"errors": [
            f"Ho-Kashyap with {params['max_iterations']:,} iterations would take too long on {n_train:,} "
            f"training records with {width} encoded columns and {n_boundaries} "
            f"{'boundary' if n_boundaries == 1 else 'boundaries'}. {advice}"
        ]}), 400

    try:
        model = HoKashyapClassifier(split["numeric"], split["categorical"], **params).fit(
            X_train, split["y_train"]
        )
        y_pred = model.predict(split["X_test"])
        training_accuracy = float((model.predict(X_train) == split["y_train"].to_numpy()).mean())
    except Exception as exc:
        return jsonify({"errors": [f"Training failed: {exc}"]}), 500

    labels = split["labels"]
    index = pd.Index(labels)
    ev = evaluate_predictions(index.get_indexer(split["y_test"]), index.get_indexer(y_pred), len(labels))

    fields = split_fields(split, test_size)
    model_id = new_model_id()
    STATE["model"] = {
        "algorithm": "ho_kashyap",
        "label": ALGORITHMS["ho_kashyap"],
        "model_id": model_id,
        "clf": model,
        "features": split["X"].columns.tolist(),
        "numeric": split["numeric"],
        "categorical": split["categorical"],
        "schema": fields["schema"],
    }

    return jsonify({
        **fields,
        "algorithm": ALGORITHMS["ho_kashyap"],
        "algorithm_key": "ho_kashyap",
        "model_id": model_id,
        "n_dimensions": int(model.n_dimensions_),
        "training_accuracy": r4(training_accuracy),
        "metrics": {**metric_fields(ev), "average": "weighted"},
        "per_class": per_class_rows(labels, ev),
        "confusion_matrix": ev["confusion_matrix"].tolist(),
        "convergence": {
            "mode": "binary" if len(model.runs_) == 1 else "one_vs_rest",
            "parameters": params,
            "n_train": n_train,
            "discriminants": [
                {**r, "max_error": round_sig(r["max_error"]), "criterion": round_sig(r["criterion"])}
                for r in model.runs_
            ],
        },
        "warnings": split["warnings"] + hk_warnings(model.runs_, params, n_train),
    })


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------
@app.route("/")
def home():
    return render_template(
        "index.html",
        algorithms=list(ALGORITHMS.items()),
        test_sizes=TEST_SIZES,
        default_test_size=DEFAULT_TEST_SIZE,
        max_k=MAX_K,
        hk_defaults=HK_DEFAULTS,
        hk_max_iterations=HK_MAX_ITERATIONS,
    )


@app.errorhandler(413)
def too_large(_error):
    limit = app.config["MAX_CONTENT_LENGTH"] // (1024 * 1024)
    return jsonify({"error": f"File is too large (limit is {limit} MB)."}), 413


@app.route("/upload", methods=["POST"])
def upload():
    file = request.files.get("file")
    if file is None:
        return jsonify({"error": "No file part in the request."}), 400
    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": "Only .csv files are accepted."}), 400

    try:
        df = read_csv_bytes(file.read())
    except pd.errors.EmptyDataError:
        return jsonify({"error": "The file is empty."}), 400
    except (pd.errors.ParserError, ValueError):
        return jsonify({"error": "Could not read the file as CSV. Check that it is a valid delimited text file."}), 400

    if df.shape[0] == 0 or df.shape[1] == 0:
        return jsonify({"error": "The file has no data rows."}), 400

    df.columns = [str(c) for c in df.columns]
    STATE.update(df=df, filename=file.filename, target=None, model=None)

    return jsonify({
        "filename": file.filename,
        "rows": int(len(df)),
        "columns": [
            {"name": c, "dtype": str(df[c].dtype), "unique": int(df[c].nunique(dropna=True))}
            for c in df.columns
        ],
    })


@app.route("/explore", methods=["POST"])
def explore():
    df, error = require_dataset()
    if error:
        return error

    missing = df.isna().sum()
    columns = [
        {
            "name": c,
            "dtype": str(df[c].dtype),
            "missing": int(missing[c]),
            "missing_pct": round(100 * missing[c] / len(df), 1),
            "unique": int(df[c].nunique(dropna=True)),
        }
        for c in df.columns
    ]

    sample = df.sample(n=min(SAMPLE_ROWS, len(df))).sort_index()
    sample_rows = [
        {
            "row_number": int(idx) + 1,
            "cells": [None if pd.isna(v) else format_cell(v) for v in row.tolist()],
        }
        for idx, row in sample.iterrows()
    ]

    return jsonify({
        "rows": int(len(df)),
        "column_count": int(df.shape[1]),
        "missing_cells": int(missing.sum()),
        "columns": columns,
        "sample": {"columns": df.columns.tolist(), "rows": sample_rows},
    })


@app.route("/target", methods=["POST"])
def target():
    df, error = require_dataset()
    if error:
        return error

    name = (request.get_json(silent=True) or {}).get("target")
    if name not in df.columns:
        return jsonify({"error": "Choose a target column from the dataset."}), 400

    summary, _ = analyze(df, name)
    STATE["target"] = name
    STATE["model"] = None  # any previously trained model belongs to the old target
    return jsonify(summary)


@app.route("/train", methods=["POST"])
def train():
    df, error = require_dataset()
    if error:
        return error

    target_name = STATE["target"]
    if not target_name:
        return jsonify({"errors": ["Choose a target variable and analyze it before training."]}), 400

    payload = request.get_json(silent=True) or {}
    algorithm = payload.get("algorithm", DEFAULT_ALGORITHM)
    if algorithm not in ALGORITHMS:
        return jsonify({"errors": [f"Unknown algorithm '{algorithm}'."]}), 400
    try:
        test_size = float(payload.get("test_size", DEFAULT_TEST_SIZE))
    except (TypeError, ValueError):
        test_size = float("nan")
    if not 0.05 <= test_size <= 0.5:
        return jsonify({"errors": ["Test split must be between 5% and 50%."]}), 400

    hk_params = None
    if algorithm == "ho_kashyap":
        hk_params, problems = parse_hk_params(payload.get("params"))
        if problems:
            return jsonify({"errors": problems}), 400

    if algorithm == "knn":
        ks, problem = parse_ks(payload.get("params"))
        if problem:
            return jsonify({"errors": [problem]}), 400

        # Running KNN again with the same target and test split keeps the fitted
        # index and adds the new values of k to the record. Anything else starts over.
        bundle = STATE["model"]
        reusable = bool(
            bundle and bundle["algorithm"] == "knn"
            and bundle["target"] == target_name and bundle["test_size"] == test_size
        )
        if not reusable:
            split, error = prepare_split(df, target_name, test_size)
            if error:
                return error
            try:
                bundle = build_knn_bundle(split, target_name, test_size)
            except Exception as exc:
                return jsonify({"errors": [f"Training failed: {exc}"]}), 500

        n_train = bundle["fields"]["n_train"]
        too_big = [k for k in ks if k > n_train]
        if too_big:
            return jsonify({"errors": [
                f"k = {', '.join(str(k) for k in too_big)} "
                f"{'is' if len(too_big) == 1 else 'are'} larger than the {n_train} training records. "
                f"Choose k up to {n_train}."
            ]}), 400

        try:
            record_knn(bundle, ks)
        except Exception as exc:
            return jsonify({"errors": [f"Training failed: {exc}"]}), 500
        STATE["model"] = bundle
        return jsonify(knn_payload(bundle))

    split, error = prepare_split(df, target_name, test_size)
    if error:
        return error
    if algorithm == "ho_kashyap":
        return train_ho_kashyap(split, test_size, hk_params)
    return train_naive_bayes(split, test_size)


def current_knn_bundle():
    bundle = STATE["model"]
    if not bundle or bundle["algorithm"] != "knn":
        return None, (jsonify({"errors": ["Run KNN first."]}), 400)
    return bundle, None


@app.route("/knn/curve", methods=["POST"])
def knn_curve():
    bundle, error = current_knn_bundle()
    if error:
        return error
    if len(bundle["records"]) < 2:
        return jsonify({"errors": ["Record error rates for at least two values of k to plot the curve."]}), 400

    n_test = bundle["fields"]["n_test"]
    records = bundle["records"]
    ks, optimal_k, elbow_k, tied_ks = knn_selection(bundle)
    elbow_errors = records[elbow_k]["errors"]

    comparison, tally = [], {"better": 0, "same": 0, "worse": 0}
    for k in ks:
        rec = records[k]
        if k == elbow_k:
            verdict = "elbow"
        else:
            verdict = ("better" if rec["errors"] < elbow_errors
                       else "same" if rec["errors"] == elbow_errors else "worse")
            tally[verdict] += 1
        comparison.append({
            "k": k,
            "errors": rec["errors"],
            "error_rate": r4(rec["errors"] / n_test),
            **metric_fields(rec),
            "delta_pts": round((rec["errors"] - elbow_errors) / n_test * 100, 2),
            "verdict": verdict,
            "is_elbow": k == elbow_k,
            "is_optimal": k == optimal_k,
        })

    return jsonify({
        "n_test": n_test,
        "points": [{"k": c["k"], "errors": c["errors"], "error_rate": c["error_rate"]} for c in comparison],
        "optimal_k": optimal_k,
        "optimal_error_rate": r4(records[optimal_k]["errors"] / n_test),
        "tied_ks": tied_ks,
        "elbow_k": elbow_k,
        "elbow_error_rate": r4(elbow_errors / n_test),
        "elbow_is_optimal": elbow_k == optimal_k,
        "comparison": comparison,
        "tally": tally,
    })


@app.route("/knn/clear", methods=["POST"])
def knn_clear():
    if STATE["model"] and STATE["model"]["algorithm"] == "knn":
        STATE["model"] = None
    return jsonify({"cleared": True})


@app.route("/predict", methods=["POST"])
def predict():
    bundle = STATE["model"]
    if bundle is None:
        return jsonify({"errors": ["Train a model before predicting."]}), 400

    payload = request.get_json(silent=True) or {}
    values = payload.get("values", {})
    row, problems, blank = {}, [], []

    for col in bundle["features"]:
        raw = values.get(col)
        text = "" if raw is None else str(raw).strip()
        if text == "":
            row[col] = np.nan
            blank.append(col)
        elif col in bundle["numeric"]:
            try:
                number = float(text)
                if not math.isfinite(number):
                    raise ValueError
                row[col] = number
            except ValueError:
                problems.append(f"'{col}' must be a number (got '{text}').")
        else:
            row[col] = text

    k = None
    if bundle["algorithm"] == "knn":
        n_train = bundle["fields"]["n_train"]
        if payload.get("k") is None:
            k = knn_selection(bundle)[1]
        else:
            k = parse_k(payload.get("k"))
            if k is None or k > n_train:
                problems.append(f"k must be a whole number from 1 to {min(MAX_K, n_train)}.")

    if problems:
        return jsonify({"errors": problems}), 400

    X_new = pd.DataFrame([row], columns=bundle["features"])
    model = bundle["clf"]

    if bundle["algorithm"] == "ho_kashyap":
        try:
            scores = model.decision_scores(X_new)[0]
        except Exception as exc:
            return jsonify({"errors": [f"Prediction failed: {exc}"]}), 500
        order = sorted(range(len(scores)), key=lambda i: -scores[i])  # stable: ties keep class order
        if len(scores) == 2:
            explanation = (
                f"Ho-Kashyap linear discriminant. g(x) = {scores[1]:+.3f}: positive values choose "
                f"'{model.classes_[1]}', negative values choose '{model.classes_[0]}'. The scores are "
                "-g(x) and +g(x), not probabilities."
            )
        else:
            explanation = (
                "Ho-Kashyap with one boundary per class (one-vs-rest). Each score is that class's "
                "discriminant value and the highest score wins. Scores are not probabilities."
            )
        return jsonify({
            "prediction": str(model.classes_[order[0]]),
            "scores": [{"label": str(model.classes_[i]), "score": round(float(scores[i]), 6)} for i in order],
            "filled_columns": blank,
            "model": bundle["label"],
            "k": None,
            "explanation": explanation,
        })

    try:
        if bundle["algorithm"] == "knn":
            votes = model.vote_counts(X_new, k)
            proba = votes / k
            if k == 1:
                explanation = (
                    "K-nearest neighbors with k = 1. The prediction is the class of the single "
                    "nearest training record."
                )
            else:
                explanation = (
                    f"K-nearest neighbors with k = {k}. Each bar shows how many of the {k} nearest "
                    "training records belong to that class."
                )
        else:
            votes = None
            proba = model.predict_proba(X_new)[0]
            explanation = "Naive Bayes. Each bar shows the model's probability for that class."
    except Exception as exc:
        return jsonify({"errors": [f"Prediction failed: {exc}"]}), 500

    # Python's sort is stable, so tied classes keep the order of classes_ (the same
    # tie-break the model itself uses).
    entries = list(range(len(model.classes_)))
    entries.sort(key=lambda i: -proba[i])
    probabilities = []
    for i in entries:
        item = {"label": str(model.classes_[i]), "probability": round(float(proba[i]), 6)}
        if votes is not None:
            item["votes"] = int(votes[i])
        probabilities.append(item)

    return jsonify({
        "prediction": probabilities[0]["label"],
        "probabilities": probabilities,
        "filled_columns": blank,
        "model": bundle["label"],
        "k": k,
        "explanation": explanation,
    })


if __name__ == "__main__":
    app.run(debug=True)
