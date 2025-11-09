import time
from dataclasses import dataclass
from typing import Callable, List
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_covtype, fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.tree import DecisionTreeClassifier as SkDecisionTree
from sklearn.metrics import accuracy_score
from scipy import sparse
from a2 import PyDecisionTreeClassifier

# Configuration
config = dict(
    n_runs=10,
    max_samples=50000,
    max_depth=10,
    min_samples_split=20,
    min_samples_leaf=10,
)


@dataclass
class BenchmarkResult:
    dataset: str
    model: str
    train_time: float
    infer_time: float
    accuracy: float


def load_covtype():
    X, y = fetch_covtype(as_frame=True, return_X_y=True)
    return X, y


def load_adult():
    X, y = fetch_openml("adult", version=2, as_frame=True, return_X_y=True)
    return X, y


def preprocess(X_train, X_test):
    num_sel = make_column_selector(dtype_include=["int64", "float64"])
    cat_sel = make_column_selector(dtype_exclude=["int64", "float64"])
    preproc = ColumnTransformer(
        [
            ("num", "passthrough", num_sel),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_sel),
        ]
    )
    X_train = preproc.fit_transform(X_train)
    X_test = preproc.transform(X_test)
    if sparse.issparse(X_train):
        X_train, X_test = X_train.toarray(), X_test.toarray()
    return X_train, X_test


def run_benchmark(
    name: str,
    loader: Callable[[], tuple],
    n_runs: int = 1,
    max_samples: int | None = None,
    max_depth: int = 10,
    min_samples_split: int = 20,
    min_samples_leaf: int = 10,
) -> List[BenchmarkResult]:
    X, y = loader()
    if max_samples and len(y) > max_samples:
        X = X.sample(n=max_samples, random_state=0)
        y = y.loc[X.index] if hasattr(y, "loc") else y[:max_samples]

    results = []

    for _ in range(n_runs):
        X_train, X_test, y_train_raw, y_test_raw = train_test_split(
            X, y, test_size=0.25, random_state=0, stratify=y
        )
        y_train = LabelEncoder().fit_transform(y_train_raw)
        y_test = LabelEncoder().fit(y_train_raw).transform(y_test_raw)
        X_train, X_test = preprocess(X_train, X_test)

        # scikit-learn
        sk = SkDecisionTree(
            criterion="gini",
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=0,
        )
        t0 = time.perf_counter()
        sk.fit(X_train, y_train)
        t1 = time.perf_counter()
        y_pred = sk.predict(X_test)
        t2 = time.perf_counter()
        results.append(
            BenchmarkResult(
                name, "sklearn", t1 - t0, t2 - t1, accuracy_score(y_test, y_pred)
            )
        )

        # Rust
        rs = PyDecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
        )
        t0 = time.perf_counter()
        rs.fit(X_train, y_train)
        t1 = time.perf_counter()
        y_pred = rs.predict(X_test)
        t2 = time.perf_counter()
        results.append(
            BenchmarkResult(
                name, "rust", t1 - t0, t2 - t1, accuracy_score(y_test, y_pred)
            )
        )
    return results


def main():

    # Print config
    print("Benchmark configuration:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    print()

    all_results = []
    for name, loader in [
        ("covtype", load_covtype),
        ("adult", load_adult),
    ]:
        print(f"Running {name} ...")
        all_results += run_benchmark(name, loader, **config)

    df = pd.DataFrame(r.__dict__ for r in all_results)
    summary = (
        df.groupby(["dataset", "model"]).mean(numeric_only=True).reset_index().round(4)
    )

    print("\nAverage results over", config["n_runs"], "runs:\n")
    print(summary.to_string(index=False))

    for dataset in summary["dataset"].unique():
        dset = summary[summary["dataset"] == dataset]
        sk, rs = (
            dset[dset["model"] == "sklearn"].iloc[0],
            dset[dset["model"] == "rust"].iloc[0],
        )
        print(f"\nDataset: {dataset}")
        print(
            f"  sklearn: acc={sk.accuracy:.4f}, train={sk.train_time:.3f}s, infer={sk.infer_time:.3f}s"
        )
        print(
            f"  rust   : acc={rs.accuracy:.4f}, train={rs.train_time:.3f}s, infer={rs.infer_time:.3f}s"
        )


if __name__ == "__main__":
    main()
