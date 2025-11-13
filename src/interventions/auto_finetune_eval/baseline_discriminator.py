from __future__ import annotations
from typing import List, Tuple, Optional, Dict
import re
import numpy as np

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression


# ------- text cleaning: remove tokens containing [], {}, <> entirely -------
_PUNCT_TOKEN_RE = re.compile(r"\S*[\[\]\{\}\<\>\"]\S*")

def _clean_text(s: Optional[str]) -> str:
    """
    Remove tokens that contain any of: [], {}, <>, \" to avoid markup/format artifacts.
    Keep only a single space where those tokens were.
    """
    if s is None:
        return " "
    s = str(s)
    # Replace any offending token with a single space
    s = _PUNCT_TOKEN_RE.sub(" ", s)
    # Normalize whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s if s else " "


def _prepare_xy(texts_m1: List[str], texts_m2: List[str]) -> Tuple[List[str], np.ndarray]:
    X = [_clean_text(t) for t in texts_m1] + [_clean_text(t) for t in texts_m2]
    y = np.array([1] * len(texts_m1) + [0] * len(texts_m2), dtype=int)
    return X, y


class LogisticBoWDiscriminator:
    """
    Logistic regression (bag-of-words) baseline with 5-fold CV.

    - Vectorizer indexes alphabetic tokens only (no punctuation/numbers) via token_pattern.
    - Pre-cleaning drops any token that contains: [, ], {, }, <, > to avoid formatting artifacts.
    - __call__(texts_m1, texts_m2) returns (mean_accuracy, mean_auc) via 5-fold CV on the provided data.
    - fit(...) trains once on all provided data so you can later inspect coefficients or score holdouts.
    - Optionally prints top-k features (by coefficient weight) for each class at training time.
    """

    def __init__(
        self,
        *,
        ngram_range: Tuple[int, int] = (1, 1),  # unigrams; set to (1,2) if you want bigrams too
        min_df: int = 1,
        max_features: Optional[int] = None,     # cap feature count if desired
        C: float = 1.0,
        penalty: str = "l2",
        max_iter: int = 2000,
        class_weight: Optional[str] = "balanced",
        random_state: int = 0,
    ):
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_features = max_features
        self.C = C
        self.penalty = penalty
        self.max_iter = max_iter
        self.class_weight = class_weight
        self.random_state = random_state

        self.pipeline_: Optional[Pipeline] = None
        self.feature_names_: Optional[np.ndarray] = None
        self.coef_: Optional[np.ndarray] = None
        self.intercept_: Optional[np.ndarray] = None

    def _build_pipeline(self) -> Pipeline:
        vec = CountVectorizer(
            # Only alphabetic words: avoids punctuation like [], {}, <> and also numbers/IDs
            token_pattern=r"(?u)\b[a-z]+\b",
            lowercase=True,
            strip_accents="unicode",
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            max_features=self.max_features,
        )
        clf = LogisticRegression(
            penalty=self.penalty,
            C=self.C,
            max_iter=self.max_iter,
            class_weight=self.class_weight,
            solver="liblinear",   # robust for small/medium datasets; supports predict_proba
            random_state=self.random_state,
        )
        return Pipeline([("vec", vec), ("clf", clf)])

    def fit(
        self,
        texts_m1: List[str],
        texts_m2: List[str],
        *,
        top_k: Optional[int] = None,
        label_names: Tuple[str, str] = ("M1", "M2"),
    ) -> "LogisticBoWDiscriminator":
        """
        Train on all provided data.
        If top_k is provided, print the top-k features for both classes by coefficient weight.

        label_names: (positive_class_name, negative_class_name)
                      Positive class corresponds to texts_m1 (label = 1).
        """
        X, y = _prepare_xy(texts_m1, texts_m2)
        pipe = self._build_pipeline()
        pipe.fit(X, y)
        self.pipeline_ = pipe

        vec = pipe.named_steps["vec"]
        clf = pipe.named_steps["clf"]

        self.feature_names_ = vec.get_feature_names_out()
        self.coef_ = clf.coef_.copy()      # shape (1, n_features) for binary
        self.intercept_ = clf.intercept_.copy()

        if top_k is not None and top_k > 0:
            self.print_top_k(k=top_k, label_names=label_names)

        return self

    def cross_validate(
        self,
        texts_m1: List[str],
        texts_m2: List[str],
        *,
        n_splits: int = 5,
    ) -> Tuple[float, float]:
        """
        5-fold stratified CV. Returns (mean_accuracy, mean_roc_auc).
        """
        X, y = _prepare_xy(texts_m1, texts_m2)
        pipe = self._build_pipeline()

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        acc = cross_val_score(pipe, X, y, cv=skf, scoring="accuracy")
        auc = cross_val_score(pipe, X, y, cv=skf, scoring="roc_auc")

        return float(acc.mean()), float(auc.mean())

    def __call__(self, texts_m1: List[str], texts_m_2: List[str]) -> Tuple[float, float]:
        """
        Allow: acc, auc = model(m1_texts, m2_texts)  # 5-fold CV on provided sets
        """
        return self.cross_validate(texts_m1=texts_m1, texts_m2=texts_m_2)

    # ------------------- feature inspection helpers -------------------

    def top_k_features(
        self,
        k: int = 20,
        *,
        label_names: Tuple[str, str] = ("M1", "M2"),
    ) -> Dict[str, List[Tuple[str, float]]]:
        """
        Return top-k features for the positive class (label 1, texts_m1)
        and the negative class (label 0, texts_m2) by coefficient weight.

        Returns dict:
          {
            label_names[0]: [(feature, weight), ...],  # positive class top-k
            label_names[1]: [(feature, weight), ...],  # negative class top-k (most negative weights)
          }
        """
        if self.pipeline_ is None or self.coef_ is None or self.feature_names_ is None:
            raise RuntimeError("Model not fit yet.")

        coef = self.coef_[0]  # shape (n_features,)
        n_features = coef.shape[0]
        k = min(k, n_features)

        # Positive class (label=1) -> largest positive weights
        pos_idx = np.argsort(coef)[::-1][:k]
        pos_feats = [(self.feature_names_[i], float(coef[i])) for i in pos_idx]

        # Negative class (label=0) -> most negative weights
        neg_idx = np.argsort(coef)[:k]
        neg_feats = [(self.feature_names_[i], float(coef[i])) for i in neg_idx]

        return {
            label_names[0]: pos_feats,
            label_names[1]: neg_feats,
        }

    def print_top_k(
        self,
        k: int = 20,
        *,
        label_names: Tuple[str, str] = ("M1", "M2"),
    ) -> None:
        """
        Pretty-print the top-k features for each class with weights.
        """
        feats = self.top_k_features(k=k, label_names=label_names)
        pos_name, neg_name = label_names

        print(f"\nTop {k} features → {pos_name} (label=1, higher → {pos_name}):")
        for w, wt in feats[pos_name]:
            print(f"  {w:>20s}  {wt:+.4f}")

        print(f"\nTop {k} features → {neg_name} (label=0, lower → {neg_name}):")
        for w, wt in feats[neg_name]:
            print(f"  {w:>20s}  {wt:+.4f}")


def baseline_discrimination(
    texts_m1: List[str],
    texts_m2: List[str],
    *,
    n_splits: int = 5,
    ngram_range: Tuple[int, int] = (1, 1),
    min_df: int = 1,
    max_features: Optional[int] = None,
    C: float = 1.0,
    penalty: str = "l2",
    max_iter: int = 2000,
    class_weight: Optional[str] = "balanced",
    random_state: int = 0,
    top_k_features_to_show: Optional[int] = None,
    label_names: Tuple[str, str] = ("M1", "M2"),
) -> Tuple[float, float, LogisticBoWDiscriminator]:
    """
    Convenience wrapper:
      - runs 5-fold CV and returns (mean_accuracy, mean_auc, model)
      - then fits the model on the FULL provided data so you can reuse it.
      - optionally prints top-k features (by coefficient weight) at training time.
    """
    model = LogisticBoWDiscriminator(
        ngram_range=ngram_range,
        min_df=min_df,
        max_features=max_features,
        C=C,
        penalty=penalty,
        max_iter=max_iter,
        class_weight=class_weight,
        random_state=random_state,
    )
    acc, auc = model.cross_validate(texts_m1, texts_m2, n_splits=n_splits)
    model.fit(texts_m1, texts_m2, top_k=top_k_features_to_show, label_names=label_names)
    return acc, auc, model