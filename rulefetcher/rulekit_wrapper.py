from __future__ import annotations

import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from rulekit.arff import read_arff
from rulekit.classification import RuleClassifier
from rulekit.rules import RuleSet, ClassificationRule
from rulekit.params import Measures


class RuleKitWrapper:
    """
    End-to-end helper for
      1. loading & cleaning CSV
      2. writing/reading ARFF
      3. training RuleKit models (multiple measures)
      4. computing evaluation metrics
      5. extracting human-readable rules per cluster
    """

    # ---------- construction & basic IO ---------- #
    def __init__(self, csv_path: str | Path, target_column: str = "cluster") -> None:
        self.csv_path = Path(csv_path)
        self.target_column = target_column
        self.df_raw: pd.DataFrame | None = None
        self.df_clean: pd.DataFrame | None = None
        self.arff_path: Path | None = None
        self.df_arff: pd.DataFrame | None = None
        self.label_encoder: LabelEncoder | None = None

    @staticmethod
    def _clean_nominal(value):
        if isinstance(value, str):
            return (
                value.strip()
                .replace(" ", "_")
                .replace(",", "")
                .replace("/", "_")
                .replace("|", "_")
                .replace("=", "_")
                .replace("[", "_")
                .replace("]", "_")
                .replace("{", "_")
                .replace("}", "_")
                .replace(":", "_")
                .replace(";", "_")
                .replace("\\", "_")
                .replace('"', "_")
                .replace("'", "_")
            )
        return value

    def load_and_clean(self) -> pd.DataFrame:
        self.df_raw = pd.read_csv(self.csv_path)
        self.df_clean = self.df_raw.map(self._clean_nominal)
        return self.df_clean

    # ---------- ARFF helpers ---------- #
    def write_arff(self, out_path: str | Path, relation: str = "dataset") -> Path:
        if self.df_clean is None:
            raise RuntimeError("Data not cleaned – call load_and_clean() first.")

        out_path = Path(out_path)
        with out_path.open("w") as f:
            f.write(f"@RELATION {relation}\n\n")
            for col in self.df_clean.columns:
                col_data = self.df_clean[col].dropna()
                if pd.api.types.is_numeric_dtype(col_data):
                    f.write(f"@ATTRIBUTE {col} NUMERIC\n")
                else:
                    values = sorted(col_data.unique())
                    f.write(f"@ATTRIBUTE {col} {{{','.join(map(str, values))}}}\n")
            f.write("\n@DATA\n")
            for _, row in self.df_clean.iterrows():
                row_str = ",".join("?" if pd.isna(x) else str(x) for x in row)
                f.write(row_str + "\n")

        self.arff_path = out_path
        return out_path

    def load_arff(self) -> pd.DataFrame:
        if self.arff_path is None:
            raise RuntimeError("ARFF not written – call write_arff() first.")
        self.df_arff = read_arff(str(self.arff_path))

        # encode categorical target
        self.label_encoder = LabelEncoder()
        self.df_arff[self.target_column] = self.label_encoder.fit_transform(
            self.df_arff[self.target_column]
        )
        self.df_arff[self.target_column] = self.df_arff[self.target_column].astype(int)
        return self.df_arff

    # ---------- training & evaluation ---------- #
    MEASURE_MAP = {
        "C2": Measures.C2,
        "Correlation": Measures.Correlation,
        "RSS": Measures.RSS,
    }

    def _split_xy(self) -> Tuple[pd.DataFrame, pd.Series]:
        if self.df_arff is None:
            raise RuntimeError("ARFF not loaded – call load_arff() first.")

        X = (
            self.df_arff.drop(columns=["sample", "patient_id", self.target_column])
            .select_dtypes(include=[np.number])
            .copy()
        )
        y = self.df_arff[self.target_column].astype(int)
        mask = y.notna()
        return X[mask], y[mask]

    def train_all(
        self, test_size: float = 0.2, random_state: int = 42
    ) -> None:  # stores results internally
        X, y = self._split_xy()
        self.results: Dict[str, Dict] = {}

        for name, measure in self.MEASURE_MAP.items():
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            clf = RuleClassifier(
                induction_measure=measure,
                pruning_measure=measure,
                voting_measure=measure,
            )
            clf.fit(X_tr, y_tr)

            y_pred, class_metrics = clf.predict(X_te, return_metrics=True)
            metrics_df, cm = self._prediction_metrics(
                name, y_pred, y_te, class_metrics
            )
            stats_df = self._ruleset_stats(name, clf.model)

            self.results[name] = dict(
                classifier=clf,
                ruleset=clf.model,
                metrics_df=metrics_df,
                confusion_matrix=cm,
                stats_df=stats_df,
            )

    # ---------- utility metric helpers ---------- #
    @staticmethod
    def _prediction_metrics(
        measure: str,
        y_pred: np.ndarray,
        y_true: pd.Series,
        classification_metrics: Dict,
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        cm = metrics.confusion_matrix(y_true, y_pred)

        base = {
            "Measure": measure,
            "Accuracy": metrics.accuracy_score(y_true, y_pred),
            "MAE": metrics.mean_absolute_error(y_true, y_pred),
            "Kappa": metrics.cohen_kappa_score(y_true, y_pred),
            "Balanced accuracy": metrics.balanced_accuracy_score(y_true, y_pred),
            "Precision (macro)": metrics.precision_score(
                y_true, y_pred, average="macro"
            ),
            "Recall (macro)": metrics.recall_score(y_true, y_pred, average="macro"),
            "F1-score (macro)": metrics.f1_score(y_true, y_pred, average="macro"),
            "F1-score (weighted)": metrics.f1_score(
                y_true, y_pred, average="weighted"
            ),
            "Rules per example": classification_metrics["rules_per_example"],
            "Voting conflicts": classification_metrics["voting_conflicts"],
        }

        if len(np.unique(y_true)) == 2:
            tn, fp, fn, tp = cm.ravel()
            sens = tp / (tp + fn)
            spec = tn / (tn + fp)
            npv = tn / (tn + fn)
            ppv = tp / (tp + fp)
            base.update(
                {
                    "Sensitivity": sens,
                    "Specificity": spec,
                    "NPV": npv,
                    "PPV": ppv,
                    "psep": ppv + npv - 1,
                    "Fall-out": fp / (fp + tn),
                    "Youden's J": sens + spec - 1,
                    "Lift": (ppv) / ((tp + fn) / (tp + tn + fp + fn)),
                    "F-measure": 2 * tp / (2 * tp + fp + fn),
                    "Fowlkes-Mallows": metrics.fowlkes_mallows_score(y_true, y_pred),
                    "FN": fn,
                    "FP": fp,
                    "TP": tp,
                    "TN": tn,
                    "Geo-mean": math.sqrt(spec * sens),
                }
            )

        return (
            pd.DataFrame.from_records([base], index="Measure"),
            cm,
        )

    @staticmethod
    def _ruleset_stats(measure: str, ruleset_model: RuleSet) -> pd.DataFrame:
        rules = getattr(ruleset_model, "rules", None)
        return pd.DataFrame.from_records(
            [
                {
                    "Measure": measure,
                    "Number of Rules": len(rules) if rules is not None else np.nan,
                    "Rule info": str(rules[:3]) + "..." if rules else "Unavailable",
                }
            ],
            index="Measure",
        )

    # ---------- rule inspection ---------- #
    @staticmethod
    def _get_rules_dict(rules: List[ClassificationRule]) -> Dict[int, List[str]]:
        out: Dict[int, List[str]] = defaultdict(list)
        for r in rules:
            txt = str(r)
            m = re.search(r"cluster\s*=\s*\{(\d+)\}", txt)
            if m:
                cid = int(m.group(1))
                out[cid].append(txt.split("THEN cluster =")[0].replace("IF", "").replace(" AND", ";").strip())
        return dict(out)

    def get_cluster_rules(self, measure_name: str) -> Dict[int, List[str]]:
        """
        Returns {cluster_id: [rule strings]} for the requested measure ('C2', 'Correlation', 'RSS')
        """
        if not hasattr(self, "results") or measure_name not in self.results:
            raise RuntimeError("Models not trained or wrong measure key.")
        ruleset: RuleSet = self.results[measure_name]["ruleset"]
        return self._get_rules_dict(ruleset.rules)

    # ---------- convenience exports ---------- #
    def summary_frames(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Concatenate all metrics and rule-stats data frames for quick inspection.
        Returns (model_stats_df, metrics_df)
        """
        if not hasattr(self, "results"):
            raise RuntimeError("train_all() has not been run yet.")
        stats = pd.concat([v["stats_df"] for v in self.results.values()])
        mets = pd.concat([v["metrics_df"] for v in self.results.values()])
        return stats, mets


