#!/usr/bin/env python3
"""
pipeline.py – End-to-end workflow for TCGA Pan-Cancer data

Usage:
    python pipeline.py \
        --csv ./data/TCGA_PanCancer_PANCAN.csv \
        --output-dir ./data \
        --n-clusters 3
"""
import argparse
import os
from pathlib import Path

from rulefetcher.data_handler import DataHandler
from rulefetcher.clusterer import Clusterer
from rulefetcher.rulekit_wrapper import RuleKitWrapper


# ---------- CONFIGURATION HELPERS ----------
RENAME_COLUMNS = {
    "_PATIENT": "patient_id",
    "cancer type abbreviation": "cancer_type_abbreviation",
    "OS": "os_event",
    "OS.time": "os_time",
    "DSS": "dss_event",
    "DSS.time": "dss_time",
    "DFI": "dfi_event",
    "DFI.time": "dfi_time",
    "PFI": "pfi_event",
    "PFI.time": "pfi_time",
    "treatment_outcome_first_course": "treatment_outcome",
}
COLUMNS_TO_DROP = ["redaction"]


# ---------- PIPELINE ----------
def run_pipeline(csv_path: Path, output_dir: Path, n_clusters: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    cluster_output = output_dir / "clustered_output.csv"
    annotation_rules_path = output_dir / "cluster_annotation_rules.txt"
    arff_path = output_dir / "tcga_dataset_clean.arff"

    # 1. Load & preprocess
    # 1. Load & preprocess
    handler = DataHandler(str(csv_path))
    df = handler.load_data()
    handler.clean_columns()
    handler.rename_columns(RENAME_COLUMNS)
    handler.drop_columns(COLUMNS_TO_DROP)
    handler.handle_missing_values("keep")
    handler.encode_categoricals()
    df = handler.df  # final cleaned dataframe

    print("[1] Data loaded & pre-processed")

    # 2. Clustering
    clusterer = Clusterer(df, n_clusters=n_clusters)
    clusterer.apply_kmodes()
    clusterer.save_clusters(str(cluster_output))
    print(f"[2] KModes clustering done {cluster_output}")

    # 3. Cluster summaries / annotation
    print("[3] Cluster summary\n", clusterer.get_cluster_summary())
    clusterer.annotate_clusters(str(annotation_rules_path), max_depth=3)
    print(f"[3] Annotation rules saved {annotation_rules_path}")

    # 4. RuleKit modelling
    wrapper = RuleKitWrapper(str(cluster_output), target_column="cluster")

    # run steps sequentially
    wrapper.load_and_clean()                      # returns DataFrame (ignored here)
    wrapper.write_arff(str(arff_path))            # creates ARFF file
    wrapper.load_arff()                           # loads ARFF back in
    wrapper.train_all()                           # trains RuleKit models

    print(f"[4] RuleKit ARFF written {arff_path}")

    model_stats_df, metrics_df = wrapper.summary_frames()
    print("[5] Model stats:\n", model_stats_df)
    print("[5] Metrics:\n", metrics_df)

    # Example: inspect one cluster’s rules
    example_cluster = model_stats_df.index[0]  # e.g., "C0"
    print(f"[6] Example rules for {example_cluster}:")
    print(wrapper.get_cluster_rules(example_cluster))


# ---------- CLI ----------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the TCGA clustering pipeline")
    parser.add_argument("--csv", required=True, help="Input CSV path")
    parser.add_argument(
        "--output-dir", default="./data", help="Directory to write outputs"
    )
    parser.add_argument(
        "--n-clusters", type=int, default=3, help="Number of K-Modes clusters"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(Path(args.csv), Path(args.output_dir), args.n_clusters)
