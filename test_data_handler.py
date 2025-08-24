from rulefetcher.data_handler import DataHandler
from rulefetcher.clusterer import Clusterer
from rulefetcher.rulekit_wrapper import RuleKitWrapper
import os

# === CONFIGURATION ===
csv_file_path = "./data/TCGA Pan-Cancer (PANCAN).csv"  # Use forward slashes for cross-platform safety

# Ensure output directory exists
output_dir = "./data"
os.makedirs(output_dir, exist_ok=True)

# Output paths
cluster_output_path = os.path.join(output_dir, "clustered_output.csv")
annotation_rules_path = os.path.join(output_dir, "cluster_annotation_rules.txt")

# Columns to rename and drop
rename_columns = {
    '_PATIENT': 'patient_id',
    'cancer type abbreviation': 'cancer_type_abbreviation',
    'OS': 'os_event',
    'OS.time': 'os_time',
    'DSS': 'dss_event',
    'DSS.time': 'dss_time',
    'DFI': 'dfi_event',
    'DFI.time': 'dfi_time',
    'PFI': 'pfi_event',
    'PFI.time': 'pfi_time',
    'treatment_outcome_first_course': 'treatment_outcome'
}
columns_to_drop = ['redaction']

# === EXECUTION ===
def main():
    try:
        # # Step 1: Load and prepare data
        # handler = DataHandler(csv_file_path)
        # df = handler.load_data()
        # df = handler.clean_columns()
        # df = handler.rename_columns(rename_columns)
        # df = handler.drop_columns(columns_to_drop)
        # df = handler.handle_missing_values("keep")
        # df = handler.encode_categoricals()
        # print("Data preprocessed successfully")

        # # Step 2: Apply clustering
        # clusterer = Clusterer(df, n_clusters=3)
        # clustered_df = clusterer.apply_kmodes()
        # print("KModes clustering applied")

        # # Step 3: Save clustered data
        # clusterer.save_clusters(cluster_output_path)
        # print(f"Clustered data saved to {cluster_output_path}")

        # # Step 4: Print and save summary
        # summary = clusterer.get_cluster_summary()
        # print("Cluster Summary:\n", summary)

        # clusterer.annotate_clusters(annotation_rules_path, max_depth=3)
        # print("Cluster annotation rules generated")

        wrapper = RuleKitWrapper("./data/clustered_output - Gender_Female_Age_less_50_VitalStat_Dead.csv", target_column="cluster")

        # 1) Load & clean
        wrapper.load_and_clean()

        # 2) Write ARFF & reload
        wrapper.write_arff("./data/tcga_dataset_clean.arff")
        wrapper.load_arff()

        # 3) Train models & evaluate
        wrapper.train_all()

        # 4) Quick summaries
        model_stats_df, metrics_df = wrapper.summary_frames()
        print(model_stats_df)
        print(metrics_df)

        # 5) Inspect rules (example: C2)
        c2_rules_dict = wrapper.get_cluster_rules("C2")
        print("C2 rules:", c2_rules_dict)

        correlation_rules_dict = wrapper.get_cluster_rules("Correlation")
        print("\nCorrelation rules:", correlation_rules_dict)

        rss_rules_dict = wrapper.get_cluster_rules("RSS")
        print("RSS rules:", rss_rules_dict)



    except Exception as e:
        print("Error during execution:", e)

if __name__ == "__main__":
    main()
