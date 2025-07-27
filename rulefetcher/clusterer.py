import pandas as pd
import numpy as np
from kmodes.kmodes import KModes
from sklearn.tree import DecisionTreeClassifier, export_text

# Rulefetcher Clusterer Module
# This module provides functionality for clustering categorical data using KModes
class Clusterer:
    def __init__(self, df: pd.DataFrame, n_clusters: int = 3):
        # Validate input DataFrame and number of clusters
        if df.empty:
            raise ValueError("Input DataFrame is empty.")
        self.df = df.copy()
        self.n_clusters = n_clusters
        self.cluster_labels = None

    def apply_kmodes(self):
        # Validate that the DataFrame has categorical data
        try:
            matrix = self.df.astype(str).to_numpy()
            km = KModes(n_clusters=self.n_clusters, init='Cao', verbose=0)
            self.cluster_labels = km.fit_predict(matrix)
            self.df['cluster'] = self.cluster_labels
        except Exception as e:
            raise RuntimeError(f"KModes clustering failed: {e}")
        return self.df

    def save_clusters(self, path: str):
        # Validate that the DataFrame has a 'cluster' column
        if self.df is None or 'cluster' not in self.df.columns:
            raise ValueError("Cluster column not found. Did you run apply_kmodes()?")
        self.df.to_csv(path, index=False)

    def get_cluster_summary(self):
        # Validate that the DataFrame has a 'cluster' column
        if 'cluster' not in self.df.columns:
            raise ValueError("Cluster column not found. Did you run apply_kmodes()?")
        return self.df['cluster'].value_counts()

    def annotate_clusters(self, path: str, max_depth=3):
        # Validate that the DataFrame has a 'cluster' column
        if 'cluster' not in self.df.columns:
            raise ValueError("Cluster column not found. Run apply_kmodes() first.")
        X = self.df.drop(columns=['cluster'])
        y = self.df['cluster']

        try:
            X_encoded = X.apply(lambda col: pd.factorize(col)[0])
            tree = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
            tree.fit(X_encoded, y)
            rules = export_text(tree, feature_names=list(X.columns))
            with open(path, "w") as f:
                f.write(rules)
            # Save the rules to a text file
            print("Cluster annotation rules saved to 'cluster_annotation_rules.txt'") 
        except Exception as e:
            raise RuntimeError(f"Error generating decision rules: {e}")
