<div align="center">
  <!-- Optional: replace with your project logo -->
  <img src="https://github.com/Ibne-Sayad/Rule-Based-Learning-RuleKit/blob/test/readmeimg/RuleFetcherlogo.png" width="460" alt="RuleFetcher Logo">
  <div><b>RuleFetcher</b></div>
</div>

***

# 🧠 RuleFetcher — Rule-Based Learning for Interpretable Subgroup Discovery

RuleFetcher is a modular and extensible pipeline designed for analyzing complex, mixed-type datasets—those containing both numerical and categorical features, often seen in clinical, multi-omics, and survey-based research. Its primary goal is to uncover interpretable patterns in the form of human-readable rules that explain how data points are grouped into clusters.

Unlike black-box machine learning models, RuleFetcher emphasizes transparency and reproducibility. It keeps missing values explicit and visible throughout the pipeline using standard markers like NaN in Python and ? in ARFF, rather than silently imputing them. This approach helps ensure that rule learning reflects real data uncertainty.

The pipeline supports:

- Data preprocessing and formatting
- Clustering using K-Modes or Decision Trees
- Conversion to ARFF for RuleKit compatibility
- Rule-based learning using multiple induction and evaluation strategies (C2, Correlation, RSS)
- Mapping learned rules back to cluster identities
- Runtime benchmarking with synthetic data expansion
- Semantic summaries for interpretation

By combining unsupervised clustering with rule-based explanation, RuleFetcher offers researchers a transparent, scalable, and reproducible approach to exploring subgroups in large datasets especially when understanding "why" a cluster exists is just as important as identifying it.

---
## Workflow
<div align="center">
  <!-- Optional: replace with your project logo -->
  <img src="https://github.com/Ibne-Sayad/Rule-Based-Learning-RuleKit/blob/test/readmeimg/Workflow.png" width="660" alt="RuleFetcher Logo">
  <div><b>Workflow</b></div>
</div>

## 🚀 Overview

RuleFetcher helps you:

- **Dataset** Supports CSV file
- **Cluster** Optional clustering large mixed-type data (e.g., clinical + multi-omics).
- **Training data** Train dataset to apply Rulekit
- **Extract rules** (support, confidence, coverage) with RuleKit.
- **Map rules ⇄ clusters** to produce semantic subgroup descriptions.
- **Benchmark runtime** at scale (e.g., simulate 100/200/… variables).

---

## 🎯 Key Features

- 🔍 **Mixed-Type Support**: Numerical and categorical features handled out of the box.
- 🧩 **Explicit Missingness**: Keep `NaN`/`?` semantics through ARFF and rule matching.
- 🧮 **Clustering + Annotation**: KModes/DecisionTree-based cluster annotation into rules.
- 📜 **Rule Induction**: Multiple measures (e.g., C2, Correlation, RSS) via RuleKit.
- 📦 **End-to-End Pipeline**: Data cleaning → ARFF → training → metrics → summaries.
- 📊 **Benchmarking**: Synthetic column expansion to stress-test runtime.
- 🧪 **Unit-Tested**: Utility functions and metrics validated with tests.

---

## 🧰 Tech Stack

- **Python**: pandas, numpy, scikit-learn (KModes if applicable), matplotlib.
- **RuleKit**: Java/Python bridge for rule induction & evaluation.
- **I/O**: CSV ↔︎ ARFF converters (preserve `?` for missing).
- **CLI**: Scripts for pipeline runs and benchmarking.

---

## 🔧 Prerequisites

# Before using RuleFetcher, ensure you have:

- Python ≥ 3.8

- Java installed and RuleKit.jar available

- Required Python packages installed (automatically handled during installation)

- Input dataset in CSV format
---

## 📦 Installation

### Install the package directly from GitHub

```bash
pip install "git+https://github.com/Ibne-Sayad/Rule-Based-Learning-RuleKit.git"
```
---

## 📦 How to use RuleFetcer

### How to import and use

```bash
from rulefetcher.data_handler import DataHandler
from rulefetcher.clusterer import Clusterer
from rulefetcher.rulekit_wrapper import RuleKitWrapper
import os


# Directory name
directory_name = "outputdata"
# Create directory
os.makedirs(directory_name, exist_ok=True)

# === Step 1: Load and preprocess data ===
handler = DataHandler("data/clustered_output.csv") # Your input CSV file
df = handler.load_data()
df = handler.clean_columns()
# df = handler.rename_columns({...})         # Optional renaming
df = handler.handle_missing_values("keep") # Keep or drop NAs
df = handler.encode_categoricals()

# === (Optional)Step 2: Apply clustering ===
# === Skip this step if your dataset has clustering  ===
clusterer = Clusterer(df, n_clusters=4)
clustered_df = clusterer.apply_kmodes()

clusterer.save_clusters("outputdata/clustered_output.csv")
clusterer.annotate_clusters("outputdata/cluster_rules.txt")

# === Step 3: Rule extraction using RuleKit ===
wrapper = RuleKitWrapper("outputdata/clustered_output.csv", target_column="cluster")
wrapper.load_and_clean()
wrapper.write_arff("outputdata/data.arff")
wrapper.load_arff()
wrapper.train_all()

# === Step 4: Inspect results ===
model_stats, metrics = wrapper.summary_frames()
rules = wrapper.get_cluster_rules("C2")  # Or "Correlation", "RSS", etc.
print("C2 rules:", rules)

```
---
## 🛡️ License 

This project is licensed under the MIT License. See LICENSE for details.

---
## 📬 Contact 

- Author: Ibne Sayad
- Email: ibne.sayad@fau.de
- GitHub: github.com/Ibne-Sayad
- LinkedIn: linkedin.com/in/ibne-sayad


