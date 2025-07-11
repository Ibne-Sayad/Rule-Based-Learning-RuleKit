# Rule-Based-Learning-RuleKit

**Rule-Based-Learning-RuleKit** is a Python package that enables interpretable, rule-based clustering and classification of patient subgroups, especially for multi-omics biomedical data such as TCGA PanCancer datasets. It uses KModes clustering and integrates [RuleKit](https://github.com/irb-jku/RuleKit) to extract semantic rules for each subgroup, enabling explainable analysis of high-dimensional categorical data.

---

## 📌 Features

- 📊 **Preprocessing** of patient datasets: column renaming, cleaning, encoding
- 🔍 **Clustering** with KModes for categorical subgroup discovery
- 📜 **Rule-based learning** using RuleKit for human-readable explanations
- ✅ Automatic generation of ARFF files
- 📁 Outputs:
  - Clustered dataset (CSV)
  - RuleKit-compatible ARFF file
  - Cluster annotation rules
  - Evaluation statistics (precision, coverage, F1, etc.)

---

## 📦 Installation

```bash
pip install git+https://github.com/Ibne-Sayad/Rule-Based-Learning-RuleKit.git
