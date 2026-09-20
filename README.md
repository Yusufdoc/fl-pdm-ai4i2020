# FL-PdM-AI4I2020

**Federated Learning for Privacy-Preserving Predictive Maintenance in Non-IID Multi-Plant Manufacturing**

Reproducibility code for:
> Bekkour, Y., Boutayeb, A., Chamat, A., En-nadi, A. (2026). *Federated Learning for Privacy-Preserving Predictive Maintenance in Non-IID Multi-Plant Manufacturing Using the AI4I 2020 Dataset.* Discover Artificial Intelligence. Submission ID: 5a77aa9b-22fd-4f7c-8c3c-ad45620fa69b

---

## Dataset

**AI4I 2020 Predictive Maintenance Dataset** (UCI Machine Learning Repository)
- License: CC BY 4.0
- DOI: https://doi.org/10.24432/C5HS5C
- Downloaded automatically by the script on first run

**Non-IID partitioning by product type:**

| Plant | Type | Records | Failure rate |
|-------|------|---------|--------------|
| A     | L    | 6,000   | 3.9%         |
| B     | M    | 2,997   | 2.8%         |
| C     | H    | 1,003   | 2.1%         |

---

## Requirements

```bash
pip install torch scikit-learn pandas numpy matplotlib seaborn requests lightgbm xgboost
```

Tested with: Python 3.9+, PyTorch 2.0+, scikit-learn 1.3+

---

## Usage

```bash
# Run all experiments (FedAvg, FedProx, Local-Only, Centralized, Baselines)
python fl_pdm_ai4i2020.py --mode all

# Run individual experiments
python fl_pdm_ai4i2020.py --mode fedavg
python fl_pdm_ai4i2020.py --mode fedprox
python fl_pdm_ai4i2020.py --mode centralized
python fl_pdm_ai4i2020.py --mode local
python fl_pdm_ai4i2020.py --mode baselines

# Custom seeds
python fl_pdm_ai4i2020.py --mode all --seeds 42 123 456 789 2024
```

---

## Reproducibility

All experiments use **5 fixed seeds**: 42, 123, 456, 789, 2024

**Key implementation details:**
- PdMNet architecture: Input(5) → FC(16, ReLU, BN, 0.3) → FC(8, ReLU, BN, 0.3) → FC(1, Sigmoid) — 2,753 parameters
- Classification threshold: calibrated on validation set via **Youden Index** (avoids test-set leakage)
- FedAvg aggregation: weighted by number of training samples per plant
- FedProx: μ = 0.01
- Communication rounds: 20 | Local epochs per round: 5 | Batch size: 32
- Optimizer: AdamW, lr=1e-3, weight_decay=1e-4

---

## Expected Results (Table 5)

| Scenario     | Accuracy | F1-score | AUC-ROC | Recall |
|-------------|----------|----------|---------|--------|
| Local Only (avg) | 0.878 | 0.334 | 0.933 | 0.814 |
| FedAvg | 0.937 ± 0.019 | 0.477 ± 0.063 | 0.963 ± 0.008 | 0.819 ± 0.061 |
| FedProx | 0.881 ± 0.054 | 0.343 ± 0.073 | 0.949 ± 0.013 | 0.842 ± 0.078 |
| Centralized | 0.927 ± 0.037 | 0.475 ± 0.127 | 0.967 ± 0.012 | 0.866 ± 0.048 |

---

## Output Files

Results are saved to `results/`:
- `all_results.json` — Complete per-seed metrics
- `fig2_f1_comparison.png` — F1-score bar chart
- `fig3_auc_comparison.png` — AUC-ROC bar chart
- `fedavg_convergence_f1.png` — FedAvg convergence (F1)
- `fedavg_convergence_auc.png` — FedAvg convergence (AUC)
- `fedprox_convergence_f1.png` — FedProx convergence (F1)

---

## License

Code: MIT License
Dataset: CC BY 4.0 (see UCI repository)
