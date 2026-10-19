# EcoMaint AI — Samsung Innovation Campus (Assignment 1)

**Team (Group 1):** Erkan Uslu, Kaan Akbaş, Enver Meriç Erdoğuş, İsmail Emre Uzun

## Project idea
EcoMaint AI is a small machine-learning project that classifies whether a machine is in a **failure** or **normal** state using its operating data. Its purpose is to support maintenance decisions in manufacturing.

- **SDG 9:** Industry, Innovation and Infrastructure
- **SDG 12:** Responsible Consumption and Production
- **Target users:** machine operators and maintenance teams
- **Assignment 1 scope:** dataset inspection (EDA), initial model comparison and project proposal. A web application is planned for Assignment 2.

## Dataset

**AI4I 2020 Predictive Maintenance Dataset**, S. Matzka (2020), UCI Machine Learning Repository. DOI: [10.24432/C5HS5C](https://doi.org/10.24432/C5HS5C)

[Official dataset page](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset)

The dataset has **10,000 rows** and **14 original columns**. It is **synthetic** (simulated data designed to resemble a real industrial environment). The target is `Machine failure` (0/1).

The code uses product type, air and process temperatures, rotational speed, torque and tool wear. It also calculates temperature difference, mechanical power and torque multiplied by tool wear. Identifier columns and the five failure-type **target** labels are not used for prediction, to reduce data leakage.

## How to run

1. Install **Python 3.10+**.
2. Download this repository (or clone it) and open a terminal in the folder.
3. Run:

```bash
python -m pip install -r requirements.txt
python train_model.py
```

The script automatically tries to download `ai4i2020.csv` from UCI on first run. **If downloading fails**, manually download the dataset from the UCI page, unzip it, and place `ai4i2020.csv` in the `data/` folder. Run the command again.

### Outputs

When the script finishes, it creates files inside `results/`:

| File | What it contains |
| --- | --- |
| `failure_distribution.png` | Chart of normal vs. failed records |
| `failure_modes.png` | Chart of five failure-mode label counts |
| `sensor_statistics.csv` | Basic sensor statistics |
| `missing_values.csv` | Missing-value counts |
| `eda_summary.json` | Counts, duplicate check, rpm–torque correlation |
| `model_metrics.csv` | Measured Accuracy, Precision, Recall, F1, ROC-AUC, confusion-matrix counts |
| `confusion_*.png` | Confusion matrix for each model |

## First experiments

We use an **80/20 stratified train/test split** (`random_state=42`). Three simple approaches are compared:

- Logistic Regression (with balanced class weights)
- Random Forest (with balanced class weights)
- XGBoost (with a weight for rare failure cases)

**Important:** The PDF draft contains preliminary accuracy figures, which have *not* been independently reproduced in this package. **Run this script and use the numbers in `results/model_metrics.csv` to update the PDF before submission.** We do not claim a particular accuracy until the code has been run on the original dataset.

Because machine failures are rare, we compare **recall, precision and F1-score**, not just accuracy.

## Limitations and future work

This is a **failure-state classifier**, not a validated prediction of how long until a machine breaks. The synthetic dataset does not prove that the model will work equally well on real factory data. A wrong prediction could create unnecessary maintenance or miss a genuine problem. The results are for learning and should not replace an engineer's judgment.

For Assignment 2, we plan to add a simple application interface. Additional explanations (SHAP) or maintenance recommendations are optional future features, not implemented in Assignment 1.

## Repository files

- `train_model.py` — main EDA and machine-learning code.
- `requirements.txt` — Python libraries needed.
- `data/README.md` — data source and download instructions.
- `results/` — charts and tables created after running the code.
- `README.md` — this guide.

**Submission note:** Include the real URL of this GitHub repository in the 1–2 page PDF. Do not leave a placeholder URL.
