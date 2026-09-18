# Medical Insurance Cost Prediction

An exploratory machine-learning project that estimates an applicant's health-insurance charges from basic demographic and lifestyle information. The repository includes the source dataset, a Jupyter notebook documenting the analysis and training workflow, and a persisted scikit-learn linear-regression model for local inference.

> **Important:** This is an educational example, not a pricing, underwriting, or medical decision-making system. Predictions are approximate estimates from a small historical dataset and must not be used to determine coverage, premiums, eligibility, or care.

## Contents

- [Project overview](#project-overview)
- [Repository layout](#repository-layout)
- [Dataset](#dataset)
- [Model and evaluation](#model-and-evaluation)
- [Getting started](#getting-started)
- [Run a prediction](#run-a-prediction)
- [Reproduce training](#reproduce-training)
- [Limitations and responsible use](#limitations-and-responsible-use)

## Project overview

The workflow in `medical_insurance_cost_prediction.ipynb`:

1. Loads and inspects the insurance dataset.
2. Explores distributions of age, sex, BMI, number of children, smoking status, region, and charges.
3. Encodes categorical features as numeric values.
4. Splits the data into training and test sets.
5. Trains a `sklearn.linear_model.LinearRegression` model.
6. Evaluates the model with R-squared and serializes both the model and expected feature order.

## Repository layout

| Path | Description |
| --- | --- |
| `medical_insurance_cost_prediction.ipynb` | End-to-end exploratory analysis, model training, evaluation, serialization, and an inference example. |
| `sample_data/insurance.csv` | Input dataset used by the notebook. |
| `health_insurance_model.pkl` | Saved trained `LinearRegression` estimator. |
| `health_insurance_model_columns.pkl` | Saved feature-order list required when preparing prediction inputs. |

## Dataset

`sample_data/insurance.csv` contains **1,338 rows** and seven columns. The prediction target is `charges`; every other column is a model feature.

| Column | Type | Description | Expected values / range |
| --- | --- | --- | --- |
| `age` | integer | Applicant age. | 18–64 in the supplied data. |
| `sex` | categorical | Recorded sex category. | `female`, `male` |
| `bmi` | float | Body mass index. | 15.96–53.13 in the supplied data. |
| `children` | integer | Number of children/dependents. | 0–5 |
| `smoker` | categorical | Smoking-status category. | `yes`, `no` |
| `region` | categorical | Geographic region category. | `northeast`, `northwest`, `southeast`, `southwest` |
| `charges` | float | Observed insurance charges; this is the target. | Monetary amount in the supplied dataset. |

The notebook reports no missing values in the provided dataset. Keep the raw categorical labels unchanged when using the helper below; it performs the required encoding.

## Model and evaluation

The saved artifact is a linear-regression model trained with an 80/20 train/test split using `random_state=2`. The notebook's recorded R-squared scores are:

| Split | R-squared |
| --- | ---: |
| Training | 0.7515 |
| Test | 0.7447 |

These figures are a single split's in-sample evaluation, not a guarantee of performance on future populations or production data. The model consumes features in this exact order:

```text
age, sex, bmi, children, smoker, region
```

### Categorical encodings

The persisted model was trained with the following mappings. They are intentionally reproduced exactly for compatibility with the included artifact.

| Feature | Mapping |
| --- | --- |
| `sex` | `male` → `0`, `female` → `1` |
| `smoker` | `yes` → `0`, `no` → `1` |
| `region` | `southeast` → `0`, `southwest` → `1`, `northeast` → `2`, `northwest` → `3` |

## Getting started

### Prerequisites

- Python 3.10 or later.
- `pip`.
- A virtual environment is recommended.

The notebook and inference example require NumPy, pandas, scikit-learn, joblib, Matplotlib, Seaborn, and Jupyter.

```bash
git clone <repository-url>
cd Medical-insurance-cost-prediction

python -m venv .venv
source .venv/bin/activate          # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install numpy pandas scikit-learn joblib matplotlib seaborn jupyter
```

### Open the notebook

Start Jupyter from the repository root so that the relative dataset and model paths resolve correctly:

```bash
jupyter notebook medical_insurance_cost_prediction.ipynb
```

Run the cells from top to bottom to inspect the data, regenerate the exploratory plots, train the model, view the evaluation, and recreate the serialized artifacts.

## Run a prediction

The following standalone example loads the included model and uses the same preprocessing as the notebook. Run it from the repository root after installing the dependencies.

```bash
python - <<'PY'
import joblib
import pandas as pd

model = joblib.load("health_insurance_model.pkl")
columns = joblib.load("health_insurance_model_columns.pkl")

applicant = {
    "age": 19,
    "sex": "female",
    "bmi": 27.9,
    "children": 0,
    "smoker": "yes",
    "region": "southwest",
}

sex_map = {"male": 0, "female": 1}
smoker_map = {"yes": 0, "no": 1}
region_map = {
    "southeast": 0,
    "southwest": 1,
    "northeast": 2,
    "northwest": 3,
}

row = {
    "age": applicant["age"],
    "sex": sex_map[applicant["sex"]],
    "bmi": applicant["bmi"],
    "children": applicant["children"],
    "smoker": smoker_map[applicant["smoker"]],
    "region": region_map[applicant["region"]],
}

input_frame = pd.DataFrame([row])[columns]
prediction = model.predict(input_frame)[0]
print(f"Approximate price: ${prediction:,.2f}")
PY
```

Use only the categories shown in the dataset table. An unknown category will not have a valid encoding and should be handled explicitly before calling the model. Loading pickle/joblib files executes deserialization code; only load model artifacts you trust.

## Reproduce training

To regenerate the shipped artifacts:

1. Set up the environment described in [Getting started](#getting-started).
2. Launch the notebook from the repository root.
3. Run all cells in `medical_insurance_cost_prediction.ipynb`.
4. The final persistence cells write `health_insurance_model.pkl` and `health_insurance_model_columns.pkl` to the repository root.

The notebook uses `train_test_split(..., test_size=0.2, random_state=2)`, fits `LinearRegression`, and saves `list(X.columns)` alongside the estimator. Save the feature list with every retrained model; column order is part of the model contract.

## Limitations and responsible use

- The dataset is small and may not represent the people, locations, healthcare systems, or time period where the model is applied.
- A linear model cannot capture all interactions and nonlinear drivers of insurance charges.
- `sex`, smoking status, and region are sensitive or context-dependent attributes. Do not use this project for automated decisions affecting people.
- Inputs outside the ranges and categories represented in the source data are extrapolations and can produce unreliable estimates.
- The notebook's preprocessing is hand-coded. If the model is extended, consider a scikit-learn `Pipeline` with explicit validation and a documented data-governance process.

## License

No license file is included in this repository. Do not assume permission to reuse, redistribute, or deploy the code, dataset, or trained artifacts until a license is added by the project owner.
