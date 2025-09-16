# train.py
from pathlib import Path
import pandas as pd

# CI-safe backend for matplotlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, ConfusionMatrixDisplay, confusion_matrix

# --- Paths & folders ---
DATA_PATH = Path("Data") / "drug200.csv"
RESULTS_DIR = Path("Results")
MODEL_DIR = Path("Model")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# --- Load data ---
data = pd.read_csv(DATA_PATH)

# Expected columns (for reference): Age, Sex, BP, Cholesterol, Na_to_K, Drug
X = data.drop(columns=["Drug"], axis=1)
y = data["Drug"]

# --- Train/test split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --- Column indices (by position) ---
# 0: Age (num), 1: Sex (cat), 2: BP (cat), 3: Cholesterol (cat), 4: Na_to_K (num)
cat_columns = [1, 2, 3]
num_columns = [0, 4]

# --- Preprocessing pipelines ---
num_pipe = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

cat_pipe = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder()),
    ]
)

preprocess = ColumnTransformer(
    transformers=[
        ("num", num_pipe, num_columns),
        ("cat", cat_pipe, cat_columns),
    ]
)

# --- Model pipeline ---
pipe = Pipeline(
    steps=[
        ("preprocessing", preprocess),
        ("model", RandomForestClassifier(n_estimators=100, random_state=125)),
    ]
)

# --- Fit ---
pipe.fit(X_train, y_train)

# --- Evaluate ---
pred = pipe.predict(X_test)
accuracy = accuracy_score(y_test, pred)
f1 = f1_score(y_test, pred, average="macro")
print("Accuracy:", f"{round(accuracy, 2) * 100}%", "F1:", round(f1, 2))

# --- Metrics file (align with Makefile) ---
# If your Makefile expects module_metrics.txt, write that name:
metrics_path = RESULTS_DIR / "module_metrics.txt"
with open(metrics_path, "w") as f:
    f.write(f"Accuracy = {round(accuracy, 2)}, F1 Score = {round(f1, 2)}.\n")

# --- Confusion matrix plot ---
cm = confusion_matrix(y_test, pred, labels=pipe.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=pipe.classes_)
disp.plot()
plt.tight_layout()
plt.savefig(RESULTS_DIR / "model_results.png", dpi=120, bbox_inches="tight")
plt.close()

# --- Persist model (skops) ---
# Requires `skops` in requirements.txt
import skops.io as sio
model_path = MODEL_DIR / "drug_pipeline.skops"
sio.dump(pipe, model_path)

# Test loading (optional)
_ = sio.load(model_path, trusted=["numpy.dtype"])
