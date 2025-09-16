import pandas as pd 

data = pd.read_csv('Data\drug200.csv')
data.head()

from sklearn.model_selection import train_test_split
x=data.drop(columns=['Drug'],axis=1)
y=data['Drug']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)


from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

cat_columns = [1,2,3]
num_columns = [0,4]

transform = ColumnTransformer(transformers=[("encoder", OrdinalEncoder(), cat_columns),
        ("num_imputer", SimpleImputer(strategy="median"), num_columns),
        ("num_scaler", StandardScaler(), num_columns),])


pipe = Pipeline(
    steps=[
        ("preprocessing", transform),
        ("model", RandomForestClassifier(n_estimators=100, random_state=125)),
    ]
)

pipe.fit(x_train, y_train)


from sklearn.metrics import accuracy_score, f1_score

predictions = pipe.predict(x_test)
accuracy = accuracy_score(y_test, predictions)
f1 = f1_score(y_test, predictions, average="macro")

print("Accuracy:", str(round(accuracy, 2) * 100) + "%", "F1:", round(f1, 2))


with open("Results/metrics.txt", "w") as outfile:
    outfile.write(f"\nAccuracy = {round(accuracy,2)}, F1 Score = {round(f1,2)}.")

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

cm = confusion_matrix(y_test, predictions, labels=pipe.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=pipe.classes_)
disp.plot()
plt.savefig("Results/model_results.png", dpi=120)


import skops.io as sio

sio.dump(pipe, "Model/drug_pipeline.skops")

sio.load("Model/drug_pipeline.skops",trusted=["numpy.dtype"])
