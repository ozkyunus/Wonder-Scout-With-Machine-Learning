import joblib
import pandas as pd
import numpy as np

df = pd.read_csv("Fc2425Corr.csv")

import sys
sys.path.append('/Users/yunusemreozkaya/PycharmProjects/PythonProject11/machine_learning')
from ScoutMLPipeline import players_data_prep

X, y = players_data_prep(df)

random_user = X.sample(1, random_state=89)

new_model = joblib.load("best_lgb_model.pkl")

y_pred_log = new_model.predict(random_user)
y_pred_real = np.exp(y_pred_log)

results = random_user.copy()
results["PREDICTED_VALUE"] = y_pred_real

print(results)

