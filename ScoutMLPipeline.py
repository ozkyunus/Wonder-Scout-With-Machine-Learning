import warnings
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.model_selection import GridSearchCV, cross_validate, RandomizedSearchCV, validation_curve
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import RobustScaler
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, r2_score, make_scorer
from sklearn.model_selection import train_test_split
import joblib
import optuna


def grab_col_names(df, cat_th = 10, car_th = 20):
  cat_cols = [col for col in df.columns if df[col].dtypes == "O"]
  num_cols = [col for col in df.columns if df[col].dtypes != "O"]
  num_but_cat = [col for col in num_cols if df[col].nunique() < cat_th]
  cat_but_car = [col for col in cat_cols if df[col].nunique() > car_th]
  cat_cols = cat_cols + num_but_cat
  cat_cols = [col for col in cat_cols if col not in cat_but_car]
  num_cols = [col for col in num_cols if col not in num_but_cat]
  print(f"Observations: {df.shape[0]}")
  print(f"Variables: {df.shape[1]}")
  print(f'cat_cols: {len(cat_cols)}')
  print(f'num_cols: {len(num_cols)}')
  print(f'cat_but_car: {len(cat_but_car)}')
  print(f'num_but_cat: {len(num_but_cat)}')

  return cat_cols, num_cols, cat_but_car, num_but_cat


def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
  quartile1 = dataframe[col_name].quantile(q1)
  quartile3 = dataframe[col_name].quantile(q3)
  interquantile_range = quartile3 - quartile1
  up_limit = quartile3 + 1.5 * interquantile_range
  low_limit = quartile1 - 1.5 * interquantile_range
  return low_limit, up_limit


def check_outlier(dataframe, col_name):
  low_limit, up_limit = outlier_thresholds(dataframe, col_name)
  if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
    return True
  else:
    return False


def replace_with_thresholds(dataframe, variable, q1=0.05, q3=0.95):
  low_limit, up_limit = outlier_thresholds(dataframe, variable, q1=0.05, q3=0.95)
  dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
  dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


def bag_rares(data, columns, percentage):
    data = data.copy()
    for col in columns:
      rares = data[col].value_counts(normalize=True) < (percentage / 100)
      rare_names = rares[rares].index.tolist()
      data.loc[data[col].isin(rare_names), col] = "Rare"
    return data


def players_data_prep(df):
  cat_cols, num_cols, cat_but_car, num_but_cat = grab_col_names(df)

  to_move = [
    'International_reputation', 'Weak_foot', 'Skill_moves'
  ]

  cat_cols = [col for col in cat_cols if col not in to_move]
  num_cols = num_cols + [col for col in to_move if col in df.columns]

  replace_with_thresholds(df, "Value")

  df = bag_rares(df, cat_cols, percentage=5)

  df = pd.get_dummies(df, columns=cat_cols, dtype=int)

  df_24 = df[df["Value"].notna()].copy()
  df_25 = df[df["Value"].isna()].copy()
  df_25 = df_25.drop("Value", axis=1)

  X = df_24.drop(["Value"], axis=1)
  y = np.log(df_24["Value"])

  return X, y


def train_lgbm_regressor(X, y, test_size=0.2, random_state=42, params=None):
  """
  LightGBM Regressor ile model eğitir ve RMSE döndürür.

  Parameters
  ----------
  X : DataFrame veya ndarray
      Özellikler (feature matrix)
  y : Series veya ndarray
      Hedef değişken
  test_size : float, optional
      Test setinin oranı (default=0.2)
  random_state : int, optional
      Rastgelelik kontrolü
  params : dict, optional
      LightGBM parametreleri

  Returns
  -------
  model : LGBMRegressor
      Eğitilmiş model
  rmse : float
      Root Mean Squared Error değeri
  """

  # Varsayılan parametreler
  default_params = {
    "objective": "root_mean_squared_error",
    "verbose": -1
  }

  if params:
    default_params.update(params)

  # Train-test split
  X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=random_state
  )

  # Model oluşturma
  model = lgb.LGBMRegressor(**default_params)
  model.fit(X_train, y_train)

  # Tahmin ve RMSE hesaplama
  mse = mean_squared_error(y_test, model.predict(X_test))
  rmse = np.sqrt(mse)

  print(f"RMSE: {rmse:.4f}")
  return model, rmse

def drop_calculate():
    attempts = {}
    best_score = 0.5
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    from tqdm.auto import tqdm
    for i in tqdm(range(1, len(X_train.columns))):
      drop_col = sorted.iloc[len(sorted) - i]["index"]
      X_train.drop(drop_col, axis=1, inplace=True)
      X_test.drop(drop_col, axis=1, inplace=True)

      lgb.fit(X_train, y_train)
      y_pred = lgb.predict(X_test)
      mse = mean_squared_error(y_test, y_pred)
      score = np.sqrt(mse)

      attempts[i] = score

      if score < best_score:
        best_score = score
    return pd.DataFrame(attempts.values(), index=attempts.keys(), columns=['Results']).sort_values(by="Results",
                                                                                                   ascending=True).head(10)


def lgbm_optuna_regressor(X, y, n_trials=100, test_size=0.2, random_state=42):
  """
  Optuna ile LGBMRegressor hiperparametre optimizasyonu yapar ve en iyi modeli döndürür.

  Parameters
  ----------
  X : DataFrame/ndarray
      Özellik matrisi
  y : Series/ndarray
      Hedef değişken
  n_trials : int
      Denenecek Optuna deneme sayısı
  test_size : float
      Test seti oranı
  random_state : int
      Rastgelelik kontrolü

  Returns
  -------
  best_model : LGBMRegressor
      En iyi parametrelerle eğitilmiş model
  best_params : dict
      En iyi hiperparametreler
  rmse : float
      Test setinde RMSE değeri
  """

  X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=random_state
  )

  def objective_lgb(trial):
    params = {
      'objective': trial.suggest_categorical('objective', ['root_mean_squared_error']),
      'max_depth': trial.suggest_int('max_depth', 3, 10),
      'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
      'n_estimators': trial.suggest_int('n_estimators', 300, 700),
      'feature_fraction': trial.suggest_float('feature_fraction', 0.1, 1),
      'lambda_l1': trial.suggest_float('lambda_l1', 0, 1),
      'lambda_l2': trial.suggest_float('lambda_l2', 0, 1),
      "random_state": random_state,
      'verbose': -1,
    }

    model_lgb = LGBMRegressor(**params)
    model_lgb.fit(X_train, y_train)
    y_pred = model_lgb.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    return rmse

  # Optuna çalıştır
  study_lgb = optuna.create_study(direction='minimize')
  optuna.logging.set_verbosity(optuna.logging.WARNING)
  study_lgb.optimize(objective_lgb, n_trials=n_trials, show_progress_bar=True)

  # En iyi parametrelerle model kur
  best_params = study_lgb.best_params
  best_model = LGBMRegressor(**best_params)
  best_model.fit(X_train, y_train)

  # Test performansı
  y_pred = best_model.predict(X_test)
  mse = mean_squared_error(y_test, y_pred)
  rmse = np.sqrt(mse)

  print("Best Params:", best_params)
  print("Error (RMSE):", rmse)

  return best_model, best_params, rmse

def main():
    # Veri yükleme
    df = pd.read_csv("FC2425Corr.csv")

    # Veri ön işleme
    X, y = players_data_prep(df)

    # Optuna ile en iyi model ve parametreleri bul
    best_model, best_params, rmse = lgbm_optuna_regressor(
      X, y, n_trials=50, test_size=0.2, random_state=42
    )

    # Sonuçları yazdır
    print("\n--- Final Results ---")
    print(f"Best Parameters: {best_params}")
    print(f"Best RMSE: {rmse:.4f}")

    # Modeli kaydet
    joblib.dump(best_model, "best_lgb_model.pkl")
    print("Model kaydedildi: best_lgb_model.pkl")


if __name__ == "__main__":
  main()

