# experiment_runner.py
import numpy as np
import time
import joblib
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class ExperimentRunner:
    """
    Knows nothing about chemistry, takes prepared tensors.
    """
    def __init__(self, model):
        self.model = model
        self.metrics = {}
        self.feature_names = []

    def run(self, X: np.ndarray, y: np.ndarray, feature_names: list, n_splits: int = 5, test_size: float = 0.2, random_state: int = 42, verbose: bool = True):
        start_time = time.time()
        self.feature_names = feature_names
        r2_scores, rmse_scores, mae_scores = [], [], []

        # --- Mode 1: FAST (w/out K-Fold) ---
        if n_splits in [0, 1]:
            if verbose: print(f"Launch simple Train/Test split (Test size = {test_size}) for {self.model.__class__.__name__}...")
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
            
            self.model.fit(X_train, y_train)
            y_pred = self.model.predict(X_test)
            
            r2_scores.append(r2_score(y_test, y_pred))
            rmse_scores.append(np.sqrt(mean_squared_error(y_test, y_pred)))
            mae_scores.append(mean_absolute_error(y_test, y_pred))

        else:
            if verbose: print(f"Launch {n_splits}-Fold CV for {self.model.__class__.__name__}...")
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            for train_index, test_index in kf.split(X):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]

                self.model.fit(X_train, y_train)
                y_pred = self.model.predict(X_test)

                r2_scores.append(r2_score(y_test, y_pred))
                rmse_scores.append(np.sqrt(mean_squared_error(y_test, y_pred)))
                mae_scores.append(mean_absolute_error(y_test, y_pred))

        self.metrics = {
            'R2_mean': np.mean(r2_scores),
            'R2_std': np.std(r2_scores),
            'RMSE_mean': np.mean(rmse_scores),
            'RMSE_std': np.std(rmse_scores),
            'MAE_mean': np.mean(mae_scores),
            'MAE_std': np.std(mae_scores),
            'Time_seconds': time.time() - start_time
        }
        
        # Final learning on all dataset
        self.model.fit(X, y)
        if verbose: print("Experiment finished.\n")

    def get_metrics(self) -> dict: return self.metrics
    def get_model(self): return self.model

    def save_model(self, filename: str = "best_model.joblib"):
        dump_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'metrics': self.metrics
        }
        joblib.dump(dump_data, filename)
        print(f"Model saved in: {filename}")
