import pandas as pd
import numpy as np
from tqdm import tqdm
from feature_calculator import FeatureCalculator
from feature_map import registry 
from sklearn.feature_selection import mutual_info_regression
from scipy.stats import rankdata


tqdm.pandas()

class FeatureBuilder:
    """
    Main orchestration. Takes DataFrame, parses request by tags,
    lazy calculates features row-by-row nd collects the results.
    """
    def __init__(self, feature_registry):
        self.registry = feature_registry

    def _parse_requests(self, requests: list) -> dict:
        """
        User request (tag or list of features)->
        flat dictionary {feature_name: get_function}.
        """
        funcs_to_run = {}
        for req in requests:
            # If exact feature requested (f.ex., 'sasa')
            if req in self.registry.by_name:
                funcs_to_run[req] = self.registry.by_name[req]
            # If feauture group or tag requested (f.ex., '3d_physics')
            elif req in self.registry.by_tag:
                for fname in self.registry.by_tag[req]:
                    funcs_to_run[fname] = self.registry.by_name[fname]
            else:
                print(f"[WARNING] Ignore unknown tag or feature: {req}")
        
        return funcs_to_run

    def _process_single_smiles(self, smiles: str, funcs_to_run: dict) -> pd.Series:
        """
        Processes one molecule.
        """
        result = {}
        try:
            # Only one initialization for calculator!
            calc = FeatureCalculator(smiles)

            for name, func in funcs_to_run.items():
                val = func(calc)

                if isinstance(val, list) and len(val) == 2:
                    if name == 'npr1':
                        result[name] = val[0]
                    elif name == 'npr2':
                        result[name] = val[1]
                    else:
                        result[name] = val
                else:
                    result[name] = val
                    
        except Exception:
            # Defense from the RDKit failures (bad rows etc.)
            for name in funcs_to_run.keys():
                # Morgan FP: NaN-vector, scalars: NaN
                if name == 'morgan_fp':
                    result[name] = np.full(1024, np.nan)
                else:
                    result[name] = np.nan
                    
        return pd.Series(result)

    def enrich(self, df: pd.DataFrame, smiles_column: str, requested_features: list) -> pd.DataFrame:
        """
        Enriches DataFrame with requested features.
        """
        funcs_to_run = self._parse_requests(requested_features)
        
        if not funcs_to_run:
            print("There are no valid features for calculaion. return initial DataFrame.")
            return df
            
        print(f"Calculation started for {len(funcs_to_run)} features: {list(funcs_to_run.keys())}")
        
        # progress_apply = apply + progress bar
        enriched_cols = df[smiles_column].progress_apply(
            lambda smiles: self._process_single_smiles(smiles, funcs_to_run)
        )
        
        return pd.concat([df, enriched_cols], axis=1)

    def select_orthogonal_basis(self, df: pd.DataFrame, feature_columns: list, target_col: str = 'Target', threshold: float = 0.85, target_metric: str = 'combined') -> list:
        """
        Collinearity filtering.
        target_metric: 'pearson', 'mutual_info' или 'combined' (hybrid).
        """
        scalar_features = [f for f in feature_columns if f in df.columns and f != 'morgan_fp']
        if not scalar_features: return feature_columns

        print(f"Optimization of the basis (metrics: {target_metric})...")
        corr_matrix = df[scalar_features].corr().abs()
        
        # 1. Pearson calculation
        if target_metric in ['pearson', 'combined']:
            pearson_scores = df[scalar_features + [target_col]].corr().abs()[target_col].drop(target_col)
            
        # 2. Mutual Information calculation (MI doesn't like NaN, so we do temporary fillna)
        if target_metric in ['mutual_info', 'combined']:
            X_mi = df[scalar_features].fillna(0)
            y_mi = df[target_col].fillna(0)
            mi_scores = pd.Series(
                mutual_info_regression(X_mi, y_mi, random_state=42), 
                index=scalar_features
            )
            
        # 3. Collect final "useful rating"
        if target_metric == 'combined':
            # To rank (from 0 to 1, where 1 - the most useful feature)
            p_ranks = rankdata(pearson_scores) / len(pearson_scores)
            m_ranks = rankdata(mi_scores) / len(mi_scores)
            target_scores = pd.Series(p_ranks + m_ranks, index=scalar_features)
        elif target_metric == 'pearson':
            target_scores = pearson_scores
        else:
            target_scores = mi_scores

        to_drop = set()
        features = corr_matrix.columns

        # Conflicts solving
        for i in range(len(features)):
            for j in range(i + 1, len(features)):
                feat_i = features[i]
                feat_j = features[j]
                
                if corr_matrix.loc[feat_i, feat_j] > threshold:
                    if feat_i in to_drop or feat_j in to_drop:
                        continue

                    if target_scores[feat_i] > target_scores[feat_j]:
                        to_drop.add(feat_j)
                        # print(f"  [Conflict] {feat_i} vs {feat_j} -> Leave {feat_i}")
                    else:
                        to_drop.add(feat_i)
                        # print(f"  [Conflict] {feat_i} vs {feat_j} -> Leave {feat_j}")

        basis = [f for f in feature_columns if f not in to_drop]
        print(f"Colinear featues dropped: {len(to_drop)}. Resulted basis: {len(basis)} features.")
        return basis
    
    def get_tensors(self, df: pd.DataFrame, feature_columns: list, target_column: str):
        """
        DataFrame -> tensors X & y, ready for learning.
        """
        print("Collecting of final tensors...")
        
        # Delete rows, if there is no y (target)
        df_clean = df.dropna(subset=[target_column]).copy()
        
        y = df_clean[target_column].values
        X_list = []
        final_feature_names = []

        for feat in feature_columns:
            if feat not in df_clean.columns:
                print(f"[WARNING] Feature {feat} Is absent in DataFrame, skip.")
                continue
                
            if feat == 'morgan_fp':
                # vertical stacking. list of vectors -> matrix (N, 1024)
                fp_matrix = np.vstack(df_clean[feat].values)
                X_list.append(fp_matrix)
                # generate names for bits
                final_feature_names.extend([f'morgan_bit_{i}' for i in range(fp_matrix.shape[1])])
            else:
                # scalar featues: vector (N,) -> column-matrix (N, 1)
                scalar_vector = df_clean[feat].values.reshape(-1, 1)
                X_list.append(scalar_vector)
                final_feature_names.append(feat)

        # Horizontal stacking. ll blocks (Morgan + scalars)
        # Resulted matrix X (N, M)
        X = np.hstack(X_list)
        
        # Imputation
        # All NaN (RDKit failures) -> 0.0
        # can be also mean or median, not only zeros.
        num_nans = np.isnan(X).sum()
        if num_nans > 0:
            print(f"Found {num_nans} missed values (NaN). Replaced with 0.0.")
            X = np.nan_to_num(X, nan=0.0)

        print(f"Matrix X collected. Dimensions: {X.shape}")
        return X, y, final_feature_names
