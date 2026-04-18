from IPython.display import display
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from rdkit import Chem
from rdkit.Chem import Draw, AllChem


class DataAnalytics:
    @staticmethod
    def plot_feature_importance(model, feature_names: list, top_n: int = 15):
        """
        Plots the top N most important features from the trained model.
        Colors indicate the physical nature of the feature (e.g., thermodynamics, electrostatics).

        Args:
            model: Trained machine learning model (must have feature_importances_).
            feature_names (list): List of feature names corresponding to the model's inputs.
            top_n (int): Number of top features to display. Default is 15.
        """
        # 1. If model wrapped in Pipeline, pull the final step
        if hasattr(model, 'named_steps'):
            model_core = list(model.named_steps.values())[-1]
        else:
            model_core = model

        # 2. Check if the model can return feature importances
        if not hasattr(model_core, 'feature_importances_'):
            print(f"Model {model_core.__class__.__name__} doesn't support feature importances.")
            return

        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]

        top_names = [feature_names[i] for i in indices]
        top_importances = importances[indices]

        plt.figure(figsize=(10, 6))
        plt.title(f"Top-{top_n} Features Determining Binding Energy ($\Delta G$)", fontsize=14)
        bars = plt.bar(range(top_n), top_importances, color='steelblue', align="center")

        # Color-code the physical features for better interpretability
        for i, name in enumerate(top_names):
            if name in ['Molecular Weight', 'AlogP']:
                bars[i].set_color('darkred')
            elif name in ['Delta_Q', 'Lambda_Max', 'Lambda_Fiedler']:
                bars[i].set_color('goldenrod')
            elif name in ['Fermi_Energy', 'Total_EHT_Energy', 'Coulomb_Max', 'Coulomb_Trace']:
                bars[i].set_color('purple')

        plt.xticks(range(top_n), top_names, rotation=45, ha='right')
        plt.ylabel("Importance (Error Reduction)")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_shap_analysis(model, X_test, feature_names, sample_size=200):
        """
        Performs SHAP analysis for feature importance. Automatically selects
        the appropriate Explainer type based on the model architecture.
        """
        print("Calculating SHAP values (analyzing physical contributions)...")
        X_sample = X_test[:sample_size]

        try:
            # Attempt fast tree-based algorithm (RF, CatBoost, XGBoost)
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample, check_additivity=False)
        except Exception:
            # Fallback for non-tree models (SVR, Neural Networks)
            print("TreeExplainer failed, falling back to universal Explainer...")
            print(f"TreeExplainer doesn't fit for {model.__class__.__name__}. Try KernelExplainer...")
            # For SVR/Pipelines (so slow)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                X_background = shap.kmeans(X_test, 10) 

                explainer = shap.KernelExplainer(model.predict, X_background)

                shap_values = explainer.shap_values(X_sample)

        plt.figure(figsize=(10, 4))
        plt.title("SHAP Summary Plot:")
        shap.summary_plot(shap_values, X_sample, feature_names=feature_names, max_display=10)


    @staticmethod
    def draw_morgan_bit(df: pd.DataFrame, target_bit: int, smiles_col: str = 'Clean_SMILES', radius: int = 2, nBits: int = 1024):
        """
        Searches the dataset for a molecule containing the specified Morgan fingerprint bit 
        and renders the 2D chemical structure of that specific fragment.
        
        Args:
            df (pd.DataFrame): DataFrame containing the SMILES strings.
            target_bit (int): The integer ID of the bit to visualize.
            smiles_col (str): The name of the column containing SMILES strings.
            radius (int): Morgan fingerprint radius used during generation.
            nBits (int): Morgan fingerprint bit vector length.
        """
        print(f"Scanning the dataset for the presence of bit {target_bit}...")
        example_mol = None
        bit_info = {}

        if smiles_col not in df.columns:
            print(f"Error: Column '{smiles_col}' not found in DataFrame.")
            return

        # Ищем молекулу "на лету", не засоряя оперативную память
        for smiles in df[smiles_col]:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None: 
                continue # Пропускаем битые строки, если вдруг попались
                
            info = {}
            AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=nBits, bitInfo=info)
            
            if target_bit in info:
                example_mol = mol
                bit_info = info
                break

        if example_mol:
            print("Fragment found! Displaying the 2D structure...")
            img = Draw.DrawMorganBit(example_mol, target_bit, bit_info)
            display(img)
        else:
            print(f"Warning: Bit {target_bit} was not found in any molecule of the current dataset.")

    @staticmethod
    def plot_eda(df: pd.DataFrame):
        """
        Performs Exploratory Data Analysis (EDA) by plotting the dataset's
        thermodynamic phase portrait (AlogP distribution) and steric bounds
        (Molecular Weight vs. pChEMBL Value).

        Args:
            df (pd.DataFrame): DataFrame containing 'AlogP', 'Molecular Weight', and 'pChEMBL Value'.
        """
        print("Running Exploratory Data Analysis (EDA)...")
        plt.figure(figsize=(12, 5))

        # Plot 1: Lipophilicity Spectrum
        plt.subplot(1, 2, 1)
        plt.hist(df['AlogP'].dropna(), bins=50, color='teal', edgecolor='black')
        plt.title('Phase Portrait: AlogP Distribution')
        plt.xlabel('AlogP (Calculated Lipophilicity)')
        plt.ylabel('Molecule Count')
        plt.xlim([-5, 10])

        # Plot 2: Energy vs. Mass
        plt.subplot(1, 2, 2)
        plt.scatter(df['Molecular Weight'], df['pChEMBL Value'], alpha=0.3, s=10, color='darkred')
        plt.title('Binding Energy vs. Molecular Weight')
        plt.xlabel('Molecular Weight (Da)')
        plt.ylabel('pChEMBL Value ($\sim \Delta G$)')

        plt.tight_layout()
        plt.show()

    @staticmethod
    def check_correlations(df, feature_cols, plot=True):
        """
        Calculates the Pearson correlation matrix for a given set of continuous features.
        Ignores binary fingerprints (like Morgan) as Pearson correlation is not meaningful for them.
        """
        print("Scanning features for multicollinearity...")
        
        # Keep only the columns that have already been calculated and exist in the dataframe
        available_cols = [col for col in feature_cols if col in df.columns]
        
        if not available_cols:
            print("Warning: Features have not been calculated yet!")
            return None
            
        # Calculate the correlation matrix
        corr_matrix = df[available_cols].corr(method='pearson')
        
        if plot:
            plt.figure(figsize=(10, 8))
            
            # Create a mask for the upper triangle to make the heatmap cleaner
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            
            # Plot the heatmap
            sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", 
                        cmap='coolwarm', vmin=-1, vmax=1, 
                        square=True, linewidths=.5, cbar_kws={"shrink": .7})
            
            plt.title("Correlation Matrix of Physicochemical Features", fontsize=14)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.show()
            
        return corr_matrix

    @staticmethod
    def plot_target_distribution(df: pd.DataFrame, target_col: str = 'Target'):
        """
        Строит гистограмму и график плотности распределения (KDE) для целевой переменной.
        Показывает, с какими значениями pKi модель будет работать чаще всего.
        """
        print(f"Анализ распределения целевой переменной '{target_col}'...")
        
        if target_col not in df.columns:
            print(f"Ошибка: Колонка {target_col} не найдена в датасете.")
            return

        plt.figure(figsize=(8, 5))
        
        # Строим гистограмму с линией тренда (KDE - Kernel Density Estimate)
        sns.histplot(df[target_col], bins=40, kde=True, color='teal', edgecolor='black', alpha=0.6)
        
        # Добавляем вертикальные линии для средних значений
        mean_val = df[target_col].mean()
        median_val = df[target_col].median()
        
        plt.axvline(mean_val, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_val:.2f}')
        plt.axvline(median_val, color='gold', linestyle='solid', linewidth=2, label=f'Median: {median_val:.2f}')
        
        plt.title(f"Distribution of Binding Affinity ($pK_i$)", fontsize=14, fontweight='bold')
        plt.xlabel("Binding Affinity ($pK_i = -\log_{10} K_i$)", fontsize=12)
        plt.ylabel("Number of Molecules", fontsize=12)
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()