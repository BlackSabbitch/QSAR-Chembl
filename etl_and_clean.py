import pandas as pd
from rdkit import Chem
from rdkit.Chem import SaltRemover
from typing import Optional


class DataLoader:
    """
    Handles data ingestion, filtering, and initial preprocessing of chemical datasets.
    Responsible for isolating high-confidence signals and converting raw SMILES strings
    into RDKit molecular objects.
    """

    @staticmethod
    def load_chembl_raw(filepath: str = "molecules_chembl_ki/chembl224.csv", sample_size: Optional[int] = None) -> pd.DataFrame:
        df = pd.read_csv(filepath, sep=';')

        mask_measurement = (df['Standard Type'] == 'Ki') & (df['Standard Relation'] == "'='")
        mask_assay = df['Assay Description'].str.contains('radioligand|binding', case=False, na=False)
        mask_biology = (
            (df['Target Organism'] == 'Homo sapiens') & 
            (df['Assay Type'] == 'B') & 
            (df['Data Validity Comment'].isna())
            )
        df = df[mask_measurement & mask_biology & mask_assay].copy()

        df = df[['Smiles', 'pChEMBL Value']].dropna()
        df = df.rename(columns={'Smiles': 'SMILES', 'pChEMBL Value': 'Target'})
        
        if sample_size and (sample_size < len(df)):
            df = df.sample(n=sample_size, random_state=42).copy()

        return df.reset_index(drop=True)


class DataCleaner:
    """
    Класс для первоначальной загрузки, очистки и фильтрации химических данных.
    Подготавливает чистый DataFrame для передачи в DatasetEnricher.
    """
    def __init__(self, smiles_col: str = 'SMILES', target_col: str = 'Target', std_threshold: float = 0.1):
        self.smiles_col = smiles_col
        self.target_col = target_col
        self.std_threshold = std_threshold
        # Инициализируем стандартный удалитель солей RDKit
        self.remover = SaltRemover.SaltRemover()

    def _strip_salts(self, smiles: str) -> str:
        """
        Преобразует SMILES в объект, удаляет соли/растворители и возвращает чистый SMILES.
        Если молекула не парсится, возвращает None.
        """
        if not isinstance(smiles, str):
            return None
            
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
            
        # Удаляем соли по внутренним паттернам RDKit
        stripped_mol = self.remover.StripMol(mol, dontRemoveEverything=True)
        
        # Альтернативная жесткая защита: если молекула все еще состоит из нескольких частей,
        # берем самый большой фрагмент (по количеству тяжелых атомов).
        frags = Chem.GetMolFrags(stripped_mol, asMols=True)
        if len(frags) > 1:
            stripped_mol = max(frags, key=lambda m: m.GetNumHeavyAtoms())
            
        return Chem.MolToSmiles(stripped_mol)

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        print("Старт химической очистки...")
        df_clean = df.copy()

        # Шаг 1: Удаление солей (оставляем только активный ион)
        print("-> Стандартизация графов и удаление солей...")
        df_clean['Clean_SMILES'] = df_clean[self.smiles_col].progress_apply(self._strip_salts)
        df_clean = df_clean.dropna(subset=['Clean_SMILES'])

        # Шаг 2: Агрегация дубликатов с фильтром по дисперсии
        print("-> Агрегация дубликатов по медиане и фильтрация шума...")
        stats = df_clean.groupby('Clean_SMILES')[self.target_col].agg(['median', 'std', 'count']).reset_index()
        
        # Для молекул, которые встретились ровно 1 раз, std будет NaN. Заполняем нулем.
        stats['std'] = stats['std'].fillna(0)
        
        # Отсекаем молекулы со слишком большим разбросом в измерениях
        filtered_stats = stats[stats['std'] <= self.std_threshold].copy()
        
        dropped = len(stats) - len(filtered_stats)
        if dropped > 0:
            print(f"   [!] Отброшено {dropped} молекул из-за противоречивых данных (std > {self.std_threshold}).")

        # Переименовываем колонку 'median' обратно в 'Target'
        final_df = filtered_stats.rename(columns={'median': self.target_col})
        
        print(f"Очистка завершена! Уникальных и надежных молекул: {len(final_df)}")
        # Возвращаем только чистый SMILES и целевую переменную
        return final_df[['Clean_SMILES', self.target_col]]
