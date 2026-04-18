from feature_calculator import FeatureCalculator
from typing import NamedTuple, Callable, List, Dict
from collections import defaultdict

class Feature(NamedTuple):
    name: str
    get: Callable
    tags: List[str]


FEATURE_MAP = [
    Feature('morgan_fp', FeatureCalculator.get_morgan_fp, ['fingerprints', '2d_topology']),
    Feature('delta_q', FeatureCalculator.get_delta_Q, ['electronic', 'charge_distribution', '2d_topology']),
    Feature('lambda_max', FeatureCalculator.get_max_eigenvalue, ['spectral', 'laplacian', '2d_topology']),
    Feature('lambda_fiedler', FeatureCalculator.get_fiedler_value, ['spectral', 'laplacian', '2d_topology']),
    Feature('fermi_energy', FeatureCalculator.get_eht_fermi_energy, ['electronic', 'charge_distribution', '3d_quantum']),
    Feature('total_energy', FeatureCalculator.get_eht_total_energy, ['electronic', 'charge_distribution', '3d_quantum']),
    Feature('coulomb_max', FeatureCalculator.get_coulomb_max, ['electrostatic', 'charge_distribution', '3d_physics']),
    Feature('coulomb_trace', FeatureCalculator.get_coulomb_trace, ['electrostatic', 'charge_distribution', '3d_physics']),
    Feature('wiener_index', FeatureCalculator.get_wiener_index, ['topological', 'connectivity', '2d_topology']),
    Feature('sasa', FeatureCalculator.get_sasa, ['structural', 'shape', '3d_physics']),
    Feature('npr1', FeatureCalculator.get_npmi, ['constitutional', 'topological', '3d_shape']),
    Feature('npr2', FeatureCalculator.get_npmi, ['constitutional', 'topological', '3d_shape']),
    Feature('mw', FeatureCalculator.get_mw, ['constitutional', 'physicochemical', 'lipinski']),
    Feature('logp', FeatureCalculator.get_logp, ['constitutional', 'physicochemical', 'lipinski']),
    Feature('hbd', FeatureCalculator.get_hbd, ['constitutional', 'physicochemical', 'lipinski']),
    Feature('hba', FeatureCalculator.get_hba, ['constitutional', 'physicochemical', 'lipinski']),
    Feature('lipinski_violations', FeatureCalculator.get_lipinski_violations, ['constitutional', 'physicochemical', 'lipinski']),
    Feature('tpsa', FeatureCalculator.get_tpsa, ['constitutional', 'physicochemical', 'weber']),
]

class FeatureRegistry:
    """
    Класс для удобного доступа к фичам. 
    Сам собирает все нужные словари при инициализации.
    """
    def __init__(self, features: List[Feature]):
        self.by_name: Dict[str, Callable] = {f.name: f.get for f in features}

        self.by_tag: Dict[str, List[str]] = defaultdict(list)
        for f in features:
            for tag in f.tags:
                self.by_tag[tag].append(f.name)


registry = FeatureRegistry(FEATURE_MAP)

# === ПРИМЕРЫ ИСПОЛЬЗОВАНИЯ ===
# registry.by_name['sasa'] -> вернет функцию FeatureCalculator.get_sasa
# registry.by_tag['charge_distribution'] -> вернет ['delta_q', 'fermi_energy', 'total_energy', 'coulomb_max', 'coulomb_trace']