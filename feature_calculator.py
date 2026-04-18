import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdmolops, rdFingerprintGenerator, rdEHTTools, rdMolDescriptors, rdFreeSASA, Descriptors
from scipy.linalg import eigvals
from typing import List


class FeatureCalculator:
    def __init__(self, mol_smiles: str):
        if not isinstance(mol_smiles, str):
            raise ValueError("Input SMILES must be a string")

        self.mol_obj = Chem.MolFromSmiles(mol_smiles)
        if self.mol_obj is None:
            raise ValueError(f"Invalid SMILES string: {mol_smiles}")

        self._cache = {}
        self._mol_3d = {'build_attempts': False, 'mol': None}
        self._laplasian = {'build_attempts': False, 'eigenvalues': None}
        self._eht = {'build_attempts': False, 'eht': None}
        self._coulomb = {'build_attempts': False, 'eigenvalues': None}

    def _calc_3d_mol(self):
        if self._mol_3d['build_attempts']:
            return
        self._mol_3d['build_attempts'] = True

        try:
            mol_3d = Chem.AddHs(self.mol_obj)
            params = AllChem.ETKDGv3()
            params.randomSeed = 42
            params.maxIterations = 200

            if AllChem.EmbedMolecule(mol_3d, params) != -1:
                AllChem.MMFFOptimizeMolecule(mol_3d)
                self._mol_3d['mol'] = mol_3d
        except Exception:
            pass

    def _calc_laplasian_eigenvalues(self):
        if self._laplasian['build_attempts']:
            return
        self._laplasian['build_attempts'] = True

        try:
            A = Chem.GetAdjacencyMatrix(self.mol_obj)
            L = np.diag(A.sum(axis=1)) - A
            eigenvalues = np.linalg.eigvalsh(L)
            self._laplasian['eigenvalues'] = eigenvalues
        except Exception:
            pass

    def _calc_eht(self):
        if self._eht['build_attempts']:
            return
        self._calc_3d_mol()
        if self._mol_3d['mol'] is None:
            self._eht['build_attempts'] = True
            return
            
        self._eht['build_attempts'] = True

        try:
            passed, eht_res = rdEHTTools.RunMol(self._mol_3d['mol'])
            fermi_energy = eht_res.fermiEnergy if passed else np.nan
            total_energy = eht_res.totalEnergy if passed else np.nan
            # fermi_energy = eht_res.GetFermiEnergy() if passed else np.nan
            # total_energy = eht_res.GetTotalEnergy() if passed else np.nan
            self._eht['eht'] = [fermi_energy, total_energy]
        except Exception:
            pass

    def _calc_coulomb(self):
        if self._coulomb['build_attempts']:
            return
        self._calc_3d_mol()
        if self._mol_3d['mol'] is None:
            self._coulomb['build_attempts'] = True
            return
        self._coulomb['build_attempts'] = True            
        
        try:
            dist_matrix = rdmolops.Get3DDistanceMatrix(self._mol_3d['mol'])
            np.fill_diagonal(dist_matrix, 100)
            if np.min(dist_matrix) < 0.4:
                return

            num_atoms = self._mol_3d['mol'].GetNumAtoms()
            C = np.zeros((num_atoms, num_atoms))
            for i in range(num_atoms):
                Zi = self._mol_3d['mol'].GetAtomWithIdx(i).GetAtomicNum()
                for j in range(i, num_atoms):
                    Zj = self._mol_3d['mol'].GetAtomWithIdx(j).GetAtomicNum()
                    if i == j:
                        C[i][i] = 0.5 * (Zi ** 2.4)
                    else:
                        coulomb_val = (Zi * Zj) / dist_matrix[i][j]
                        C[i][j] = coulomb_val
                        C[j][i] = coulomb_val # Enforce symmetry

            self._coulomb['eigenvalues'] = np.sort(np.real(eigvals(C)))
        except Exception:
            pass

    def get_morgan_fp(self) -> np.ndarray:
        if 'morgan_fp' not in self._cache:
            mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
            self._cache['morgan_fp'] = mfpgen.GetFingerprintAsNumPy(self.mol_obj) if self.mol_obj is not None else [np.nan]*1024
        return self._cache['morgan_fp']

    def get_delta_Q(self) ->  float:
        if 'delta_q' not in self._cache:
            try:
                AllChem.ComputeGasteigerCharges(self.mol_obj)
                charges = [float(atom.GetProp('_GasteigerCharge')) for atom in self.mol_obj.GetAtoms()]
                charges = [c for c in charges if not np.isnan(c) and not np.isinf(c)]   
                self._cache['delta_q'] = max(charges) - min(charges) if charges else np.nan
            except Exception:
                self._cache['delta_q'] = np.nan
        return self._cache['delta_q']

    def get_fiedler_value(self) -> float:
        if 'fiedler_value' not in self._cache:
            self._calc_laplasian_eigenvalues()
            if self._laplasian['eigenvalues'] is not None and len(self._laplasian['eigenvalues']) > 1:
                self._cache['fiedler_value'] = self._laplasian['eigenvalues'][1]
            else:
                self._cache['fiedler_value'] = np.nan
        return self._cache['fiedler_value']

    def get_max_eigenvalue(self) -> float:
        if 'max_eigenvalue' not in self._cache:
            self._calc_laplasian_eigenvalues()
            if self._laplasian['eigenvalues'] is not None and len(self._laplasian['eigenvalues']) > 0:
                self._cache['max_eigenvalue'] = self._laplasian['eigenvalues'][-1]
            else:
                self._cache['max_eigenvalue'] = np.nan
        return self._cache['max_eigenvalue']

    def get_eht_fermi_energy(self) -> float:
        if 'fermi_energy' not in self._cache:
             self._calc_eht()             
             self._cache['fermi_energy'] = self._eht['eht'][0] if self._eht['eht'] is not None else np.nan
        return self._cache['fermi_energy']

    def get_eht_total_energy(self) -> float:
        if 'total_energy' not in self._cache:
             self._calc_eht()
             self._cache['total_energy'] = self._eht['eht'][1] if self._eht['eht'] is not None else np.nan
        return self._cache['total_energy']

    def get_coulomb_max(self) -> float:
        if 'coulomb_max' not in self._cache:
            self._calc_coulomb()
            self._cache['coulomb_max'] = self._coulomb['eigenvalues'][-1] if self._coulomb['eigenvalues'] is not None else np.nan
        return self._cache['coulomb_max']

    def get_coulomb_trace(self) -> float:
        if 'coulomb_trace' not in self._cache:
            self._calc_coulomb()
            self._cache['coulomb_trace'] = self._coulomb['eigenvalues'].sum() if self._coulomb['eigenvalues'] is not None else np.nan
        return self._cache['coulomb_trace']

    def get_wiener_index(self) -> float:
        if 'wiener_index' not in self._cache:
            try:
                dist_matrix_2d = Chem.GetDistanceMatrix(self.mol_obj)
                wiener_index = 0.5 * np.sum(dist_matrix_2d)
            except: 
                wiener_index = np.nan
            self._cache['wiener_index'] = wiener_index
        return self._cache['wiener_index']

    def get_sasa(self) -> float:
        if 'sasa' not in self._cache:
             self._calc_3d_mol()
             if self._mol_3d['mol'] is not None:
                 try:
                     radii = rdFreeSASA.classifyAtoms(self._mol_3d['mol'])
                     sasa = rdFreeSASA.CalcSASA(self._mol_3d['mol'], radii)
                     self._cache['sasa'] = sasa
                 except Exception:
                     self._cache['sasa'] = np.nan
             else:
                 self._cache['sasa'] = np.nan

        return self._cache['sasa']

    def get_npmi(self) -> List[float]:
        if 'npmi' not in self._cache:
            self._calc_3d_mol()
            if self._mol_3d['mol'] is not None:
                try:
                    pmi3 = rdMolDescriptors.CalcPMI3(self._mol_3d['mol'])

                    if pmi3 > 1e-4:
                        npr1 = rdMolDescriptors.CalcPMI1(self._mol_3d['mol']) / pmi3
                        npr2 = rdMolDescriptors.CalcPMI2(self._mol_3d['mol']) / pmi3
                    else:
                        npr1, npr2 = np.nan, np.nan
                except:
                    npr1, npr2 = np.nan, np.nan
            else:
                npr1, npr2 = np.nan, np.nan

            self._cache['npmi'] = [npr1, npr2]

        return self._cache['npmi']

    def get_mw(self) -> float:
        if 'mw' not in self._cache:
            self._cache['mw'] = Descriptors.MolWt(self.mol_obj)
        return self._cache['mw']

    def get_logp(self) -> float:
        if 'logp' not in self._cache:
            self._cache['logp'] = Descriptors.MolLogP(self.mol_obj)
        return self._cache['logp']

    def get_hbd(self) -> int:
        if 'hbd' not in self._cache:
            self._cache['hbd'] = rdMolDescriptors.CalcNumHBD(self.mol_obj)
        return self._cache['hbd']

    def get_hba(self) -> int:
        if 'hba' not in self._cache:
            self._cache['hba'] = rdMolDescriptors.CalcNumHBA(self.mol_obj)
        return self._cache['hba']

    def get_lipinski_violations(self) -> int:
        if 'lipinski_violations' not in self._cache:
            violations = sum([
                self.get_mw() > 500,
                self.get_logp() > 5,
                self.get_hbd() > 5,
                self.get_hba() > 10
            ])
            self._cache['lipinski_violations'] = int(violations)
        return self._cache['lipinski_violations']

    def get_tpsa(self) -> float:
        if 'tpsa' not in self._cache:
            self._cache['tpsa'] = Descriptors.TPSA(self.mol_obj)
        return self._cache['tpsa']
