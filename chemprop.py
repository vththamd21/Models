"""
PerceiverCPI - Code Refactorisé
Système amélioré pour la prédiction d'interactions composé-protéine
"""

from typing import Callable, List, Union, Tuple, Optional, Dict, Any
from abc import ABC, abstractmethod
from dataclasses import dataclass
from itertools import zip_longest
import numpy as np
import torch
import torch.nn as nn
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =====================================================================
# 1. SYSTÈME DE GÉNÉRATION DE FEATURES REFACTORISÉ
# =====================================================================

@dataclass
class FeatureConfig:
    """Configuration pour la génération de features moléculaires"""
    morgan_radius: int = 2
    morgan_num_bits: int = 2048
    use_rdkit_2d: bool = False
    normalize_features: bool = False

class FeatureGenerator(ABC):
    """Classe de base abstraite pour les générateurs de features"""
    
    @abstractmethod
    def generate(self, mol: Union[str, Chem.Mol]) -> np.ndarray:
        """Génère des features pour une molécule"""
        pass
    
    @abstractmethod
    def get_feature_dim(self) -> int:
        """Retourne la dimension des features générées"""
        pass

class MorganFingerprintGenerator(FeatureGenerator):
    """Générateur d'empreintes Morgan optimisé"""
    
    def __init__(self, config: FeatureConfig):
        self.radius = config.morgan_radius
        self.num_bits = config.morgan_num_bits
        self.binary = True
    
    def generate(self, mol: Union[str, Chem.Mol]) -> np.ndarray:
        """Génère une empreinte Morgan binaire"""
        try:
            if isinstance(mol, str):
                mol = Chem.MolFromSmiles(mol)
                if mol is None:
                    raise ValueError(f"Impossible de parser le SMILES: {mol}")
            
            # Génération de l'empreinte
            if self.binary:
                fp = AllChem.GetMorganFingerprintAsBitVect(
                    mol, self.radius, nBits=self.num_bits
                )
            else:
                fp = AllChem.GetHashedMorganFingerprint(
                    mol, self.radius, nBits=self.num_bits
                )
            
            # Conversion en array numpy
            features = np.zeros(self.num_bits)
            DataStructs.ConvertToNumpyArray(fp, features)
            
            return features
            
        except Exception as e:
            logger.error(f"Erreur lors de la génération Morgan: {e}")
            return np.zeros(self.num_bits)
    
    def get_feature_dim(self) -> int:
        return self.num_bits

class MorganCountGenerator(MorganFingerprintGenerator):
    """Générateur d'empreintes Morgan basé sur les comptages"""
    
    def __init__(self, config: FeatureConfig):
        super().__init__(config)
        self.binary = False

class FeatureGeneratorFactory:
    """Factory pour créer des générateurs de features"""
    
    _generators = {
        'morgan': MorganFingerprintGenerator,
        'morgan_count': MorganCountGenerator,
    }
    
    @classmethod
    def create(cls, generator_type: str, config: FeatureConfig) -> FeatureGenerator:
        """Crée un générateur de features"""
        if generator_type not in cls._generators:
            available = list(cls._generators.keys())
            raise ValueError(f"Type {generator_type} non supporté. Disponibles: {available}")
        
        return cls._generators[generator_type](config)
    
    @classmethod
    def register(cls, name: str, generator_class: type):
        """Enregistre un nouveau type de générateur"""
        cls._generators[name] = generator_class

# =====================================================================
# 2. SYSTÈME DE FEATURISATION MOLÉCULAIRE REFACTORISÉ
# =====================================================================

@dataclass
class AtomFeatureConfig:
    """Configuration pour les features atomiques"""
    max_atomic_num: int = 100
    degree_options: List[int] = None
    formal_charge_options: List[int] = None
    chiral_tag_options: List[int] = None
    num_hs_options: List[int] = None
    hybridization_options: List = None
    
    def __post_init__(self):
        if self.degree_options is None:
            self.degree_options = [0, 1, 2, 3, 4, 5]
        if self.formal_charge_options is None:
            self.formal_charge_options = [-2, -1, 0, 1, 2]
        if self.chiral_tag_options is None:
            self.chiral_tag_options = [0, 1, 2, 3]
        if self.num_hs_options is None:
            self.num_hs_options = [0, 1, 2, 3, 4]
        if self.hybridization_options is None:
            self.hybridization_options = [
                Chem.rdchem.HybridizationType.SP,
                Chem.rdchem.HybridizationType.SP2,
                Chem.rdchem.HybridizationType.SP3,
                Chem.rdchem.HybridizationType.SP3D,
                Chem.rdchem.HybridizationType.SP3D2
            ]

class FeaturizerUtils:
    """Utilitaires pour la featurisation"""
    
    @staticmethod
    def one_hot_encode(value: Any, choices: List[Any], include_unknown: bool = True) -> List[int]:
        """Encode une valeur en one-hot avec gestion des valeurs inconnues"""
        encoding_length = len(choices) + (1 if include_unknown else 0)
        encoding = [0] * encoding_length
        
        try:
            idx = choices.index(value)
            encoding[idx] = 1
        except ValueError:
            if include_unknown:
                encoding[-1] = 1  # Valeur inconnue
            else:
                logger.warning(f"Valeur inconnue {value} ignorée")
        
        return encoding

class AtomFeaturizer:
    """Featurizer pour les atomes optimisé"""
    
    def __init__(self, config: AtomFeatureConfig = None):
        self.config = config or AtomFeatureConfig()
        self._feature_dim = self._calculate_feature_dim()
    
    def _calculate_feature_dim(self) -> int:
        """Calcule la dimension des features atomiques"""
        dim = 0
        # Atomic number
        dim += self.config.max_atomic_num + 1
        # Degree
        dim += len(self.config.degree_options) + 1
        # Formal charge
        dim += len(self.config.formal_charge_options) + 1
        # Chiral tag
        dim += len(self.config.chiral_tag_options) + 1
        # Num Hs
        dim += len(self.config.num_hs_options) + 1
        # Hybridization
        dim += len(self.config.hybridization_options) + 1
        # IsAromatic + Mass
        dim += 2
        return dim
    
    def featurize(self, atom: Optional[Chem.rdchem.Atom]) -> List[float]:
        """Génère les features pour un atome"""
        if atom is None:
            return [0.0] * self._feature_dim
        
        features = []
        
        # Atomic number (0-based indexing)
        atomic_nums = list(range(self.config.max_atomic_num))
        features.extend(
            FeaturizerUtils.one_hot_encode(atom.GetAtomicNum() - 1, atomic_nums)
        )
        
        # Degree
        features.extend(
            FeaturizerUtils.one_hot_encode(atom.GetTotalDegree(), self.config.degree_options)
        )
        
        # Formal charge
        features.extend(
            FeaturizerUtils.one_hot_encode(atom.GetFormalCharge(), self.config.formal_charge_options)
        )
        
        # Chiral tag
        features.extend(
            FeaturizerUtils.one_hot_encode(int(atom.GetChiralTag()), self.config.chiral_tag_options)
        )
        
        # Number of Hs
        features.extend(
            FeaturizerUtils.one_hot_encode(int(atom.GetTotalNumHs()), self.config.num_hs_options)
        )
        
        # Hybridization
        features.extend(
            FeaturizerUtils.one_hot_encode(int(atom.GetHybridization()), self.config.hybridization_options)
        )
        
        # Aromaticity
        features.append(1.0 if atom.GetIsAromatic() else 0.0)
        
        # Mass (normalized)
        features.append(atom.GetMass() * 0.01)
        
        return features
    
    def get_feature_dim(self) -> int:
        return self._feature_dim

class BondFeaturizer:
    """Featurizer pour les liaisons optimisé"""
    
    def __init__(self):
        self._feature_dim = 14  # Dimension fixe pour les liaisons
        self.stereo_options = list(range(6))
    
    def featurize(self, bond: Optional[Chem.rdchem.Bond]) -> List[float]:
        """Génère les features pour une liaison"""
        if bond is None:
            # Liaison virtuelle (padding)
            return [1.0] + [0.0] * (self._feature_dim - 1)
        
        bond_type = bond.GetBondType()
        
        features = [
            0.0,  # Liaison réelle (pas de padding)
            1.0 if bond_type == Chem.rdchem.BondType.SINGLE else 0.0,
            1.0 if bond_type == Chem.rdchem.BondType.DOUBLE else 0.0,
            1.0 if bond_type == Chem.rdchem.BondType.TRIPLE else 0.0,
            1.0 if bond_type == Chem.rdchem.BondType.AROMATIC else 0.0,
            1.0 if bond.GetIsConjugated() else 0.0,
            1.0 if bond.IsInRing() else 0.0
        ]
        
        # Stereochemistry
        stereo_features = FeaturizerUtils.one_hot_encode(
            int(bond.GetStereo()), self.stereo_options
        )
        features.extend(stereo_features)
        
        return features
    
    def get_feature_dim(self) -> int:
        return self._feature_dim

# =====================================================================
# 3. GRAPHE MOLÉCULAIRE REFACTORISÉ
# =====================================================================

class MolecularGraph:
    """Représentation de graphe moléculaire optimisée"""
    
    def __init__(self, 
                 mol: Union[str, Chem.Mol],
                 atom_featurizer: AtomFeaturizer = None,
                 bond_featurizer: BondFeaturizer = None,
                 extra_atom_features: Optional[np.ndarray] = None,
                 extra_bond_features: Optional[np.ndarray] = None):
        
        self.atom_featurizer = atom_featurizer or AtomFeaturizer()
        self.bond_featurizer = bond_featurizer or BondFeaturizer()
        
        # Conversion en molécule RDKit si nécessaire
        if isinstance(mol, str):
            mol = Chem.MolFromSmiles(mol)
            if mol is None:
                raise ValueError(f"Impossible de parser le SMILES: {mol}")
        
        self.mol = mol
        self._build_graph(extra_atom_features, extra_bond_features)
    
    def _build_graph(self, 
                     extra_atom_features: Optional[np.ndarray] = None,
                     extra_bond_features: Optional[np.ndarray] = None):
        """Construit la représentation de graphe"""
        
        # Initialisation
        self.n_atoms = self.mol.GetNumAtoms()
        self.n_bonds = 0
        
        # Features atomiques
        self.atom_features = []
        for atom in self.mol.GetAtoms():
            features = self.atom_featurizer.featurize(atom)
            self.atom_features.append(features)
        
        # Ajout des features atomiques supplémentaires
        if extra_atom_features is not None:
            if len(extra_atom_features) != self.n_atoms:
                raise ValueError(f"Nombre d'atomes {self.n_atoms} != features extra {len(extra_atom_features)}")
            
            for i, extra in enumerate(extra_atom_features):
                self.atom_features[i].extend(extra.tolist())
        
        # Structures de graphe
        self.bond_features = []
        self.atom_to_bonds = [[] for _ in range(self.n_atoms)]
        self.bond_to_atom = []
        self.bond_to_reverse = []
        
        # Construction des liaisons
        for a1 in range(self.n_atoms):
            for a2 in range(a1 + 1, self.n_atoms):
                bond = self.mol.GetBondBetweenAtoms(a1, a2)
                
                if bond is None:
                    continue
                
                # Features de liaison
                bond_feat = self.bond_featurizer.featurize(bond)
                
                # Ajout des features supplémentaires si disponibles
                if extra_bond_features is not None:
                    bond_idx = bond.GetIdx()
                    if bond_idx < len(extra_bond_features):
                        bond_feat.extend(extra_bond_features[bond_idx].tolist())
                
                # Liaison directe (a1 -> a2)
                self.bond_features.append(self.atom_features[a1] + bond_feat)
                # Liaison inverse (a2 -> a1)
                self.bond_features.append(self.atom_features[a2] + bond_feat)
                
                # Mise à jour des index
                b1, b2 = self.n_bonds, self.n_bonds + 1
                
                self.atom_to_bonds[a2].append(b1)  # a1 -> a2
                self.bond_to_atom.append(a1)
                
                self.atom_to_bonds[a1].append(b2)  # a2 -> a1
                self.bond_to_atom.append(a2)
                
                self.bond_to_reverse.extend([b2, b1])
                self.n_bonds += 2
    
    def get_features(self) -> Tuple[List[List[float]], List[List[float]]]:
        """Retourne les features atomiques et de liaisons"""
        return self.atom_features, self.bond_features
    
    def get_graph_structure(self) -> Tuple[List[List[int]], List[int], List[int]]:
        """Retourne la structure du graphe"""
        return self.atom_to_bonds, self.bond_to_atom, self.bond_to_reverse

class BatchMolecularGraph:
    """Batch de graphes moléculaires optimisé"""
    
    def __init__(self, molecular_graphs: List[MolecularGraph]):
        if not molecular_graphs:
            raise ValueError("Liste de graphes vide")
        
        self.graphs = molecular_graphs
        self._build_batch()
    
    def _build_batch(self):
        """Construit le batch de graphes"""
        
        # Calcul des dimensions
        first_graph = self.graphs[0]
        self.atom_feature_dim = len(first_graph.atom_features[0]) if first_graph.atom_features else 0
        self.bond_feature_dim = len(first_graph.bond_features[0]) if first_graph.bond_features else 0
        
        # Initialisation avec padding (index 0)
        self.n_atoms = 1
        self.n_bonds = 1
        
        all_atom_features = [[0.0] * self.atom_feature_dim]
        all_bond_features = [[0.0] * self.bond_feature_dim]
        all_atom_to_bonds = [[]]
        all_bond_to_atom = [0]
        all_bond_to_reverse = [0]
        
        self.atom_scopes = []
        self.bond_scopes = []
        
        # Assemblage des graphes
        for graph in self.graphs:
            atom_features, bond_features = graph.get_features()
            atom_to_bonds, bond_to_atom, bond_to_reverse = graph.get_graph_structure()
            
            all_atom_features.extend(atom_features)
            all_bond_features.extend(bond_features)
            
            # Mise à jour des indices
            for bonds in atom_to_bonds:
                all_atom_to_bonds.append([b + self.n_bonds for b in bonds])
            
            for atom_idx in bond_to_atom:
                all_bond_to_atom.append(atom_idx + self.n_atoms)
            
            for bond_idx in bond_to_reverse:
                all_bond_to_reverse.append(bond_idx + self.n_bonds)
            
            # Sauvegarde des portées
            self.atom_scopes.append((self.n_atoms, graph.n_atoms))
            self.bond_scopes.append((self.n_bonds, graph.n_bonds))
            
            self.n_atoms += graph.n_atoms
            self.n_bonds += graph.n_bonds
        
        # Conversion en tenseurs PyTorch
        self.atom_features = torch.FloatTensor(all_atom_features)
        self.bond_features = torch.FloatTensor(all_bond_features)
        self.bond_to_atom = torch.LongTensor(all_bond_to_atom)
        self.bond_to_reverse = torch.LongTensor(all_bond_to_reverse)
        
        # Padding pour atom_to_bonds
        max_bonds = max(len(bonds) for bonds in all_atom_to_bonds)
        padded_atom_to_bonds = []
        for bonds in all_atom_to_bonds:
            padded = bonds + [0] * (max_bonds - len(bonds))
            padded_atom_to_bonds.append(padded)
        
        self.atom_to_bonds = torch.LongTensor(padded_atom_to_bonds)
    
    def get_components(self) -> Tuple[torch.Tensor, ...]:
        """Retourne tous les composants du batch"""
        return (
            self.atom_features,
            self.bond_features, 
            self.atom_to_bonds,
            self.bond_to_atom,
            self.bond_to_reverse,
            self.atom_scopes,
            self.bond_scopes
        )

# =====================================================================
# 4. FONCTIONS UTILITAIRES REFACTORISÉES
# =====================================================================

class MolecularDataProcessor:
    """Processeur de données moléculaires"""
    
    @staticmethod
    def smiles_to_graph_batch(smiles_list: List[str],
                             atom_featurizer: AtomFeaturizer = None,
                             bond_featurizer: BondFeaturizer = None,
                             extra_atom_features: List[np.ndarray] = None,
                             extra_bond_features: List[np.ndarray] = None) -> BatchMolecularGraph:
        """Convertit une liste de SMILES en batch de graphes"""
        
        if extra_atom_features is None:
            extra_atom_features = [None] * len(smiles_list)
        
        if extra_bond_features is None:
            extra_bond_features = [None] * len(smiles_list)
        
        graphs = []
        for smiles, atom_extra, bond_extra in zip_longest(
            smiles_list, extra_atom_features, extra_bond_features
        ):
            try:
                graph = MolecularGraph(
                    smiles,
                    atom_featurizer=atom_featurizer,
                    bond_featurizer=bond_featurizer,
                    extra_atom_features=atom_extra,
                    extra_bond_features=bond_extra
                )
                graphs.append(graph)
            except Exception as e:
                logger.error(f"Erreur processing {smiles}: {e}")
                # Ajouter un graphe vide ou ignorer
                continue
        
        return BatchMolecularGraph(graphs)

# =====================================================================
# 5. EXEMPLE D'UTILISATION
# =====================================================================

def example_usage():
    """Exemple d'utilisation du code refactorisé"""
    
    # Configuration
    feature_config = FeatureConfig(
        morgan_radius=3,
        morgan_num_bits=1024
    )
    
    atom_config = AtomFeatureConfig(
        max_atomic_num=120
    )
    
    # Générateur de features Morgan
    morgan_gen = FeatureGeneratorFactory.create('morgan', feature_config)
    
    # Featurizers
    atom_featurizer = AtomFeaturizer(atom_config)
    bond_featurizer = BondFeaturizer()
    
    # Données d'exemple
    smiles_list = [
        'CCO',  # Éthanol
        'CC(=O)O',  # Acide acétique
        'c1ccccc1'  # Benzène
    ]
    
    # Génération des features Morgan
    morgan_features = []
    for smiles in smiles_list:
        features = morgan_gen.generate(smiles)
        morgan_features.append(features)
        print(f"SMILES: {smiles}, Features dim: {len(features)}")
    
    # Création du batch de graphes
    batch_graph = MolecularDataProcessor.smiles_to_graph_batch(
        smiles_list,
        atom_featurizer=atom_featurizer,
        bond_featurizer=bond_featurizer
    )
    
    print(f"Batch créé: {batch_graph.n_atoms} atomes, {batch_graph.n_bonds} liaisons")
    
    # Récupération des composants pour le modèle
    components = batch_graph.get_components()
    print(f"Composants: atom_features shape: {components[0].shape}")
    print(f"bond_features shape: {components[1].shape}")

if __name__ == "__main__":
    example_usage()