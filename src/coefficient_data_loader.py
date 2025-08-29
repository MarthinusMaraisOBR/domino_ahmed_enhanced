# coefficient_data_loader.py
# Enhanced data loader that integrates coefficient data with surface fields

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import Dict, Optional, Tuple
from torch.utils.data import Dataset


class EnhancedDatasetWithCoefficients(Dataset):
    """
    Enhanced dataset that loads surface fields and corresponding coefficients.
    """
    
    def __init__(
        self,
        npy_data_path: str,
        coefficient_path: str,
        split: str = "train",
        use_enhanced_features: bool = True,
        normalize_coefficients: bool = True,
    ):
        """
        Args:
            npy_data_path: Path to NPY files with surface data
            coefficient_path: Path to coefficient CSV files
            split: Data split (train/val/test)
            use_enhanced_features: Whether using 8-feature enhanced data
            normalize_coefficients: Whether to normalize coefficient targets
        """
        self.npy_data_path = Path(npy_data_path)
        self.coefficient_path = Path(coefficient_path)
        self.split = split
        self.use_enhanced_features = use_enhanced_features
        self.normalize_coefficients = normalize_coefficients
        
        # Get all NPY files
        self.npy_files = sorted(list(self.npy_data_path.glob("*.npy")))
        print(f"Found {len(self.npy_files)} NPY files in {self.npy_data_path}")
        
        # Load coefficient data
        self._load_coefficients()
        
        # Compute coefficient statistics for normalization
        if normalize_coefficients:
            self._compute_coefficient_stats()
    
    def _load_coefficients(self):
        """Load coefficient data from CSV files."""
        self.coefficients = {}
        
        # Try to load merged coefficient file
        merged_coarse = self.coefficient_path / self.split / "coarse_coefficients_all.csv"
        merged_fine = self.coefficient_path / self.split / "fine_coefficients_all.csv"
        
        if merged_coarse.exists() and merged_fine.exists():
            print(f"Loading merged coefficient files for {self.split}")
            coarse_df = pd.read_csv(merged_coarse)
            fine_df = pd.read_csv(merged_fine)
            
            # Create mapping from case number to coefficients
            for _, row in coarse_df.iterrows():
                case_num = int(row['case'])
                self.coefficients[f"run_{case_num}"] = {
                    'coarse_cd': row['cd'],
                    'coarse_cl': row['cl']
                }
            
            for _, row in fine_df.iterrows():
                case_num = int(row['case'])
                if f"run_{case_num}" in self.coefficients:
                    self.coefficients[f"run_{case_num}"].update({
                        'fine_cd': row['cd'],
                        'fine_cl': row['cl']
                    })
            
            print(f"Loaded coefficients for {len(self.coefficients)} cases")
        else:
            print(f"Warning: Coefficient files not found at {self.coefficient_path}")
            # Create dummy coefficients
            for npy_file in self.npy_files:
                case_name = npy_file.stem
                self.coefficients[case_name] = {
                    'coarse_cd': 0.0, 'coarse_cl': 0.0,
                    'fine_cd': 0.0, 'fine_cl': 0.0
                }
    
    def _compute_coefficient_stats(self):
        """Compute mean and std for coefficient normalization."""
        all_cd_fine = []
        all_cl_fine = []
        all_cd_coarse = []
        all_cl_coarse = []
        
        for coeffs in self.coefficients.values():
            all_cd_fine.append(coeffs['fine_cd'])
            all_cl_fine.append(coeffs['fine_cl'])
            all_cd_coarse.append(coeffs['coarse_cd'])
            all_cl_coarse.append(coeffs['coarse_cl'])
        
        self.coeff_stats = {
            'fine_cd_mean': np.mean(all_cd_fine),
            'fine_cd_std': np.std(all_cd_fine),
            'fine_cl_mean': np.mean(all_cl_fine),
            'fine_cl_std': np.std(all_cl_fine),
            'coarse_cd_mean': np.mean(all_cd_coarse),
            'coarse_cd_std': np.std(all_cd_coarse),
            'coarse_cl_mean': np.mean(all_cl_coarse),
            'coarse_cl_std': np.std(all_cl_coarse),
        }
        
        print(f"Coefficient statistics computed:")
        print(f"  Fine Cd: mean={self.coeff_stats['fine_cd_mean']:.4f}, std={self.coeff_stats['fine_cd_std']:.4f}")
        print(f"  Fine Cl: mean={self.coeff_stats['fine_cl_mean']:.4f}, std={self.coeff_stats['fine_cl_std']:.4f}")
    
    def __len__(self):
        return len(self.npy_files)
    
    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        """Load NPY data and corresponding coefficients."""
        
        # Load NPY file
        npy_file = self.npy_files[idx]
        data = np.load(npy_file, allow_pickle=True).item()
        
        # Get case name
        case_name = npy_file.stem
        
        # Prepare output dictionary
        output = {}
        
        # Add all NPY data to output
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                output[key] = torch.from_numpy(value.astype(np.float32))
            else:
                output[key] = value
        
        # Validate enhanced features
        if self.use_enhanced_features and 'surface_fields' in output:
            surface_fields = output['surface_fields']
            if surface_fields.shape[-1] != 8:
                print(f"Warning: Expected 8 features, got {surface_fields.shape[-1]} for {case_name}")
        
        # Add coefficients
        if case_name in self.coefficients:
            coeffs = self.coefficients[case_name]
            
            # Create coefficient tensors
            coarse_coeffs = torch.tensor([coeffs['coarse_cd'], coeffs['coarse_cl']], dtype=torch.float32)
            fine_coeffs = torch.tensor([coeffs['fine_cd'], coeffs['fine_cl']], dtype=torch.float32)
            
            # Normalize if requested
            if self.normalize_coefficients and hasattr(self, 'coeff_stats'):
                fine_coeffs[0] = (fine_coeffs[0] - self.coeff_stats['fine_cd_mean']) / (self.coeff_stats['fine_cd_std'] + 1e-8)
                fine_coeffs[1] = (fine_coeffs[1] - self.coeff_stats['fine_cl_mean']) / (self.coeff_stats['fine_cl_std'] + 1e-8)
                
                coarse_coeffs[0] = (coarse_coeffs[0] - self.coeff_stats['coarse_cd_mean']) / (self.coeff_stats['coarse_cd_std'] + 1e-8)
                coarse_coeffs[1] = (coarse_coeffs[1] - self.coeff_stats['coarse_cl_mean']) / (self.coeff_stats['coarse_cl_std'] + 1e-8)
            
            output['coarse_coefficients'] = coarse_coeffs
            output['fine_coefficients'] = fine_coeffs
            output['coefficient_targets'] = fine_coeffs  # Target for training
        else:
            # Dummy coefficients if not found
            output['coarse_coefficients'] = torch.zeros(2, dtype=torch.float32)
            output['fine_coefficients'] = torch.zeros(2, dtype=torch.float32)
            output['coefficient_targets'] = torch.zeros(2, dtype=torch.float32)
        
        return output
    
    def get_coefficient_stats(self) -> Dict:
        """Return coefficient statistics for denormalization."""
        return self.coeff_stats if hasattr(self, 'coeff_stats') else None


class CoefficientManager:
    """
    Manager for coefficient data during testing.
    """
    
    def __init__(self, base_path: str = "/mnt/windows/ahmed-ml-project/ahmed_data/organized"):
        self.base_path = Path(base_path)
        self.coefficients = {}
        self._load_all_coefficients()
    
    def _load_all_coefficients(self):
        """Load all coefficient files."""
        for split in ['train', 'val', 'test']:
            self.coefficients[split] = {}
            
            for resolution in ['coarse', 'fine']:
                merged_file = self.base_path / split / f"{resolution}_coefficients_all.csv"
                
                if merged_file.exists():
                    df = pd.read_csv(merged_file)
                    self.coefficients[split][resolution] = df
                    print(f"Loaded {split}/{resolution}: {len(df)} cases")
    
    def get_coefficients(self, case_num: int, split: str = 'test') -> Optional[Dict]:
        """Get coefficients for a specific case."""
        if split not in self.coefficients:
            return None
        
        if 'coarse' not in self.coefficients[split] or 'fine' not in self.coefficients[split]:
            return None
        
        coarse_df = self.coefficients[split]['coarse']
        fine_df = self.coefficients[split]['fine']
        
        coarse_row = coarse_df[coarse_df['case'] == case_num]
        fine_row = fine_df[fine_df['case'] == case_num]
        
        if coarse_row.empty or fine_row.empty:
            return None
        
        return {
            'coarse_cd': coarse_row['cd'].iloc[0],
            'coarse_cl': coarse_row['cl'].iloc[0],
            'fine_cd': fine_row['cd'].iloc[0],
            'fine_cl': fine_row['cl'].iloc[0],
            'case': case_num
        }
    
    def calculate_improvement(self, predicted_cd: float, predicted_cl: float, 
                             case_num: int, split: str = 'test') -> Dict:
        """Calculate improvement metrics for predictions."""
        coeffs = self.get_coefficients(case_num, split)
        if coeffs is None:
            return None
        
        # Calculate errors
        coarse_cd_error = abs(coeffs['coarse_cd'] - coeffs['fine_cd'])
        pred_cd_error = abs(predicted_cd - coeffs['fine_cd'])
        
        coarse_cl_error = abs(coeffs['coarse_cl'] - coeffs['fine_cl'])
        pred_cl_error = abs(predicted_cl - coeffs['fine_cl'])
        
        # Calculate improvements
        cd_improvement = (coarse_cd_error - pred_cd_error) / (coarse_cd_error + 1e-10) * 100
        cl_improvement = (coarse_cl_error - pred_cl_error) / (coarse_cl_error + 1e-10) * 100
        
        return {
            'cd_improvement': cd_improvement,
            'cl_improvement': cl_improvement,
            'cd_relative_error': pred_cd_error / (abs(coeffs['fine_cd']) + 1e-10),
            'cl_relative_error': pred_cl_error / (abs(coeffs['fine_cl']) + 1e-10),
            'ground_truth': coeffs
        }
