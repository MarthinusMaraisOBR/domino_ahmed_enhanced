#!/usr/bin/env python3
"""coefficient_loader.py - Load coefficients during testing"""

import pandas as pd
from pathlib import Path

class CoefficientManager:
    """Manage coefficient data for testing"""
    
    def __init__(self, base_path="/mnt/windows/ahmed-ml-project/ahmed_data/organized"):
        self.base_path = Path(base_path)
        self._load_all_coefficients()
    
    def _load_all_coefficients(self):
        """Load merged coefficient files"""
        self.coefficients = {}
        
        for split in ['train', 'val', 'test']:
            self.coefficients[split] = {}
            
            for resolution in ['coarse', 'fine']:
                merged_file = self.base_path / split / f"{resolution}_coefficients_all.csv"
                
                if merged_file.exists():
                    self.coefficients[split][resolution] = pd.read_csv(merged_file)
                    print(f"Loaded {split}/{resolution}: {len(self.coefficients[split][resolution])} cases")
    
    def get_coefficients(self, case_num, split='test'):
        """Get both coarse and fine coefficients for a case"""
        
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
    
    def calculate_baseline_error(self, case_num, split='test'):
        """Calculate baseline error (coarse vs fine)"""
        
        coeffs = self.get_coefficients(case_num, split)
        if coeffs is None:
            return None
        
        return {
            'cd_error': abs(coeffs['coarse_cd'] - coeffs['fine_cd']),
            'cl_error': abs(coeffs['coarse_cl'] - coeffs['fine_cl']),
            'cd_relative': abs(coeffs['coarse_cd'] - coeffs['fine_cd']) / abs(coeffs['fine_cd'] + 1e-10),
            'cl_relative': abs(coeffs['coarse_cl'] - coeffs['fine_cl']) / abs(coeffs['fine_cl'] + 1e-10)
        }
    
    def get_split_statistics(self, split='test'):
        """Get statistics for all cases in a split"""
        
        if split not in self.coefficients:
            return None
            
        stats = {}
        
        for resolution in ['coarse', 'fine']:
            if resolution not in self.coefficients[split]:
                continue
                
            df = self.coefficients[split][resolution]
            
            if 'cd' in df.columns and 'cl' in df.columns:
                stats[resolution] = {
                    'cd_mean': df['cd'].mean(),
                    'cd_std': df['cd'].std(),
                    'cl_mean': df['cl'].mean(),
                    'cl_std': df['cl'].std(),
                    'n_cases': len(df)
                }
        
        return stats
