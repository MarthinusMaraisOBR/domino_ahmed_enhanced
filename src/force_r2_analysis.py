#!/usr/bin/env python3
"""
Ahmed Body DoMINO Model - Separate Drag and Lift Force R² Plots
Creates two individual PNG files: one for drag, one for lift
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os

# Set plotting style
plt.style.use('default')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

def calculate_r2_score(y_true, y_pred):
    """
    Calculate R² score manually (coefficient of determination).
    R² = 1 - (SS_res / SS_tot)
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Total sum of squares
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    
    # Residual sum of squares
    ss_res = np.sum((y_true - y_pred) ** 2)
    
    # R² calculation
    r2 = 1 - (ss_res / ss_tot)
    
    return r2

def extract_force_data():
    """
    Extract force data from your test results.
    """
    force_data = {
        471: {'drag_pred': 0.014622377, 'drag_true': 0.019690385, 'lift_pred': 0.013706224, 'lift_true': 0.017482838},
        451: {'drag_pred': 0.015251859, 'drag_true': 0.020506265, 'lift_pred': -0.0050089946, 'lift_true': -0.008251523},
        452: {'drag_pred': 0.013661954, 'drag_true': 0.016773319, 'lift_pred': -0.0009903582, 'lift_true': -0.0011722692},
        453: {'drag_pred': 0.015869789, 'drag_true': 0.021863844, 'lift_pred': 0.021097284, 'lift_true': 0.027807597},
        454: {'drag_pred': 0.017768567, 'drag_true': 0.020035118, 'lift_pred': -0.002086381, 'lift_true': -0.005423565},
        455: {'drag_pred': 0.018233161, 'drag_true': 0.021919718, 'lift_pred': 0.019724455, 'lift_true': 0.024284588},
        456: {'drag_pred': 0.017103598, 'drag_true': 0.02057663, 'lift_pred': -0.008685019, 'lift_true': -0.010250852},
        457: {'drag_pred': 0.013760599, 'drag_true': 0.019583957, 'lift_pred': 0.018575892, 'lift_true': 0.024955261},
        458: {'drag_pred': 0.013030946, 'drag_true': 0.015693163, 'lift_pred': 0.003862555, 'lift_true': 0.0017811507},
        459: {'drag_pred': 0.015502801, 'drag_true': 0.021159949, 'lift_pred': 0.012264274, 'lift_true': 0.026903482},
        460: {'drag_pred': 0.013010113, 'drag_true': 0.016847843, 'lift_pred': 0.014291094, 'lift_true': 0.022238567},
        461: {'drag_pred': 0.010820551, 'drag_true': 0.015669802, 'lift_pred': 0.004768997, 'lift_true': -0.00018326567},
        462: {'drag_pred': 0.019424804, 'drag_true': 0.019545946, 'lift_pred': 0.012816621, 'lift_true': 0.02637948},
        463: {'drag_pred': 0.009155005, 'drag_true': 0.011927249, 'lift_pred': 0.0025035492, 'lift_true': 0.0029476194},
        464: {'drag_pred': 0.015074913, 'drag_true': 0.018261002, 'lift_pred': 0.0008914949, 'lift_true': 0.00034978005},
        465: {'drag_pred': 0.015525596, 'drag_true': 0.017513907, 'lift_pred': 0.009950588, 'lift_true': 0.021725714},
        466: {'drag_pred': 0.009720119, 'drag_true': 0.012378348, 'lift_pred': 0.00067923655, 'lift_true': -0.004082053},
        467: {'drag_pred': 0.01763683, 'drag_true': 0.021258172, 'lift_pred': -0.0018979489, 'lift_true': -0.0029980158},
        468: {'drag_pred': 0.017694706, 'drag_true': 0.020116696, 'lift_pred': -0.0027041142, 'lift_true': -0.006364572},
        469: {'drag_pred': 0.013834336, 'drag_true': 0.017520998, 'lift_pred': -0.0049789846, 'lift_true': -0.009318217},
        470: {'drag_pred': 0.012916472, 'drag_true': 0.01712855, 'lift_pred': 0.018733507, 'lift_true': 0.023018533},
        472: {'drag_pred': 0.010906483, 'drag_true': 0.017138, 'lift_pred': 0.0042439336, 'lift_true': -0.0019516664},
        473: {'drag_pred': 0.01137045, 'drag_true': 0.015192661, 'lift_pred': 0.004646816, 'lift_true': -0.00062415795},
        474: {'drag_pred': 0.014308507, 'drag_true': 0.017342638, 'lift_pred': 0.0057271053, 'lift_true': 0.001138271},
        475: {'drag_pred': 0.011602955, 'drag_true': 0.015063497, 'lift_pred': -0.0023156349, 'lift_true': -0.0041994094},
        476: {'drag_pred': 0.011356012, 'drag_true': 0.015697008, 'lift_pred': 0.001169168, 'lift_true': -0.002219163},
        477: {'drag_pred': 0.015945854, 'drag_true': 0.01915284, 'lift_pred': -0.0010878507, 'lift_true': -0.0006776011},
        478: {'drag_pred': 0.009145568, 'drag_true': 0.011696235, 'lift_pred': 0.008901246, 'lift_true': 0.010303942},
        479: {'drag_pred': 0.018817097, 'drag_true': 0.019277247, 'lift_pred': -0.009958901, 'lift_true': -0.01042224},
        480: {'drag_pred': 0.01833296, 'drag_true': 0.02253193, 'lift_pred': -0.0015354007, 'lift_true': -0.00065256143},
        481: {'drag_pred': 0.012428884, 'drag_true': 0.014501071, 'lift_pred': 0.0040251226, 'lift_true': 9.413913e-05},
        482: {'drag_pred': 0.0177359, 'drag_true': 0.02002936, 'lift_pred': -0.0072830357, 'lift_true': -0.0040939674},
        483: {'drag_pred': 0.01416528, 'drag_true': 0.016104724, 'lift_pred': 0.015944848, 'lift_true': 0.018799026},
        484: {'drag_pred': 0.014116286, 'drag_true': 0.016610576, 'lift_pred': 0.0027674064, 'lift_true': 0.009491528},
        485: {'drag_pred': 0.014294538, 'drag_true': 0.017309153, 'lift_pred': 0.002036808, 'lift_true': 0.0016813837},
        486: {'drag_pred': 0.009363242, 'drag_true': 0.012209302, 'lift_pred': 0.0026755766, 'lift_true': 0.002545014},
        487: {'drag_pred': 0.018267166, 'drag_true': 0.01703201, 'lift_pred': 0.018063204, 'lift_true': 0.0032006553},
        488: {'drag_pred': 0.010996926, 'drag_true': 0.013594559, 'lift_pred': 0.011884054, 'lift_true': 0.010172614},
        489: {'drag_pred': 0.013515325, 'drag_true': 0.017592296, 'lift_pred': -0.004158451, 'lift_true': -0.0039974223},
        490: {'drag_pred': 0.013589865, 'drag_true': 0.01847162, 'lift_pred': 0.004321606, 'lift_true': -0.0021978985},
        491: {'drag_pred': 0.016976425, 'drag_true': 0.01900511, 'lift_pred': 0.0007583749, 'lift_true': -0.0019824503},
        492: {'drag_pred': 0.0127689075, 'drag_true': 0.017835112, 'lift_pred': 0.021588672, 'lift_true': 0.031423803},
        493: {'drag_pred': 0.014738664, 'drag_true': 0.017620172, 'lift_pred': -0.0014192088, 'lift_true': 0.000540521},
        494: {'drag_pred': 0.012391338, 'drag_true': 0.016364504, 'lift_pred': 0.014109424, 'lift_true': 0.017479029},
        495: {'drag_pred': 0.0113184815, 'drag_true': 0.014674928, 'lift_pred': 0.011758382, 'lift_true': 0.0185406},
        496: {'drag_pred': 0.010872352, 'drag_true': 0.014179592, 'lift_pred': -0.0045317, 'lift_true': -0.005436354},
        497: {'drag_pred': 0.016088285, 'drag_true': 0.016516969, 'lift_pred': 0.010899753, 'lift_true': 0.023130007},
        498: {'drag_pred': 0.012766424, 'drag_true': 0.018368697, 'lift_pred': -0.003208895, 'lift_true': -0.005080638},
        499: {'drag_pred': 0.013718652, 'drag_true': 0.015192436, 'lift_pred': -0.0031214622, 'lift_true': -0.0063606803},
        500: {'drag_pred': 0.015035077, 'drag_true': 0.018198896, 'lift_pred': 0.015663102, 'lift_true': 0.020889087}
    }
    
    return force_data

def create_drag_force_plot(drag_true, drag_pred, save_path="/data/ahmed_data/analysis/"):
    """
    Create drag force R² plot and save as PNG.
    """
    # Calculate R² and statistics
    r2 = calculate_r2_score(drag_true, drag_pred)
    rmse = np.sqrt(np.mean((drag_true - drag_pred) ** 2))
    mae = np.mean(np.abs(drag_true - drag_pred))
    corr_coef, _ = stats.pearsonr(drag_true, drag_pred)
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Plot scatter points
    ax.scatter(drag_true, drag_pred, c='#1f77b4', s=100, alpha=0.7, 
               edgecolors='white', linewidth=1.5)
    
    # Perfect prediction line (1:1 line)
    drag_min = min(drag_true.min(), drag_pred.min())
    drag_max = max(drag_true.max(), drag_pred.max())
    ax.plot([drag_min, drag_max], [drag_min, drag_max], 'k--', 
            linewidth=2, alpha=0.8, label='Perfect Prediction (1:1)')
    
    # Best fit line
    z = np.polyfit(drag_true, drag_pred, 1)
    p = np.poly1d(z)
    ax.plot(drag_true, p(drag_true), 'r-', linewidth=2, alpha=0.8, 
            label=f'Best Fit (y={z[0]:.3f}x+{z[1]:.4f})')
    
    # Labels and title
    ax.set_xlabel('Drag Force (True)', fontsize=16, fontweight='bold')
    ax.set_ylabel('Drag Force (Predicted)', fontsize=16, fontweight='bold')
    ax.set_title(f'Drag Force Prediction\nR² = {r2:.4f}', 
                fontsize=18, fontweight='bold', pad=20)
    
    # Styling
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12, frameon=True, fancybox=True, shadow=True)
    
    # Statistics text box
    stats_text = f"""Statistics:
R² = {r2:.4f}
RMSE = {rmse:.6f}
MAE = {mae:.6f}
Correlation = {corr_coef:.4f}
n = {len(drag_true)}"""
    
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=dict(boxstyle='round', 
            facecolor='white', alpha=0.9, edgecolor='gray'))
    
    # Adjust layout and save
    plt.tight_layout()
    
    os.makedirs(save_path, exist_ok=True)
    drag_file = os.path.join(save_path, "ahmed_drag_force_r2.png")
    plt.savefig(drag_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()  # Close to free memory
    
    return drag_file, r2

def create_lift_force_plot(lift_true, lift_pred, save_path="/data/ahmed_data/analysis/"):
    """
    Create lift force R² plot and save as PNG.
    """
    # Calculate R² and statistics
    r2 = calculate_r2_score(lift_true, lift_pred)
    rmse = np.sqrt(np.mean((lift_true - lift_pred) ** 2))
    mae = np.mean(np.abs(lift_true - lift_pred))
    corr_coef, _ = stats.pearsonr(lift_true, lift_pred)
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Plot scatter points
    ax.scatter(lift_true, lift_pred, c='#ff7f0e', s=100, alpha=0.7, 
               edgecolors='white', linewidth=1.5)
    
    # Perfect prediction line (1:1 line)
    lift_min = min(lift_true.min(), lift_pred.min())
    lift_max = max(lift_true.max(), lift_pred.max())
    ax.plot([lift_min, lift_max], [lift_min, lift_max], 'k--', 
            linewidth=2, alpha=0.8, label='Perfect Prediction (1:1)')
    
    # Best fit line
    z = np.polyfit(lift_true, lift_pred, 1)
    p = np.poly1d(z)
    ax.plot(lift_true, p(lift_true), 'r-', linewidth=2, alpha=0.8, 
            label=f'Best Fit (y={z[0]:.3f}x+{z[1]:.4f})')
    
    # Labels and title
    ax.set_xlabel('Lift Force (True)', fontsize=16, fontweight='bold')
    ax.set_ylabel('Lift Force (Predicted)', fontsize=16, fontweight='bold')
    ax.set_title(f'Lift Force Prediction\nR² = {r2:.4f}', 
                fontsize=18, fontweight='bold', pad=20)
    
    # Styling
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12, frameon=True, fancybox=True, shadow=True)
    
    # Statistics text box
    stats_text = f"""Statistics:
R² = {r2:.4f}
RMSE = {rmse:.6f}
MAE = {mae:.6f}
Correlation = {corr_coef:.4f}
n = {len(lift_true)}"""
    
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=dict(boxstyle='round', 
            facecolor='white', alpha=0.9, edgecolor='gray'))
    
    # Adjust layout and save
    plt.tight_layout()
    
    os.makedirs(save_path, exist_ok=True)
    lift_file = os.path.join(save_path, "ahmed_lift_force_r2.png")
    plt.savefig(lift_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()  # Close to free memory
    
    return lift_file, r2

def main():
    """
    Main function to create separate drag and lift force R² plots.
    """
    print("🚀 CREATING SEPARATE DRAG AND LIFT FORCE R² PLOTS")
    print("="*60)
    
    # Extract force data
    force_data = extract_force_data()
    print(f"📊 Loaded force data for {len(force_data)} test cases")
    
    # Extract arrays
    drag_true = np.array([data['drag_true'] for data in force_data.values()])
    drag_pred = np.array([data['drag_pred'] for data in force_data.values()])
    lift_true = np.array([data['lift_true'] for data in force_data.values()])
    lift_pred = np.array([data['lift_pred'] for data in force_data.values()])
    
    print(f"📈 Drag force range: {drag_true.min():.6f} to {drag_true.max():.6f}")
    print(f"📈 Lift force range: {lift_true.min():.6f} to {lift_true.max():.6f}")
    
    # Create drag force plot
    print("\n🎯 Creating drag force R² plot...")
    drag_file, drag_r2 = create_drag_force_plot(drag_true, drag_pred)
    print(f"✅ Drag plot saved: {drag_file}")
    print(f"   Drag R² = {drag_r2:.4f}")
    
    # Create lift force plot
    print("\n🎯 Creating lift force R² plot...")
    lift_file, lift_r2 = create_lift_force_plot(lift_true, lift_pred)
    print(f"✅ Lift plot saved: {lift_file}")
    print(f"   Lift R² = {lift_r2:.4f}")
    
    # Summary
    print("\n" + "="*60)
    print("📊 SUMMARY OF R² ANALYSIS")
    print("="*60)
    print(f"Drag Force R²: {drag_r2:.4f}")
    print(f"Lift Force R²: {lift_r2:.4f}")
    print(f"Combined Average: {(drag_r2 + lift_r2)/2:.4f}")
    
    # Performance assessment
    def assess_r2(r2_value):
        if r2_value >= 0.95:
            return "🎉 OUTSTANDING"
        elif r2_value >= 0.90:
            return "✅ EXCELLENT"
        elif r2_value >= 0.80:
            return "👍 GOOD"
        elif r2_value >= 0.70:
            return "📈 FAIR"
        else:
            return "⚠️ NEEDS IMPROVEMENT"
    
    print(f"\nDrag Performance: {assess_r2(drag_r2)}")
    print(f"Lift Performance: {assess_r2(lift_r2)}")
    
    print("\n📁 OUTPUT FILES:")
    print(f"   1. {drag_file}")
    print(f"   2. {lift_file}")
    print("\n🎉 Analysis complete!")
    print("="*60)

if __name__ == "__main__":
    main()