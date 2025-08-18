#!/usr/bin/env python3
"""
Monitor the progress of the fixed Enhanced DoMINO training
"""

import time
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime

def monitor_training_log(log_file="train_fixed.log"):
    """Monitor training progress from log file."""
    
    print("="*80)
    print("MONITORING FIXED ENHANCED DOMINO TRAINING")
    print("="*80)
    
    if not Path(log_file).exists():
        print(f"❌ Log file not found: {log_file}")
        print("Make sure training is running with: bash start_fixed_training.sh")
        return
    
    print(f"\n📊 Monitoring: {log_file}")
    print("Press Ctrl+C to stop monitoring\n")
    
    last_position = 0
    epoch_losses = []
    improvements = []
    
    try:
        while True:
            with open(log_file, 'r') as f:
                f.seek(last_position)
                new_lines = f.readlines()
                last_position = f.tell()
            
            for line in new_lines:
                # Look for key indicators
                
                # Check for epoch start
                if "EPOCH" in line and "=====" not in line:
                    print(f"\n{line.strip()}")
                
                # Check for loss values
                if "loss norm:" in line or "total_loss" in line:
                    try:
                        # Extract loss value
                        if "loss norm:" in line:
                            loss_str = line.split("loss norm:")[-1].strip()
                            loss_val = float(loss_str)
                        else:
                            parts = line.split()
                            for i, part in enumerate(parts):
                                if "total" in part.lower():
                                    loss_val = float(parts[i+1])
                                    break
                        
                        epoch_losses.append(loss_val)
                        print(f"  Loss: {loss_val:.6f}")
                        
                        # Check if loss is decreasing
                        if len(epoch_losses) > 10:
                            recent_trend = epoch_losses[-1] / epoch_losses[-10]
                            if recent_trend < 0.9:
                                print(f"  ✅ Loss decreasing (10-batch trend: {(1-recent_trend)*100:.1f}% reduction)")
                            elif recent_trend > 1.1:
                                print(f"  ⚠️ Loss increasing! (10-batch trend: {(recent_trend-1)*100:.1f}% increase)")
                    except:
                        pass
                
                # Check for improvement metrics
                if "improvement" in line.lower() or "Improvement" in line:
                    try:
                        # Extract improvement value
                        import re
                        match = re.search(r'([-+]?\d*\.?\d+)%', line)
                        if match:
                            improvement = float(match.group(1))
                            improvements.append(improvement)
                            
                            if improvement > 0:
                                print(f"  ✅ Improvement: {improvement:.1f}%")
                            else:
                                print(f"  ❌ Degradation: {improvement:.1f}%")
                    except:
                        pass
                
                # Check for debug output from model
                if "[Training Mode]" in line or "[Inference Mode]" in line:
                    print(f"  {line.strip()}")
                
                # Check for coarse/fine statistics
                if "Coarse input stats:" in line or "Fine features" in line:
                    print(f"  {line.strip()}")
                
                # Check for errors or warnings
                if "ERROR" in line or "Error" in line:
                    print(f"  ❌ ERROR: {line.strip()}")
                
                if "WARNING" in line or "Warning" in line:
                    print(f"  ⚠️ WARNING: {line.strip()}")
                
                # Check for validation results
                if "Validation loss:" in line or "valid" in line.lower():
                    print(f"  {line.strip()}")
            
            # Show summary every 60 seconds
            if len(epoch_losses) > 0 and int(time.time()) % 60 == 0:
                print(f"\n📈 Training Summary at {datetime.now().strftime('%H:%M:%S')}:")
                print(f"  Total batches processed: {len(epoch_losses)}")
                print(f"  Current loss: {epoch_losses[-1]:.6f}")
                print(f"  Initial loss: {epoch_losses[0]:.6f}")
                print(f"  Overall reduction: {(1 - epoch_losses[-1]/epoch_losses[0])*100:.1f}%")
                
                if improvements:
                    print(f"  Latest improvement: {improvements[-1]:.1f}%")
                    print(f"  Average improvement: {np.mean(improvements):.1f}%")
            
            time.sleep(2)  # Check every 2 seconds
            
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")
        
        # Plot training progress if we have data
        if epoch_losses:
            print("\n📊 Generating training plot...")
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Plot loss
            ax1.plot(epoch_losses)
            ax1.set_xlabel('Iteration')
            ax1.set_ylabel('Loss')
            ax1.set_title('Training Loss')
            ax1.set_yscale('log')
            ax1.grid(True)
            
            # Plot improvements if available
            if improvements:
                ax2.plot(improvements)
                ax2.axhline(y=0, color='r', linestyle='--', label='Baseline')
                ax2.set_xlabel('Validation Step')
                ax2.set_ylabel('Improvement (%)')
                ax2.set_title('Improvement over Coarse Baseline')
                ax2.grid(True)
                ax2.legend()
            
            plt.tight_layout()
            plt.savefig('training_progress_fixed.png', dpi=150)
            print(f"✅ Plot saved to training_progress_fixed.png")

def check_checkpoint_quality():
    """Check if new checkpoints are improving."""
    
    print("\n" + "="*80)
    print("CHECKING CHECKPOINT QUALITY")
    print("="*80)
    
    checkpoint_dir = Path("outputs/Ahmed_Dataset/enhanced_fixed/models")
    
    if not checkpoint_dir.exists():
        print(f"❌ Checkpoint directory not found: {checkpoint_dir}")
        return
    
    checkpoints = list(checkpoint_dir.glob("*.pt"))
    
    if not checkpoints:
        print("❌ No checkpoints found yet")
        return
    
    print(f"\n📊 Found {len(checkpoints)} checkpoints:")
    
    # Sort by modification time
    checkpoints.sort(key=lambda x: x.stat().st_mtime)
    
    for ckpt in checkpoints[-5:]:  # Show last 5
        size_mb = ckpt.stat().st_size / (1024*1024)
        mod_time = datetime.fromtimestamp(ckpt.stat().st_mtime)
        print(f"  {ckpt.name}: {size_mb:.1f}MB, modified {mod_time.strftime('%H:%M:%S')}")
    
    # Load and check latest checkpoint
    latest = checkpoints[-1]
    print(f"\n🔍 Analyzing latest: {latest.name}")
    
    try:
        import torch
        ckpt = torch.load(latest, map_location='cpu')
        
        # Check for residual weights
        residual_keys = [k for k in ckpt.keys() if 'residual' in k]
        if residual_keys:
            print(f"  ⚠️ Residual weights still present: {len(residual_keys)} keys")
            for key in residual_keys[:2]:
                print(f"    {key}: norm={ckpt[key].norm():.4f}")
        else:
            print(f"  ✅ No residual weights (good!)")
        
        # Check output projection statistics
        output_keys = [k for k in ckpt.keys() if 'output_projection' in k and 'weight' in k]
        if output_keys:
            key = output_keys[0]
            weight = ckpt[key]
            if hasattr(weight, 'mean'):
                print(f"  Output weights: mean={weight.mean():.6f}, std={weight.std():.6f}")
                
                if weight.std() < 0.01:
                    print(f"    ⚠️ Low variance in output weights!")
                else:
                    print(f"    ✅ Good variance in output weights")
                    
    except Exception as e:
        print(f"  ❌ Error loading checkpoint: {str(e)}")

def main():
    """Main monitoring function."""
    
    print("\n🔍 ENHANCED DOMINO TRAINING MONITOR\n")
    
    # Check if training is running
    log_file = "train_fixed.log"
    
    if not Path(log_file).exists():
        print(f"Log file not found. Starting training first...")
        print("\nRun: bash start_fixed_training.sh")
        return
    
    # Monitor training
    monitor_training_log(log_file)
    
    # After monitoring stops, check checkpoint quality
    check_checkpoint_quality()
    
    print("\n" + "="*80)
    print("MONITORING COMPLETE")
    print("="*80)
    
    print("\n📝 Next steps:")
    print("1. Check training_progress_fixed.png for loss curves")
    print("2. If loss is decreasing steadily, continue training")
    print("3. If loss plateaus or increases, stop and debug")
    print("4. Test on training data after 100 epochs")

if __name__ == "__main__":
    main()
