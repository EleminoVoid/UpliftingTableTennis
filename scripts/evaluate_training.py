#!/usr/bin/env python3
"""
Evaluate the most recent training run by reading TensorBoard logs.
Extract and display key metrics to determine if training was successful.
"""

import os
import sys
from pathlib import Path

# Add repo root to path
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

import paths

def evaluate_training():
    """Read and summarize the most recent training run."""
    logs_base = Path(paths.logs_path.replace('C:/', 'C:\\')) / 'tabledetection' / 'logs' / 'results'
    
    if not logs_base.exists():
        print(f"Logs directory not found: {logs_base}")
        return
    
    # Find the most recent batch_size 2 run
    log_dirs = sorted([d for d in logs_base.iterdir() if d.is_dir() and 'bs_02' in d.name], 
                      key=lambda x: x.stat().st_mtime, reverse=True)
    
    if not log_dirs:
        print("No recent training logs found with batch_size 2")
        return
    
    latest_log = log_dirs[0]
    print(f"\n{'='*80}")
    print(f"Evaluating Training Run:")
    print(f"  {latest_log.name}")
    print(f"  Path: {latest_log}")
    print(f"{'='*80}\n")
    
    # Try to read TensorBoard event files
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        print("TensorBoard not available in environment. Installing...")
        os.system(f"{sys.executable} -m pip install tensorboard -q")
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    
    ea = EventAccumulator(str(latest_log))
    ea.Reload()
    
    # Get available metrics
    print("Available scalar metrics:")
    scalars = ea.Tags()['scalars']
    for scalar in sorted(scalars):
        print(f"  - {scalar}")
    
    print("\n" + "="*80)
    print("TRAINING EVALUATION")
    print("="*80 + "\n")
    
    # Extract key metrics
    metrics_summary = {}
    
    if 'Validation/Loss' in scalars:
        val_loss = ea.Scalars('Validation/Loss')
        if val_loss:
            initial_loss = val_loss[0].value
            final_loss = val_loss[-1].value
            loss_improved = (initial_loss - final_loss) / initial_loss * 100 if initial_loss > 0 else 0
            metrics_summary['Validation Loss'] = {
                'initial': initial_loss,
                'final': final_loss,
                'improvement_pct': loss_improved,
                'steps': len(val_loss)
            }
            print(f"Validation Loss:")
            print(f"  Initial:  {initial_loss:.6f}")
            print(f"  Final:    {final_loss:.6f}")
            print(f"  Improvement: {loss_improved:.2f}%")
            print(f"  Eval steps: {len(val_loss)}")
    
    if 'Validation/PCK@5px' in scalars:
        pck5 = ea.Scalars('Validation/PCK@5px')
        if pck5:
            initial_pck = pck5[0].value
            final_pck = pck5[-1].value
            metrics_summary['PCK@5px'] = {
                'initial': initial_pck,
                'final': final_pck,
                'steps': len(pck5)
            }
            print(f"\nPCK@5px (% correct keypoints within 5px):")
            print(f"  Initial:  {initial_pck:.2f}%")
            print(f"  Final:    {final_pck:.2f}%")
            print(f"  Change:   {final_pck - initial_pck:+.2f}%")
    
    if 'Validation/AverageDistance' in scalars:
        avg_dist = ea.Scalars('Validation/AverageDistance')
        if avg_dist:
            initial_dist = avg_dist[0].value
            final_dist = avg_dist[-1].value
            dist_improved = (initial_dist - final_dist) / initial_dist * 100 if initial_dist > 0 else 0
            metrics_summary['Avg Distance'] = {
                'initial': initial_dist,
                'final': final_dist,
                'improvement_pct': dist_improved,
                'steps': len(avg_dist)
            }
            print(f"\nAverage Distance (pixels):")
            print(f"  Initial:  {initial_dist:.2f}")
            print(f"  Final:    {final_dist:.2f}")
            print(f"  Improvement: {dist_improved:.2f}%")
    
    # Overall assessment
    print(f"\n{'='*80}")
    print("ASSESSMENT")
    print(f"{'='*80}\n")
    
    quality_score = 0
    if metrics_summary.get('Validation Loss'):
        loss_improvement = metrics_summary['Validation Loss']['improvement_pct']
        if loss_improvement > 10:
            quality_score += 25
            print("✓ Loss improvement >10%: GOOD")
        elif loss_improvement > 0:
            quality_score += 10
            print("✓ Loss improved: OK")
        else:
            print("✗ Loss not improved: CONCERN")
    
    if metrics_summary.get('PCK@5px'):
        pck_final = metrics_summary['PCK@5px']['final']
        if pck_final > 0.5:
            quality_score += 25
            print("✓ Final PCK@5 >0.5%: GOOD")
        elif pck_final > 0:
            quality_score += 5
            print("✓ PCK@5 starting to improve: OK")
        else:
            print("⚠ PCK@5 still at 0%: needs more training")
    
    if metrics_summary.get('Avg Distance'):
        dist_improvement = metrics_summary['Avg Distance']['improvement_pct']
        if dist_improvement > 20:
            quality_score += 25
            print("✓ Distance improved >20%: GOOD")
        elif dist_improvement > 10:
            quality_score += 15
            print("✓ Distance improved >10%: OK")
        elif dist_improvement > 0:
            quality_score += 5
            print("✓ Distance improved: OK")
        else:
            print("✗ Distance not improved: CONCERN")
    
    # Training duration assessment
    steps_trained = len(val_loss) if 'val_loss' in locals() else 0
    if steps_trained > 50:
        quality_score += 25
        print(f"✓ Trained {steps_trained} validation steps (>50): GOOD")
    elif steps_trained > 10:
        quality_score += 10
        print(f"⚠ Trained {steps_trained} validation steps: NEEDS MORE TRAINING")
    else:
        print(f"✗ Only {steps_trained} validation steps: RESTART NEEDED")
    
    print(f"\nOVERALL QUALITY SCORE: {quality_score}/100")
    
    if quality_score >= 75:
        print("\n🟢 TRAINING IS GOOD ENOUGH - Ready to use or continue with fine-tuning")
        recommendation = "KEEP"
    elif quality_score >= 50:
        print("\n🟡 TRAINING SHOWS PROMISE - Continue training for better results")
        recommendation = "CONTINUE"
    else:
        print("\n🔴 TRAINING NEEDS IMPROVEMENT - Restart with adjusted parameters")
        recommendation = "RESTART"
    
    print(f"\nRECOMMENDATION: {recommendation}")
    return recommendation, metrics_summary

if __name__ == '__main__':
    evaluate_training()
