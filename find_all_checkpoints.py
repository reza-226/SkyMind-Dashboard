"""
Find ALL checkpoints and analyze training history
"""

import torch
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def analyze_all_checkpoints():
    """Find and analyze all checkpoint files"""
    
    base_dir = Path("models/level3_complex")
    
    all_checkpoints = []
    
    # Search all subdirectories
    for pth_file in base_dir.rglob("*.pth"):
        try:
            checkpoint = torch.load(pth_file, map_location='cpu', weights_only=False)
            
            if 'actor' in checkpoint:
                actor_state = checkpoint['actor']
                first_weight = actor_state['network.0.weight']
                hidden_dim, obs_dim = first_weight.shape
                last_weight = actor_state['network.4.weight']
                action_dim = last_weight.shape[0]
                
                info = {
                    'path': str(pth_file.relative_to(base_dir)),
                    'obs_dim': obs_dim,
                    'action_dim': action_dim,
                    'hidden_dim': hidden_dim,
                    'size_kb': pth_file.stat().st_size / 1024,
                    'modified': pth_file.stat().st_mtime
                }
                
                if 'episode' in checkpoint:
                    info['episode'] = checkpoint['episode']
                if 'step' in checkpoint:
                    info['step'] = checkpoint['step']
                
                all_checkpoints.append(info)
                
        except Exception as e:
            logger.warning(f"   ⚠️  Could not load {pth_file.name}: {e}")
    
    return sorted(all_checkpoints, key=lambda x: x['modified'])

def main():
    logger.info("=" * 80)
    logger.info("🔍 SEARCHING ALL CHECKPOINTS")
    logger.info("=" * 80)
    
    checkpoints = analyze_all_checkpoints()
    
    if not checkpoints:
        logger.error("\n❌ No checkpoints found!")
        return
    
    logger.info(f"\n✅ Found {len(checkpoints)} checkpoint(s)\n")
    
    # Group by obs_dim
    by_obs_dim = {}
    for ckpt in checkpoints:
        obs = ckpt['obs_dim']
        if obs not in by_obs_dim:
            by_obs_dim[obs] = []
        by_obs_dim[obs].append(ckpt)
    
    logger.info("📊 Grouped by obs_dim:\n")
    
    for obs_dim in sorted(by_obs_dim.keys()):
        ckpts = by_obs_dim[obs_dim]
        logger.info(f"   obs_dim={obs_dim} → {len(ckpts)} checkpoint(s)")
        
        # Calculate N from formula
        if (obs_dim - 4) % 4 == 0:
            n = (obs_dim - 4) // 4
            logger.info(f"      (simple_spread N={n})")
        else:
            logger.info(f"      (⚠️  doesn't match simple_spread formula)")
        
        for ckpt in ckpts:
            logger.info(f"      - {ckpt['path']}")
            if 'episode' in ckpt:
                logger.info(f"        Episode: {ckpt['episode']}")
        logger.info("")
    
    logger.info("=" * 80)
    logger.info("💡 RECOMMENDATION")
    logger.info("=" * 80)
    
    if 12 in by_obs_dim:
        logger.info("✅ Found obs_dim=12 checkpoints (N=2)")
        logger.info("   These should work with: simple_spread_v3(N=2)")
        logger.info("\n   To evaluate:")
        logger.info("   1. Use checkpoints with obs_dim=12")
        logger.info("   2. Create environment with N=2")
    else:
        logger.info("❌ No obs_dim=12 checkpoints found")
        logger.info("   Options:")
        logger.info("   1. Re-train from scratch")
        logger.info("   2. Check backup/older checkpoint directories")
    
    logger.info("=" * 80)

if __name__ == '__main__':
    main()
