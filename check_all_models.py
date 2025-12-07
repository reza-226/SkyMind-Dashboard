"""
Check all model dimensions
"""
import torch
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_model_dims(model_path):
    """Extract dimensions from a model checkpoint"""
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Get dimensions from network structure
        first_layer = checkpoint['actor']['network.0.weight']
        last_layer = checkpoint['actor']['network.4.weight']
        
        obs_dim = first_layer.shape[1]
        action_dim = last_layer.shape[0]
        
        return obs_dim, action_dim
    except Exception as e:
        return None, None

def scan_all_models():
    """Scan all model checkpoints"""
    
    base_path = Path("models")
    
    for level_dir in sorted(base_path.glob("level*")):
        logger.info(f"\n📁 {level_dir.name}")
        logger.info("=" * 60)
        
        for checkpoint_dir in sorted(level_dir.glob("**/final")):
            logger.info(f"\n  📂 {checkpoint_dir.relative_to(base_path)}")
            
            for model_file in sorted(checkpoint_dir.glob("*.pth")):
                obs_dim, action_dim = check_model_dims(model_file)
                
                if obs_dim:
                    logger.info(f"    ✅ {model_file.name}: obs={obs_dim}, action={action_dim}")
                else:
                    logger.info(f"    ❌ {model_file.name}: Failed d")

if __name__ == '__main__':
    scan_all_models()
