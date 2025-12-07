"""
Find all checkpoint files in the project
"""

from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def find_checkpoints():
    """Find all .pth checkpoint files"""
    
    logger.info("=" * 80)
    logger.info("🔍 SEARCHING FOR CHECKPOINT FILES")
    logger.info("=" * 80)
    
    project_root = Path(".")
    checkpoint_files = list(project_root.rglob("*.pth"))
    
    if not checkpoint_files:
        logger.info("\n❌ No .pth files found!")
        logger.info("\n💡 Possible reasons:")
        logger.info("   1. Training hasn't been completed yet")
        logger.info("   2. Checkpoints are in a different location")
        logger.info("   3. Files have different extension")
        return
    
    logger.info(f"\n✅ Found {len(checkpoint_files)} checkpoint file(s):\n")
    
    # Group by directory
    by_dir = {}
    for f in checkpoint_files:
        dir_name = str(f.parent)
        if dir_name not in by_dir:
            by_dir[dir_name] = []
        by_dir[dir_name].append(f.name)
    
    # Display organized by directory
    for dir_name, files in sorted(by_dir.items()):
        logger.info(f"📁 {dir_name}/")
        for filename in sorted(files):
            file_path = Path(dir_name) / filename
            size_kb = file_path.stat().st_size / 1024
            logger.info(f"   ├─ {filename} ({size_kb:.1f} KB)")
        logger.info("")
    
    logger.info("=" * 80)
    logger.info("💡 NEXT STEPS:")
    logger.info("=" * 80)
    logger.info("Update the checkpoint path in evaluate_simple_tag.py:")
    logger.info("   CHECKPOINT_DIR = '<correct_path_from_above>'")
    logger.info("=" * 80)

if __name__ == '__main__':
    find_checkpoints()
