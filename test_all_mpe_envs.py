"""
Test ALL MPE environments to find which gives obs_dim=14
"""

from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def test_environment(env_name, env_func, **kwargs):
    """Test environment and return obs dimensions"""
    try:
        env = env_func(**kwargs)
        observations, _ = env.reset()
        
        obs_dims = {agent: obs.shape[0] for agent, obs in observations.items()}
        env.close()
        
        return obs_dims
    except Exception as e:
        return f"ERROR: {e}"

def main():
    logger.info("=" * 80)
    logger.info("🔍 TESTING ALL MPE ENVIRONMENTS")
    logger.info("=" * 80)
    
    # Suppress pygame warnings
    import warnings
    warnings.filterwarnings('ignore')
    
    from pettingzoo.mpe import (
        simple_v3,
        simple_adversary_v3,
        simple_crypto_v3,
        simple_push_v3,
        simple_reference_v3,
        simple_speaker_listener_v4,
        simple_spread_v3,
        simple_tag_v3,
        simple_world_comm_v3
    )
    
    tests = [
        ("simple_v3", simple_v3.parallel_env, {}),
        ("simple_adversary_v3 (N=2)", simple_adversary_v3.parallel_env, {'N': 2}),
        ("simple_adversary_v3 (N=3)", simple_adversary_v3.parallel_env, {'N': 3}),
        ("simple_crypto_v3", simple_crypto_v3.parallel_env, {}),
        ("simple_push_v3", simple_push_v3.parallel_env, {}),
        ("simple_reference_v3", simple_reference_v3.parallel_env, {}),
        ("simple_speaker_listener_v4", simple_speaker_listener_v4.parallel_env, {}),
        ("simple_spread_v3 (N=2)", simple_spread_v3.parallel_env, {'N': 2, 'continuous_actions': True}),
        ("simple_spread_v3 (N=3)", simple_spread_v3.parallel_env, {'N': 3, 'continuous_actions': True}),
        ("simple_spread_v3 (N=4)", simple_spread_v3.parallel_env, {'N': 4, 'continuous_actions': True}),
        ("simple_tag_v3 (N=2)", simple_tag_v3.parallel_env, {'num_good': 2, 'num_adversaries': 1}),
        ("simple_tag_v3 (N=3)", simple_tag_v3.parallel_env, {'num_good': 3, 'num_adversaries': 1}),
        ("simple_world_comm_v3", simple_world_comm_v3.parallel_env, {}),
    ]
    
    logger.info("\n🔎 Target: Find environment with obs_dim=14 or obs_dim=16\n")
    
    matches_14 = []
    matches_16 = []
    
    for env_name, env_func, kwargs in tests:
        logger.info(f"Testing: {env_name}")
        result = test_environment(env_name, env_func, **kwargs)
        
        if isinstance(result, dict):
            logger.info(f"  {result}")
            
            # Check for matches
            for agent, obs_dim in result.items():
                if obs_dim == 14:
                    matches_14.append((env_name, agent, obs_dim))
                    logger.info(f"  🎯 MATCH! {agent} has obs_dim=14")
                if obs_dim == 16:
                    matches_16.append((env_name, agent, obs_dim))
                    logger.info(f"  🎯 MATCH! {agent} has obs_dim=16")
        else:
            logger.info(f"  {result}")
        
        logger.info("")
    
    logger.info("=" * 80)
    logger.info("📊 RESULTS")
    logger.info("=" * 80)
    
    if matches_14:
        logger.info(f"\n✅ Found {len(matches_14)} match(es) for obs_dim=14:")
        for env_name, agent, obs_dim in matches_14:
            logger.info(f"   - {env_name} → {agent}: obs_dim={obs_dim}")
    else:
        logger.info("\n❌ No matches found for obs_dim=14")
    
    if matches_16:
        logger.info(f"\n✅ Found {len(matches_16)} match(es) for obs_dim=16:")
        for env_name, agent, obs_dim in matches_16:
            logger.info(f"   - {env_name} → {agent}: obs_dim={obs_dim}")
    else:
        logger.info("\n❌ No matches found for obs_dim=16")
    
    if not matches_14 and not matches_16:
        logger.info("\n🔴 CONCLUSION: Your models were trained with a CUSTOM environment!")
        logger.info("   The environment is NOT a standard PettingZoo MPE environment.")
        logger.info("   You need to:")
        logger.info("   1. Find the custom environment code used during training")
        logger.info("   2. Or recreate the environment based on training logs")
    
    logger.info("=" * 80)

if __name__ == '__main__':
    main()
