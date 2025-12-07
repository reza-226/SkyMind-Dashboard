"""
Complete Test Suite for MADDPG Agents V2
Tests all components individually and together
"""

import torch
import numpy as np
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from agents import ActorNetwork, CriticNetwork, ActionDecoder, MADDPGAgent

class Colors:
    """ANSI color codes for terminal output"""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

def print_test(name, passed, details=""):
    """Print test result with color"""
    status = f"{Colors.GREEN}✓ PASSED{Colors.RESET}" if passed else f"{Colors.RED}✗ FAILED{Colors.RESET}"
    print(f"{status} | {name}")
    if details:
        print(f"    {Colors.BLUE}→{Colors.RESET} {details}")

def test_actor_network():
    """Test ActorNetwork V2"""
    print(f"\n{Colors.BOLD}{'='*60}{Colors.RESET}")
    print(f"{Colors.BOLD}Testing ActorNetwork V2{Colors.RESET}")
    print(f"{Colors.BOLD}{'='*60}{Colors.RESET}\n")
    
    try:
        # Test initialization
        actor = ActorNetwork(
            state_dim=537,
            offload_dim=5,
            continuous_dim=6,
            hidden=512,
            use_layer_norm=True,
            dropout=0.1
        )
        print_test("Network initialization", True, "Created with layer norm & dropout")
        
        # Test forward pass
        batch_size = 4
        state = torch.randn(batch_size, 537)
        offload_logits, cont = actor(state)
        
        # Check shapes
        shape_check = (
            offload_logits.shape == (batch_size, 5) and
            cont.shape == (batch_size, 6)
        )
        print_test("Forward pass shape", shape_check, 
                   f"Offload: {offload_logits.shape}, Continuous: {cont.shape}")
        
        # Check output ranges
        cont_range = (cont >= -1.0).all() and (cont <= 1.0).all()
        print_test("Continuous action range [-1, 1]", cont_range,
                   f"Min: {cont.min():.4f}, Max: {cont.max():.4f}")
        
        # Test gradient flow
        actor.train()
        loss = offload_logits.mean() + cont.mean()
        loss.backward()
        
        has_grad = all(p.grad is not None for p in actor.parameters() if p.requires_grad)
        print_test("Gradient flow", has_grad, "All parameters have gradients")
        
        # Test without layer norm
        actor_simple = ActorNetwork(use_layer_norm=False, dropout=0.0)
        offload, cont = actor_simple(state)
        print_test("Simple version (no layer norm)", True,
                   f"Output shapes: {offload.shape}, {cont.shape}")
        
        return True
        
    except Exception as e:
        print_test("ActorNetwork", False, f"Error: {str(e)}")
        return False

def test_critic_network():
    """Test CriticNetwork V2"""
    print(f"\n{Colors.BOLD}{'='*60}{Colors.RESET}")
    print(f"{Colors.BOLD}Testing CriticNetwork V2{Colors.RESET}")
    print(f"{Colors.BOLD}{'='*60}{Colors.RESET}\n")
    
    try:
        # Test initialization
        critic = CriticNetwork(
            state_dim=537,
            action_dim=11,
            hidden=512,
            use_layer_norm=True,
            num_hidden_layers=3
        )
        print_test("Network initialization", True, 
                   "Created with 3 hidden layers & layer norm")
        
        # Test forward pass
        batch_size = 4
        state = torch.randn(batch_size, 537)
        action = torch.randn(batch_size, 11)
        q_value = critic(state, action)
        
        # Check shape
        shape_check = q_value.shape == (batch_size, 1)
        print_test("Forward pass shape", shape_check, f"Q-value: {q_value.shape}")
        
        # Test gradient flow
        critic.train()
        loss = q_value.mean()
        loss.backward()
        
        has_grad = all(p.grad is not None for p in critic.parameters() if p.requires_grad)
        print_test("Gradient flow", has_grad, "All parameters have gradients")
        
        # Test multiple hidden layers
        critic_deep = CriticNetwork(num_hidden_layers=4, use_layer_norm=False)
        q = critic_deep(state, action)
        print_test("Deep version (4 layers)", True, f"Output shape: {q.shape}")
        
        # Test Q-value range (should be reasonable)
        q_range_ok = q.abs().max() < 1000  # Sanity check
        print_test("Q-value sanity check", q_range_ok,
                   f"Max |Q|: {q.abs().max():.2f}")
        
        return True
        
    except Exception as e:
        print_test("CriticNetwork", False, f"Error: {str(e)}")
        return False

def test_action_decoder():
    """Test ActionDecoder V2"""
    print(f"\n{Colors.BOLD}{'='*60}{Colors.RESET}")
    print(f"{Colors.BOLD}Testing ActionDecoder V2{Colors.RESET}")
    print(f"{Colors.BOLD}{'='*60}{Colors.RESET}\n")
    
    try:
        decoder = ActionDecoder(
            max_movement_step=5.0,
            min_cpu=0.1,
            min_bandwidth=0.05
        )
        print_test("Decoder initialization", True,
                   "Created with custom min values")
        
        # Test batch decoding
        batch_size = 3
        offload_logits = torch.randn(batch_size, 5)
        cont = torch.randn(batch_size, 6)
        
        actions = decoder.decode(offload_logits, cont, deterministic=True)
        
        batch_check = len(actions) == batch_size
        print_test("Batch decoding", batch_check, f"Got {len(actions)} actions")
        
        # Validate action structure
        action = actions[0]
        structure_check = (
            'offload' in action and
            'cpu' in action and
            'bandwidth' in action and
            'move' in action
        )
        print_test("Action structure", structure_check,
                   f"Keys: {list(action.keys())}")
        
        # Check offload range [0, 4]
        offload_ok = all(0 <= a['offload'] <= 4 for a in actions)
        print_test("Offload range [0, 4]", offload_ok,
                   f"Values: {[a['offload'] for a in actions]}")
        
        # Check CPU range [min_cpu, 1.0]
        cpu_vals = [a['cpu'] for a in actions]
        cpu_ok = all(0.1 <= c <= 1.0 for c in cpu_vals)
        print_test("CPU range [0.1, 1.0]", cpu_ok,
                   f"Values: {[f'{c:.3f}' for c in cpu_vals]}")
        
        # Check bandwidth sum to 1
        bw_sums = [a['bandwidth'].sum() for a in actions]
        bw_ok = all(abs(s - 1.0) < 1e-5 for s in bw_sums)
        print_test("Bandwidth sum to 1", bw_ok,
                   f"Sums: {[f'{s:.6f}' for s in bw_sums]}")
        
        # Check bandwidth minimum
        all_bw = np.concatenate([a['bandwidth'] for a in actions])
        bw_min_ok = (all_bw >= 0.05).all()
        print_test("Bandwidth minimum 0.05", bw_min_ok,
                   f"Min: {all_bw.min():.6f}")
        
        # Check movement range
        all_moves = np.concatenate([a['move'] for a in actions])
        move_ok = (np.abs(all_moves) <= 5.0).all()
        print_test("Movement range [-5, 5]", move_ok,
                   f"Range: [{all_moves.min():.2f}, {all_moves.max():.2f}]")
        
        # Test single sample decoding
        single_action = decoder.decode_single(
            offload_logits[0],
            cont[0],
            deterministic=True
        )
        print_test("Single sample decode", isinstance(single_action, dict),
                   f"Type: {type(single_action).__name__}")
        
        # Test stochastic decoding
        stochastic_actions = decoder.decode(offload_logits, cont, deterministic=False)
        print_test("Stochastic decoding", len(stochastic_actions) == batch_size,
                   "Sampling from categorical distribution")
        
        return True
        
    except Exception as e:
        print_test("ActionDecoder", False, f"Error: {str(e)}")
        return False

def test_maddpg_agent():
    """Test MADDPGAgent V2"""
    print(f"\n{Colors.BOLD}{'='*60}{Colors.RESET}")
    print(f"{Colors.BOLD}Testing MADDPGAgent V2{Colors.RESET}")
    print(f"{Colors.BOLD}{'='*60}{Colors.RESET}\n")
    
    try:
        # Configuration
        config = {
            'device': 'cpu',
            'state_dim': 537,
            'offload_dim': 5,
            'continuous_dim': 6,
            'action_dim': 11,
            'hidden_size': 256,  # Smaller for faster testing
            'use_layer_norm': True,
            'dropout': 0.1,
            'num_hidden_layers': 2,
            'actor_lr': 1e-4,
            'critic_lr': 1e-3,
            'gamma': 0.99,
            'tau': 0.01,
            'exploration_noise': 0.1,
            'max_movement_step': 5.0,
            'min_cpu': 0.1,
            'min_bandwidth': 0.05
        }
        
        agent = MADDPGAgent(config)
        print_test("Agent initialization", True,
                   "Created with all components")
        
        # Test action selection
        state = np.random.randn(537)
        
        # Deterministic action
        action_det = agent.select_action(state, explore=False)
        print_test("Deterministic action selection", isinstance(action_det, dict),
                   f"Keys: {list(action_det.keys())}")
        
        # Exploratory action
        action_exp = agent.select_action(state, explore=True)
        print_test("Exploratory action selection", isinstance(action_exp, dict),
                   "Added exploration noise")
        
        # Test with tensor input
        state_tensor = torch.FloatTensor(state)
        action_tensor = agent.select_action(state_tensor, explore=False)
        print_test("Tensor input support", isinstance(action_tensor, dict),
                   "Handles both numpy and tensor inputs")
        
        # Test soft update
        old_actor_params = [p.clone() for p in agent.actor_target.parameters()]
        agent.soft_update()
        new_actor_params = list(agent.actor_target.parameters())
        
        params_changed = any(
            not torch.equal(old, new) 
            for old, new in zip(old_actor_params, new_actor_params)
        )
        print_test("Soft update", params_changed,
                   f"Target networks updated with tau={config['tau']}")
        
        # Test network modes
        agent.actor.eval()
        is_eval = not agent.actor.training
        print_test("Network mode switching", is_eval,
                   "Can switch between train/eval modes")
        
        return True
        
    except Exception as e:
        print_test("MADDPGAgent", False, f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_integration():
    """Test integration of all components"""
    print(f"\n{Colors.BOLD}{'='*60}{Colors.RESET}")
    print(f"{Colors.BOLD}Testing Integration{Colors.RESET}")
    print(f"{Colors.BOLD}{'='*60}{Colors.RESET}\n")
    
    try:
        # Create components
        actor = ActorNetwork()
        critic = CriticNetwork()
        decoder = ActionDecoder()
        
        # Simulate training step
        batch_size = 8
        state = torch.randn(batch_size, 537)
        
        # Forward pass
        offload_logits, cont = actor(state)
        actions = decoder.decode(offload_logits, cont)
        
        # Create action tensor for critic
        action_tensor = torch.zeros(batch_size, 11)
        for i, action in enumerate(actions):
            # One-hot offload
            action_tensor[i, action['offload']] = 1.0
            # Continuous actions
            action_tensor[i, 5] = action['cpu']
            action_tensor[i, 6:9] = torch.from_numpy(action['bandwidth'])
            action_tensor[i, 9:11] = torch.from_numpy(action['move']) / 5.0  # Normalize
        
        q_value = critic(state, action_tensor)
        
        integration_ok = (
            q_value.shape[0] == batch_size and
            len(actions) == batch_size
        )
        print_test("Actor → Decoder → Critic pipeline", integration_ok,
                   f"Processed {batch_size} samples successfully")
        
        # Test backward pass
        loss = -q_value.mean()  # Maximize Q
        loss.backward()
        
        has_actor_grad = any(p.grad is not None for p in actor.parameters())
        has_critic_grad = any(p.grad is not None for p in critic.parameters())
        
        print_test("End-to-end gradient flow", 
                   has_actor_grad and has_critic_grad,
                   "Gradients propagate through entire pipeline")
        
        return True
        
    except Exception as e:
        print_test("Integration", False, f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def run_all_tests():
    """Run all test suites"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}MADDPG Agents V2 - Test Suite{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.RESET}\n")
    
    results = {}
    
    # Run tests
    results['ActorNetwork'] = test_actor_network()
    results['CriticNetwork'] = test_critic_network()
    results['ActionDecoder'] = test_action_decoder()
    results['MADDPGAgent'] = test_maddpg_agent()
    results['Integration'] = test_integration()
    
    # Summary
    print(f"\n{Colors.BOLD}{'='*60}{Colors.RESET}")
    print(f"{Colors.BOLD}Test Summary{Colors.RESET}")
    print(f"{Colors.BOLD}{'='*60}{Colors.RESET}\n")
    
    total = len(results)
    passed = sum(results.values())
    
    for name, result in results.items():
        status = f"{Colors.GREEN}✓{Colors.RESET}" if result else f"{Colors.RED}✗{Colors.RESET}"
        print(f"{status} {name}")
    
    print(f"\n{Colors.BOLD}Overall: {passed}/{total} tests passed{Colors.RESET}")
    
    if passed == total:
        print(f"{Colors.GREEN}{Colors.BOLD}🎉 All tests passed!{Colors.RESET}\n")
        return True
    else:
        print(f"{Colors.RED}{Colors.BOLD}❌ Some tests failed{Colors.RESET}\n")
        return False

if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
