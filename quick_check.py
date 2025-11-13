# quick_check.py
import sys
sys.path.append('.')

print("Checking core.env_multi...")
try:
    import core.env_multi as em
    classes = [name for name in dir(em) if not name.startswith('_') and isinstance(getattr(em, name), type)]
    print(f"  Classes: {classes}")
except Exception as e:
    print(f"  Error: {e}")

print("\nChecking core.env...")
try:
    import core.env as e
    classes = [name for name in dir(e) if not name.startswith('_') and isinstance(getattr(e, name), type)]
    print(f"  Classes: {classes}")
except Exception as e:
    print(f"  Error: {e}")

print("\nChecking agents.agent_maddpg_multi...")
try:
    import agents.agent_maddpg_multi as am
    classes = [name for name in dir(am) if not name.startswith('_') and isinstance(getattr(am, name), type)]
    print(f"  Classes: {classes}")
except Exception as e:
    print(f"  Error: {e}")
