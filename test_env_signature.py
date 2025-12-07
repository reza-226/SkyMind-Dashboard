# test_env_signature.py
from environments.uav_mec_env import UAVMECEnvironment
import inspect

# نمایش signature
sig = inspect.signature(UAVMECEnvironment.__init__)
print("=" * 80)
print("UAVMECEnvironment.__init__ signature:")
print("=" * 80)
print(sig)
print()

# نمایش پارامترها
print("Parameters:")
for param_name, param in sig.parameters.items():
    if param_name != 'self':
        default = f" = {param.default}" if param.default != inspect.Parameter.empty else ""
        print(f"  - {param_name}{default}")
