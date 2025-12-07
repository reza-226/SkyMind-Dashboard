# check_env_signature.py
import inspect
from environments.uav_mec_env import UAVMECEnvironment

# نمایش signature
sig = inspect.signature(UAVMECEnvironment.__init__)
print("✅ UAVMECEnvironment.__init__ parameters:")
print(sig)

# نمایش docstring
if UAVMECEnvironment.__init__.__doc__:
    print("\n📖 Docstring:")
    print(UAVMECEnvironment.__init__.__doc__)
