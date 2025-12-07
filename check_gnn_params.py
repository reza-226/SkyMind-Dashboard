# check_gnn_params.py
"""بررسی signature واقعی GNNTaskEncoder"""
import inspect
from models.gnn.task_encoder import GNNTaskEncoder

# نمایش signature
sig = inspect.signature(GNNTaskEncoder.__init__)
print("🔍 GNNTaskEncoder.__init__ parameters:")
print("=" * 60)
for param_name, param in sig.parameters.items():
    if param_name != 'self':
        default = param.default if param.default != inspect.Parameter.empty else "❌ REQUIRED"
        annotation = param.annotation if param.annotation != inspect.Parameter.empty else "Any"
        print(f"  ✓ {param_name}: {annotation} = {default}")
print("=" * 60)
