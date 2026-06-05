# brian2cuda notes

## HIP/ROCm Backend Implementation

This is a **runtime code generator** project (not Strategy A or B). The port adds HIP/ROCm support by:

1. Adding a CUDA-to-HIP compatibility header (`brianlib/cuda_to_hip.h`) that maps CUDA symbols to HIP equivalents at compile time
2. Modifying the device to detect HIP backend and generate HIP-compatible makefiles
3. Using `-fgpu-rdc` flag for relocatable device code (required for cross-TU device symbol linking)

### Build Instructions (linux-gfx90a)

```bash
# Install brian2cuda
cd projects/brian2cuda/src
pip install -e .

# Run with HIP backend (auto-detected when ROCm present and CUDA absent, or set explicitly)
export USE_HIP=1
export HIP_VISIBLE_DEVICES=0  # or 1 for second GPU

# Example test
python -c "
from brian2 import *
import brian2cuda
set_device('cuda_standalone', build_on_run=False)
G = NeuronGroup(100, 'dv/dt = -v/(10*ms) : 1', method='linear')
G.v = 'i / 100.'
run(1*ms)
device.build(directory='/tmp/brian2cuda_test', compile=True, run=True)
"
```

### Validated

- Platform: linux-gfx90a (MI250X)
- ROCm: 7.2.1
- Simple neuron group simulation matches expected values
