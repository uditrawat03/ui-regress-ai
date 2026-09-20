import json
import platform

import torch

payload = {
    "python": platform.python_version(),
    "platform": platform.platform(),
    "torch": torch.__version__,
    "cuda_runtime": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
}

if torch.cuda.is_available() and torch.cuda.device_count():
    payload["cuda_device_name"] = torch.cuda.get_device_name(0)

print(json.dumps(payload, indent=2))
