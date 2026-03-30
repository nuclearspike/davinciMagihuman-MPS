# Copyright (c) 2026 SandAI. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import torch


def is_hopper_arch():
    if not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_capability()[0] == 9


def get_arch_memory(unit: str = "GB"):
    if torch.cuda.is_available():
        total_bytes = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory
    elif torch.backends.mps.is_available():
        # Apple Silicon unified memory — use os.sysconf for total system RAM
        total_bytes = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    else:
        return 0
    if unit == "B":
        return float(total_bytes)
    elif unit == "KB":
        return total_bytes / 1024
    elif unit == "MB":
        return total_bytes / 1024 / 1024
    elif unit == "GB":
        return total_bytes / 1024 / 1024 / 1024
    else:
        raise ValueError(f"Invalid unit: {unit}")
