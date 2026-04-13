# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Device abstraction utilities for multi-platform (CUDA/XPU) test support.

The dynamo test harness needs to launch vllm worker subprocesses with the
correct device visibility environment variable. On NVIDIA this is
CUDA_VISIBLE_DEVICES; on Intel XPU it is ZE_AFFINITY_MASK. This module
provides a single source of truth for device detection and env var selection.

Detection priority:
1. VLLM_TARGET_DEVICE env var (set by the XPU container image / CI workflow)
2. /dev/dri presence (Intel GPU indicator)
3. Default to "cuda"
"""

import os
import shutil
from typing import Literal

DeviceType = Literal["cuda", "xpu"]

# Map from device type to the environment variable that controls device
# visibility. These match vllm's platform definitions:
#   vllm/platforms/cuda.py:  device_control_env_var = "CUDA_VISIBLE_DEVICES"
#   vllm/platforms/xpu.py:   device_control_env_var = "ZE_AFFINITY_MASK"
_DEVICE_ENV_VAR = {
    "cuda": "CUDA_VISIBLE_DEVICES",
    "xpu": "ZE_AFFINITY_MASK",
}


def get_device_type() -> DeviceType:
    """Detect the accelerator device type for this environment.

    Returns:
        "cuda" or "xpu"
    """
    # Explicit override (set by CI or user)
    target = os.environ.get("VLLM_TARGET_DEVICE", "").lower()
    if target in ("cuda", "xpu"):
        return target

    # Auto-detect: check for Intel GPU render devices
    if os.path.exists("/dev/dri/renderD128"):
        # Confirm it's Intel via sysfs
        for card in sorted(os.listdir("/sys/class/drm/")):
            vendor_path = f"/sys/class/drm/{card}/device/vendor"
            try:
                with open(vendor_path) as f:
                    if f.read().strip() == "0x8086":
                        return "xpu"
            except (FileNotFoundError, PermissionError):
                continue

    return "cuda"


def get_device_env_var(device_type: DeviceType | None = None) -> str:
    """Return the device visibility environment variable name.

    Args:
        device_type: "cuda" or "xpu". Auto-detected if None.

    Returns:
        "CUDA_VISIBLE_DEVICES" for CUDA, "ZE_AFFINITY_MASK" for XPU.
    """
    if device_type is None:
        device_type = get_device_type()
    return _DEVICE_ENV_VAR.get(device_type, "CUDA_VISIBLE_DEVICES")


def get_vllm_extra_args(device_type: DeviceType | None = None) -> list[str]:
    """Return additional vllm CLI arguments needed for this device type.

    On XPU, vllm workers need --connector none because the default
    NixlConnector assumes a CUDA kv_buffer device. This will be
    removed once NixlConnector supports XPU.

    Args:
        device_type: "cuda" or "xpu". Auto-detected if None.

    Returns:
        List of extra CLI args (may be empty for CUDA).
    """
    if device_type is None:
        device_type = get_device_type()
    if device_type == "xpu":
        return ["--connector", "none"]
    return []


def get_vllm_extra_env(device_type: DeviceType | None = None) -> dict[str, str]:
    """Return additional environment variables needed for vllm workers.

    On XPU, workers need VLLM_TARGET_DEVICE=xpu and (on consumer Arc
    GPUs) UR_ADAPTERS_FORCE_LOAD for the V1 Level Zero adapter.

    Args:
        device_type: "cuda" or "xpu". Auto-detected if None.

    Returns:
        Dict of extra env vars (may be empty for CUDA).
    """
    if device_type is None:
        device_type = get_device_type()
    env = {}
    if device_type == "xpu":
        env["VLLM_TARGET_DEVICE"] = "xpu"
        # Propagate UR adapter override if set in parent environment
        ur_force = os.environ.get("UR_ADAPTERS_FORCE_LOAD")
        if ur_force:
            env["UR_ADAPTERS_FORCE_LOAD"] = ur_force
    return env


def is_xpu() -> bool:
    """Convenience check: are we running on Intel XPU?"""
    return get_device_type() == "xpu"
