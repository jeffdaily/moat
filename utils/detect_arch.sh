#!/usr/bin/env bash
# Detect this host's AMD GPU arch. Emits KEY=VALUE lines (GFX_ARCH, GFX_TRIPLE,
# PLATFORM) to stdout for `eval`. Exits nonzero if no AMD GPU. Never hardcodes
# the arch; arch varies per container. Set MOAT_OS=windows on a Windows host.
set -uo pipefail

arch=""
if command -v rocm_agent_enumerator >/dev/null 2>&1; then
  arch=$(rocm_agent_enumerator 2>/dev/null \
         | grep -E '^gfx[0-9a-f]+' | grep -v '^gfx000$' | sort -u | head -1)
fi
if [ -z "$arch" ] && command -v rocminfo >/dev/null 2>&1; then
  arch=$(rocminfo 2>/dev/null | grep -oE 'gfx[0-9a-f]+' | grep -v '^gfx000$' | sort -u | head -1)
fi
if [ -z "$arch" ]; then
  echo "detect_arch: no AMD GPU found (rocm_agent_enumerator/rocminfo)" >&2
  exit 1
fi

distinct=$(rocm_agent_enumerator 2>/dev/null | grep -E '^gfx[0-9a-f]+' | grep -v '^gfx000$' | sort -u | wc -l)
if [ "${distinct:-1}" -gt 1 ]; then
  echo "detect_arch: multiple GPU archs present; using $arch (set HIP_VISIBLE_DEVICES to pin)" >&2
fi

triple=""
if command -v rocminfo >/dev/null 2>&1; then
  triple=$(rocminfo 2>/dev/null | grep -oE 'amdgcn-amd-amdhsa--gfx[0-9a-f:+-]+' | head -1)
fi

os="${MOAT_OS:-linux}"
echo "GFX_ARCH=${arch}"
echo "GFX_TRIPLE=${triple:-${arch}}"
echo "PLATFORM=${os}-${arch}"
