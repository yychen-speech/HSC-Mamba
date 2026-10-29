#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

for script in \
    generate_scp_wsj0_2mix.sh \
    generate_scp_libri2mix.sh \
    generate_scp_wham.sh \
    generate_scp_whamr.sh; do
    echo "[RUN] $script"
    bash "${SCRIPT_DIR}/${script}"
done

echo "[DONE] All SCP files were generated under: $SCRIPT_DIR"
