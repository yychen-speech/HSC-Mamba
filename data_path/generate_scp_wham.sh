#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${DATA_ROOT:-/Featurize/data}"
DATASET_ROOT="${DATA_ROOT}/wham/wav8k/min"
OUT_DIR="${SCP_OUT:-${SCRIPT_DIR}/scp_ss_8k_wham}"
SNR_STATS_OUT="${WHAM_SNR_STATS_OUT:-${SCRIPT_DIR}/../hsc_mamba/data/statistics/wham_train_snr_db.csv}"

[[ -d "$DATASET_ROOT" ]] || {
    echo "[ERROR] Missing WHAM directory: $DATASET_ROOT" >&2
    echo "        Put the dataset under /Featurize/data or set DATA_ROOT explicitly." >&2
    exit 1
}

write_scp() {
    local source_dir="$1"
    local output_file="$2"

    [[ -d "$source_dir" ]] || {
        echo "[ERROR] Missing directory: $source_dir" >&2
        exit 1
    }

    find "$source_dir" -maxdepth 1 -type f -name '*.wav' \
        -printf '%f %p\n' | LC_ALL=C sort > "${output_file}.tmp"
    [[ -s "${output_file}.tmp" ]] || {
        echo "[ERROR] No wav files found in: $source_dir" >&2
        exit 1
    }
    mv "${output_file}.tmp" "$output_file"
}

validate_split() {
    local split="$1"
    local expected="$2"
    shift 2
    local reference="$1"
    local manifest count

    for manifest in "$@"; do
        count="$(wc -l < "$manifest")"
        [[ "$count" -eq "$expected" ]] || {
            echo "[ERROR] $manifest has $count entries; expected $expected." >&2
            exit 1
        }
        while read -r key path; do
            [[ -n "$key" && -f "$path" ]] || {
                echo "[ERROR] Invalid SCP entry in $manifest: $key $path" >&2
                exit 1
            }
        done < "$manifest"
    done

    for manifest in "$@"; do
        cmp -s <(cut -d' ' -f1 "$reference") <(cut -d' ' -f1 "$manifest") || {
            echo "[ERROR] File keys are not aligned for split $split: $manifest" >&2
            exit 1
        }
    done
    echo "[OK] $split: $expected aligned utterances"
}

mkdir -p "$OUT_DIR"

for split in tr cv tt; do
    write_scp "$DATASET_ROOT/$split/mix_both" "$OUT_DIR/${split}_mix.scp"
    write_scp "$DATASET_ROOT/$split/s1" "$OUT_DIR/${split}_s1.scp"
    write_scp "$DATASET_ROOT/$split/s2" "$OUT_DIR/${split}_s2.scp"
done
write_scp "$DATASET_ROOT/tr/noise" "$OUT_DIR/tr_n.scp"

validate_split tr 20000 \
    "$OUT_DIR/tr_mix.scp" "$OUT_DIR/tr_s1.scp" \
    "$OUT_DIR/tr_s2.scp" "$OUT_DIR/tr_n.scp"
validate_split cv 5000 \
    "$OUT_DIR/cv_mix.scp" "$OUT_DIR/cv_s1.scp" "$OUT_DIR/cv_s2.scp"
validate_split tt 3000 \
    "$OUT_DIR/tt_mix.scp" "$OUT_DIR/tt_s1.scp" "$OUT_DIR/tt_s2.scp"

if [[ ! -s "$SNR_STATS_OUT" || "${FORCE_SNR_STATS:-0}" == "1" ]]; then
    python "$SCRIPT_DIR/generate_noise_snr_stats.py" \
        --dataset-root "$DATASET_ROOT" \
        --output "$SNR_STATS_OUT" \
        --speech-subdir mix_clean \
        --dataset-name WHAM
fi

echo "[DONE] WHAM SCP files: $OUT_DIR"
