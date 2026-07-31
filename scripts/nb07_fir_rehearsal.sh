#!/bin/bash
# Execute notebooks/07_real_checkpoints_colab.ipynb on a real GPU, as a rehearsal for the
# Colab run. Deliberately builds a FRESH venv rather than reusing ~/nnunet_env_fir, so that
# the notebook's trainer-shim cell is exercised the way it will be on Colab, and so nothing
# can touch the lab env (whose lr_schedule trainers are symlinks into the deployed repo).
#
# Everything lives on $SCRATCH or node-local $SLURM_TMPDIR; $HOME is ~88% full.
#
#   sbatch scripts/nb07_fir_rehearsal.sh
#
#SBATCH --account=rrg-jma_gpu
#SBATCH --gpus-per-node=nvidia_h100_80gb_hbm3_2g.20gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1:30:00
#SBATCH --job-name=nb07_rehearsal
#SBATCH --output=/scratch/kcuoft/nb07_run/%x-%j.out

set -uo pipefail

WORK=/scratch/kcuoft/nb07_run
mkdir -p "$WORK"

echo "[nb07] host=$(hostname) job=${SLURM_JOB_ID} start=$(date -Is)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# --- keep every cache off $HOME ------------------------------------------------------------
export HF_HOME=/scratch/kcuoft/hf_home
export HF_XET_CACHE=/scratch/kcuoft/hf_xet_cache
export PIP_CACHE_DIR=/scratch/kcuoft/pip_cache
export MPLCONFIGDIR=/scratch/kcuoft/mpl_cache
mkdir -p "$HF_HOME" "$HF_XET_CACHE" "$PIP_CACHE_DIR" "$MPLCONFIGDIR"

cd "$SLURM_TMPDIR" || exit 1

# --- fresh venv ----------------------------------------------------------------------------
module load python/3.11
virtualenv --no-download venv || exit 1
source venv/bin/activate
pip install --no-index --upgrade pip setuptools wheel

# Prefer the Alliance wheelhouse; fall back to PyPI (Fir permits outbound) if the pinned
# version is not mirrored. nnU-Net 2.6.4 is what these checkpoints were trained with.
echo "[nb07] installing nnunetv2==2.6.4"
if ! pip install --no-index nnunetv2==2.6.4; then
    echo "[nb07] wheelhouse miss -> PyPI"
    pip install nnunetv2==2.6.4 || exit 1
fi
pip install --no-index nbconvert nbformat ipykernel matplotlib nibabel || exit 1
pip install --no-index huggingface_hub || pip install huggingface_hub || exit 1

python -c "import nnunetv2, torch; print('[nb07] nnunetv2', nnunetv2.__version__ if hasattr(nnunetv2,'__version__') else 'n/a', '| torch', torch.__version__, '| cuda', torch.cuda.is_available())"

# course_utils.dsc / .viz, staged by the caller so the notebook's import succeeds without
# needing to clone the (still unpushed) course repo.
export PYTHONPATH="$WORK/course_repo:${PYTHONPATH:-}"

# --- sample GPU memory while the notebook runs ---------------------------------------------
# Query per-process usage, NOT --query-gpu=memory.used: the first run of this script was
# handed a full 80 GB H100 rather than the requested MIG slice, and device-wide memory.used
# included other tenants (a 27 GB "idle floor"), making the T4 verdict meaningless.
( while true; do
      nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits
      sleep 5
  done ) > "$WORK/gpu_mem.csv" 2>&1 &
SAMPLER=$!

# --- execute ------------------------------------------------------------------------------
cp "$WORK/07_real_checkpoints_colab.ipynb" ./nb07.ipynb
# --allow-errors: we want the executed notebook (with tracebacks) back either way.
python -m nbconvert --to notebook --execute --inplace nb07.ipynb \
    --ExecutePreprocessor.timeout=4800 --allow-errors
RC=$?
cp ./nb07.ipynb "$WORK/07_executed.ipynb"

kill "$SAMPLER" 2>/dev/null

# --- report -------------------------------------------------------------------------------
python - <<'PY'
import json, pathlib
p = pathlib.Path('/scratch/kcuoft/nb07_run/gpu_mem.csv')
if p.exists():
    # sum concurrent processes per sample, then take the peak across samples
    peaks, cur = [], 0
    for line in p.read_text().splitlines():
        parts = [x.strip() for x in line.split(',')]
        if len(parts) == 2 and parts[1].isdigit():
            cur += int(parts[1])
        else:
            peaks.append(cur); cur = 0
    peaks.append(cur)
    peak = max(peaks) if peaks else 0
    if peak:
        print(f"[nb07] peak GPU memory across this job's own processes: {peak} MiB")
        print(f"[nb07] a 16 GB T4 has 16384 MiB -> {'FITS with headroom' if peak < 14000 else 'TIGHT OR TOO SMALL'}")
    else:
        print("[nb07] no per-process GPU samples captured; T4 headroom still unmeasured")

nb = json.loads(pathlib.Path('/scratch/kcuoft/nb07_run/07_executed.ipynb').read_text(encoding='utf-8'))
fails = 0
for i, c in enumerate(nb['cells']):
    for o in c.get('outputs', []):
        if o.get('output_type') == 'error':
            fails += 1
            print(f"\n[nb07] CELL {i} FAILED: {o.get('ename')}: {o.get('evalue')}")
            print('\n'.join(o.get('traceback', [])[-12:]))
print(f"\n[nb07] cells with errors: {fails}")
PY

echo "[nb07] nbconvert rc=$RC end=$(date -Is)"
echo "[nb07] executed notebook: $WORK/07_executed.ipynb"
