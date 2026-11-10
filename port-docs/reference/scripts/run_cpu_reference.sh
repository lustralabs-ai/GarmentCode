#!/bin/bash
# Usage: run_cpu_reference.sh <venv> <garmentcode_dir> <data_root> [device] [extra harness args...]
# Runs every garment in data_root/inputs twice (run1, run2) in parallel; writes data_root/<garment>/<backend>_<run>/ and a done marker.
VENV=$1; GC=$2; DATA=$3; DEV=${4:-cpu}; shift 4
BACKEND=${DEV%%:*}; [ "$BACKEND" = cuda ] && BACKEND=cuda
HERE=$(cd $(dirname $0); pwd)
source $VENV/bin/activate
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/mpl-$USER}
for g in $(ls $DATA/inputs); do for r in run1 run2; do
  out=$DATA/$g/${BACKEND}_$r
  mkdir -p $out
  ( python $HERE/dump_harness.py run --garmentcode-dir $GC --inputs $DATA/inputs/$g --device $DEV --out $out "$@" > $out.log 2>&1; touch $out.done ) &
done; done
wait
