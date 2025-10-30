export MKL_THREADING_LAYER=GNU
apptainer exec --nv rf_diffusion/exec/bakerlab_rf_diffusion_aa.sif \
  rf_diffusion/benchmark/pipeline.py \
  --config-name=open_source_demo \
  sweep.benchmarks=$(seq -f tri_sweep_%02g -s, 0 19) \
  outdir=./triad_sweep20
  --multirun
