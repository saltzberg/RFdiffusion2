export MKL_THREADING_LAYER=GNU
apptainer exec --nv rf_diffusion/exec/bakerlab_rf_diffusion_aa.sif \
  rf_diffusion/benchmark/pipeline.py \
  --config-name=open_source_demo \
  sweep.benchmarks=\'8jx4_native,8jx4_weird\' \
  outdir=./8jx4_2
