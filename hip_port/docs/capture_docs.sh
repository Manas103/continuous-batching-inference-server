#!/bin/bash
set -u
cd "/mnt/c/Users/Manas/Downloads/Auto Job Applier/Total Job Application process flow/projects/continuous-batching-inference-server/hip_port"

{
  echo "hipify-clang version:"
  /opt/rocm/bin/hipify-clang --version
  echo
  for f in attention elementwise linear model_weights paged_kv_cache transformer correctness_check; do
    src="../kernels/${f}.cu"
    [ -f "$src" ] || src="../src/${f}.cpp"
    echo "=== hipify-clang ${src} ==="
    /opt/rocm/bin/hipify-clang "$src" -o "/tmp/${f}.hipify_check.hip" --print-stats --cuda-path=/usr/local/cuda-12.3 -- -I ../include -std=c++17 2>&1
    echo
  done
} > docs/hipify_raw_output.txt 2>&1

rm -rf build_obj correctness_check
make check > docs/build_output.txt 2>&1
echo "build exit: $?" >> docs/build_output.txt

./correctness_check ../python/weights.bin ../python/reference.bin > docs/correctness_check_output.txt 2>&1
echo "run exit: $?" >> docs/correctness_check_output.txt

make gfx942 > docs/gfx942_compile_output.txt 2>&1
echo "gfx942 compile-only exit: $?" >> docs/gfx942_compile_output.txt

echo done
