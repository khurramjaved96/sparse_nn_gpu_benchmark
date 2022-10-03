#include <stdio.h>
#include <iostream>
#include <curand.h>
#include <curand_kernel.h>
#include <math.h>
#include <chrono>
#include <vector>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/sequence.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/replace.h>
#include <thrust/functional.h>
#include <iostream>
#include <thrust/execution_policy.h>
#include "include/sequential_memory.cuh"

int main() {
//
  std::vector<int> threads{64, 128, 256, 512, 1024};
  std::vector<int> blocks{ 60, 80, 100, 120};

  using std::chrono::high_resolution_clock;
  using std::chrono::duration_cast;
  using std::chrono::duration;
  using std::chrono::milliseconds;


  int total_gpus = 1;
  int total_elem = 10000000/total_gpus;

  Parameter* parameter_pointer;
  Feature* features_pointer;
  float* results;

  cudaMalloc(&results, sizeof(float) * total_elem);
  cudaMalloc(&parameter_pointer, sizeof(Parameter) * total_elem);
  cudaMalloc(&features_pointer, sizeof(Feature) * total_elem);

  setvalues<<<blocks[0], threads[0]>>>(parameter_pointer, 1e-3, 0.1, 3e-4, total_elem);
  setfeatures<<<blocks[0], threads[0]>>>(features_pointer, 1, total_elem);
  std::cout << "Blocks\tThreads\tTime\n";
  float total_prediction;
  for (auto block : blocks) {
    for (auto t_no : threads) {
      auto t1 = high_resolution_clock::now();
      int steps = 10000;
      for (int a = 0; a < steps; a++) {
        if (a % 10000 == 0) {
        }
        for(int gpu = 0; gpu < total_gpus; gpu++) {
          cudaSetDevice(gpu);
          update_value<<<block, t_no>>>(total_elem, parameter_pointer, features_pointer, results);
          thrust::device_ptr<float> cptr = thrust::device_pointer_cast(results);
        }

      }
      cudaDeviceSynchronize();
      auto t2 = high_resolution_clock::now();
      duration<double, std::milli> ms_double = t2 - t1;
      std::cout << block << "\t" << t_no << "\t" << ms_double.count() / steps << "ms\n";
    }
  }
  exit(1);

}