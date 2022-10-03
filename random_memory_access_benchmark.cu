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
#include "include/random_memory.cuh"

#include <random>
#include <fstream>


int main(int argc, char *argv[]) {

  std::vector<std::vector<std::string>> error_logger;

  std::vector<int> threads{256};
  std::vector<int> blocks{32};

  using std::chrono::high_resolution_clock;
  using std::chrono::duration_cast;
  using std::chrono::duration;
  using std::chrono::milliseconds;

  int total_features = 10000;
  int parameters_per_feature = 100;
  Parameter *parameter_pointer;
  Feature *features_pointer;

  std::vector<std::string> error;

  cudaMallocManaged(&parameter_pointer, sizeof(Parameter) * total_features * parameters_per_feature);

  std::cout << "Parameter memory allocated\n";

  cudaMallocManaged(&features_pointer, sizeof(Feature) * total_features);

  std::cout << "Feature memory allocated\n";

  std::cout << "Constructing features\n";
  for (int i = 0; i < total_features; i++) {
    if (i % 1000000 == 0)
      std::cout << "#";
    features_pointer[i].construct();
  }
  std::cout << "\n";
  std::mt19937 mt(0);
  std::uniform_int_distribution<int> dist(0, total_features - 1);
  for (int i = 0; i < total_features * parameters_per_feature; i++) {
    if (i % 1000000 == 0)
      std::cout << "#";
    parameter_pointer[i].construct(&features_pointer[dist(mt)], &features_pointer[dist(mt)]);
  }
  std::cout << "\n";

  std::cout << "Running experiment\n";
  for (auto block: blocks) {
    for (auto thread: threads) {

      for (int temp = 0; temp < 10; temp++) {
        set_feature_kernel<<<block, thread>>>(features_pointer, total_features);
        cudaDeviceSynchronize();

        update_value_kernel<<<block, thread>>>(features_pointer, total_features);
        cudaDeviceSynchronize();

        fire_kernel<<<block, thread>>>(features_pointer, total_features);
        cudaDeviceSynchronize();

        neuron_utility_update_kernel<<<block, thread>>>(features_pointer, total_features);
        cudaDeviceSynchronize();

      }
      std::cout << "Memory moved to GPU\n";
      auto t1 = high_resolution_clock::now();
      int total_steps = 500;
      for (int i = 0; i < total_steps; i++) {
        set_feature_kernel<<<block, thread>>>(features_pointer, 3000);
        cudaDeviceSynchronize();

        update_value_kernel<<<block, thread>>>(features_pointer, total_features);
        cudaDeviceSynchronize();

        fire_kernel<<<block, thread>>>(features_pointer, total_features);
        cudaDeviceSynchronize();

        neuron_utility_update_kernel<<<block, thread>>>(features_pointer, total_features);
        cudaDeviceSynchronize();

      }
      auto t2 = high_resolution_clock::now();
      duration<double, std::milli> ms_double = t2 - t1;
      std::cout << "Blocks\tThreads\tTime\n";
      std::cout << block << "\t" << thread << "\t" << 1000.0 / (ms_double.count() / (total_steps)) << "fps\n";

    }
  }
}