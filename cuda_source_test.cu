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


class Parameter{
 public:
  float weight;
  float step_size;
  float meta_step_size;


  __host__ __device__
  Parameter(float step_size, float weight, float meta_step_size){
    this->weight = weight;
    this->step_size = step_size;
    this->meta_step_size = meta_step_size;
  }
  Parameter(){
    this->weight = 0;
    this->step_size = 1e-6;
    this->meta_step_size = 3e-3;
  }
};

class Feature{
 public:
  float value;
  float trace;
  Parameter* incoming_parameters;
  __host__ __device__
  Feature(){
    this->value = 1;
    this->trace = 0;
    cudaMalloc(&this->incoming_parameters, sizeof(Parameter*)*10);
  }

  __host__ __device__
  void set_feature(){
    this->value = 1;
  }
  __host__ __device__
  void unset_feature(){
    this->value = 0;
  }
};




__global__ void setvalues(Parameter *data, float step_size, float weight, float meta_step_size, int n) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride) {
    data[i].weight = weight;
    data[i].step_size = step_size;
    data[i].meta_step_size = meta_step_size;
  }
}


__global__ void setfeatures(Feature *data, float value, int n) {
//  printf("test\n");
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride) {
    data[i].value = value;
  }
}


__global__ void printkernel(Parameter *data, int total_elem) {
  for (int i = 0; i < total_elem; i++) {
    printf("%f\n", data[i].weight);
  }
}

__global__
void update_value(int n, Parameter *x, Feature *y, float *result) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride) {
    result[i] = x[i].weight * y[i].value;
    y[i].trace = y[i].trace*0.9 + 0.1;
  }
}

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
//      std::cout << "Step = " << a << std::endl;
        }
        for(int gpu = 0; gpu < total_gpus; gpu++) {
          cudaSetDevice(gpu);
          update_value<<<block, t_no>>>(total_elem, parameter_pointer, features_pointer, results);
          thrust::device_ptr<float> cptr = thrust::device_pointer_cast(results);
//          float prediction = thrust::reduce(cptr, cptr + total_elem, (float) 0, thrust::plus<float>());
//          total_prediction += prediction;
//          std::cout << "Prediction = " << prediction << std::endl;
        }
//
      }
      cudaDeviceSynchronize();
      auto t2 = high_resolution_clock::now();
      duration<double, std::milli> ms_double = t2 - t1;
      std::cout << block << "\t" << t_no << "\t" << ms_double.count() / steps << "ms\n";
    }
  }
  exit(1);

}