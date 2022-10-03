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
#include "../include/sequential_memory.cuh"

__host__ __device__
Parameter::Parameter(float step_size, float weight, float meta_step_size) {
  this->weight = weight;
  this->step_size = step_size;
  this->meta_step_size = meta_step_size;
}
Parameter::Parameter() {
  this->weight = 0;
  this->step_size = 1e-6;
  this->meta_step_size = 3e-3;
}

__host__ __device__
Feature::Feature() {
  this->value = 1;
  this->trace = 0;
  cudaMalloc(&this->incoming_parameters, sizeof(Parameter * ) * 10);
}

__host__ __device__
void Feature::set_feature() {
  this->value = 1;
}
__host__ __device__
void Feature::unset_feature() {
  this->value = 0;
}

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
    y[i].trace = y[i].trace * 0.9 + 0.1;
  }
}