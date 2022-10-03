//
// Created by Khurram Javed on 2022-10-03.
//

#ifndef INCLUDE_SEQUENTIAL_MEMORY_H_
#define INCLUDE_SEQUENTIAL_MEMORY_H_

class Parameter {
 public:
  float weight;
  float step_size;
  float meta_step_size;

  __host__ __device__
  Parameter(float step_size, float weight, float meta_step_size);
  Parameter();
};

class Feature {
 public:
  float value;
  float trace;
  Parameter *incoming_parameters;
  __host__ __device__
  Feature();

  __host__ __device__
  void set_feature();
  __host__ __device__
  void unset_feature();
};

__global__ void setvalues(Parameter *data, float step_size, float weight, float meta_step_size, int n);

__global__ void setfeatures(Feature *data, float value, int n);

__global__ void printkernel(Parameter *data, int total_elem);
__global__
void update_value(int n, Parameter *x, Feature *y, float *result);

#endif //INCLUDE_SEQUENTIAL_MEMORY_H_
