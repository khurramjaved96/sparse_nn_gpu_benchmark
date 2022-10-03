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
#include "../include/random_memory.cuh"
#include <random>
#include <fstream>

Parameter::Parameter(float step_size, float weight, float meta_step_size, Feature *to, Feature *from) {
  this->weight = weight;
  this->step_size = step_size;
  this->meta_step_size = meta_step_size;
  this->to = to;
  this->from = from;
}

void Parameter::construct(Feature *from, Feature *to) {
  this->weight = 0.5;
  this->step_size = 1e-6;
  this->meta_step_size = 3e-3;
  this->to = to;
  this->synapse_local_utility_trace = 0;
  this->utility_to_distribute = 0;
  this->from = from;
  this->to->add_incoming_parameter(this);
}

float Feature::forward(float pre_activation) {
  return pre_activation;
}

void Parameter::update_utility() {
  float diff = this->to->value - this->to->forward(
      this->to->temp_value - this->from->old_value * this->weight);
  if (diff < 0)
    diff *= -1;

  this->synapse_local_utility_trace = 0.99999 * this->synapse_local_utility_trace + 0.00001 * diff;
  this->utility_to_distribute = (synapse_local_utility_trace * this->to->feature_utility);
}
Feature::Feature() {
//  this->value = 0;
//  this->trace = 0;
//  this->total_features = 0;
//  cudaMallocManaged(&this->incoming_parameters, sizeof(Parameter *) * 4000);
}

void Feature::construct() {
  this->value = 0;
  this->trace = 0;
  this->is_output_feature = false;
  this->is_input_feature = false;
  this->age = 0;
  this->sum_of_utility_traces = 0;
  this->total_features = 0;
  this->total_outgoing_features = 0;
  cudaMallocManaged(&this->incoming_parameters, sizeof(Parameter *) * 4000);
  cudaMallocManaged(&this->outgoing_parameters, sizeof(Parameter *) * 4000);
}

void Feature::add_incoming_parameter(Parameter *p) {
  if (total_features < 4000) {
    incoming_parameters[total_features] = p;
    total_features++;
  } else {
    printf("Adding more features than space\n");
  }
}

void Feature::add_outgoing_parameter(Parameter *p) {
  if (total_outgoing_features < 4000) {
    outgoing_parameters[total_outgoing_features] = p;
    total_outgoing_features++;
  } else {
    printf("Adding more outgoing features than space\n");
  }
}

void Feature::update_value() {
  temp_value = 0;
  for (int i = 0; i < total_features; i++) {
    temp_value += incoming_parameters[i]->weight * incoming_parameters[i]->from->value;
  }
}

void Feature::update_utility() {

  this->feature_utility = 0;
  if (this->is_output_feature)
    this->feature_utility = 1;
  else {
    for (int counter = 0; counter < this->total_outgoing_features; counter++) {
      this->feature_utility += this->outgoing_parameters[counter]->utility_to_distribute;
    }
  }

  this->sum_of_utility_traces = 0;
  for (int counter = 0; counter < this->total_features; counter++) {
    this->sum_of_utility_traces += this->incoming_parameters[counter]->synapse_local_utility_trace;
  }

  for (int counter = 0; counter < this->total_features; counter++) {
    this->incoming_parameters[counter]->update_utility();
  }

}
void Feature::fire() {
  value = temp_value;
}

void Feature::set_feature() {
  this->value = 1;
}

void Feature::unset_feature() {
  this->value = 0;
}

__global__ void set_feature_kernel(Feature *data, int n) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride) {
    data[i].set_feature();
  }
}

__global__ void update_value_kernel(Feature *data, int n) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride) {
    data[i].update_value();
  }
}

__global__ void fire_kernel(Feature *data, int n) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride) {
    data[i].fire();
  }
}

__global__ void neuron_utility_update_kernel(Feature *data, int n) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride) {
    data[i].update_utility();
  }
}