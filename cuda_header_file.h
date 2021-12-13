//
// Created by Khurram Javed on 2021-10-23.
//

#ifndef _CUDA_HEADER_FILE_H_
#define _CUDA_HEADER_FILE_H_

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


class Feature;

class Parameter{
 public:
  float weight;
  float step_size;
  float meta_step_size;
  float utility_to_distribute;
  float synapse_local_utility_trace;
  Feature* to;
  Feature* from;

  __host__ __device__
  void update_utility();

  __host__ __device__
  Parameter(float step_size, float weight, float meta_step_size, Feature* to, Feature* from);
  __host__ __device__
  void construct(Feature* to, Feature* from);
};

class Feature{
 public:
  float value;
  float temp_value;
  float old_value;
  float old_temp_value;
  float trace;
  float threshold;
  float sum_of_utility_traces;
  int id;
  int age;
  bool is_output_feature;
  bool is_input_feature;
  int feature_utility;
  int total_features;
  int total_outgoing_features;
  Parameter** incoming_parameters;
  Parameter** outgoing_parameters;
  __host__
  Feature();
  __host__
  void construct();


  __host__ __device__
  void update_utility();

  __host__ __device__
  float forward(float pre_activation);

  __host__ __device__
  void add_incoming_parameter(Parameter* p);
  __host__ __device__
  void add_outgoing_parameter(Parameter* p);

  __host__ __device__
  void update_value();


  __host__ __device__
  void fire();


  __host__ __device__
  void set_feature();
  __host__ __device__
  void unset_feature();
};



#endif //_CUDA_HEADER_FILE_H_
