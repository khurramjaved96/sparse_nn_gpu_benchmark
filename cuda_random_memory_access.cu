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
#include "cuda_header_file.h"
#include <random>
#include "include/experiment/Experiment.h"
#include "include/experiment/Metric.h"
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

float Feature::forward(float pre_activation){
  return pre_activation;
}

void Parameter::update_utility() {
  float diff = this->to->value - this->to->forward(
      this->to->temp_value - this->from->old_value * this->weight);
  if(diff < 0)
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
  for (int counter = 0; counter< this->total_features; counter++) {
      this->sum_of_utility_traces += this->incoming_parameters[counter]->synapse_local_utility_trace;
  }

  for (int counter = 0; counter< this->total_features; counter++) {
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

//
//  __global__
//  void update_value(int n, Parameter **x, Feature **y, float *result) {
//    int index = blockIdx.x * blockDim.x + threadIdx.x;
//    int stride = blockDim.x * gridDim.x;
//    for (int i = index; i < n; i += stride) {
//      result[i] = x[i][0].weight * y[i][0].value;
//      y[i][0].trace = y[i][0].trace * 0.9 + 0.1;
////    printf("%f\t%f\t%f\n", result[i], x[i][0].weight, y[i]->value);
//    }
//  }

int main(int argc, char *argv[]) {
  Experiment my_experiment = Experiment(argc, argv);


  Metric error_metric = Metric(my_experiment.database_name, "time",
                               std::vector < std::string > {"run", "fps"},
                               std::vector < std::string > {"int", "real"},
                               std::vector < std::string > {"run"});

  std::vector<std::vector<std::string>> error_logger;

  std::vector<int> threads{my_experiment.get_int_param("threads")};
  std::vector<int> blocks{my_experiment.get_int_param("blocks")};

  using std::chrono::high_resolution_clock;
  using std::chrono::duration_cast;
  using std::chrono::duration;
  using std::chrono::milliseconds;

  int total_features = my_experiment.get_int_param("features");
  int parameters_per_feature = my_experiment.get_int_param("connections");
  Parameter *parameter_pointer;
  Feature *features_pointer;


  std::vector<std::string> error;
  error.push_back(std::to_string(my_experiment.get_int_param("run")));


  cudaMallocManaged(&parameter_pointer, sizeof(Parameter) * total_features * parameters_per_feature);

  std::cout << "Parameter memory allocated\n";

  cudaMallocManaged(&features_pointer, sizeof(Feature) * total_features);

  std::cout << "Feature memory allocated\n";

  std::cout << "Constructing features\n";
  for (int i = 0; i < total_features; i++) {
    if(i%1000000 == 0)
      std::cout << "#";
    features_pointer[i].construct();
  }
  std::cout << "\n";
  std::mt19937 mt(0);
  std::uniform_int_distribution<int> dist(0, total_features - 1);
  for (int i = 0; i < total_features * parameters_per_feature; i++) {
    if(i%1000000 == 0)
      std::cout << "#";
    parameter_pointer[i].construct(&features_pointer[dist(mt)], &features_pointer[dist(mt)]);
  }
  std::cout << "\n";

  std::cout << "Running experiment\n";
  for(auto block: blocks){
    for(auto thread: threads)
    {

      for(int temp = 0; temp < 10; temp++){
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


//        if (i % 100 == 0) {
//          auto t2 = high_resolution_clock::now();
//          duration<double, std::milli> ms_double = t2 - t1;
//          std::cout << "Step\t" << i << " Time\t" << ms_double.count() / (i + 1) << "ms\n";
//        }
      }
      auto t2 = high_resolution_clock::now();
      duration<double, std::milli> ms_double = t2 - t1;
      std::cout << "Blocks\tThreads\tTime\n";
      std::cout << block <<"\t" << thread << "\t" << 1000.0/(ms_double.count() / (total_steps)) << "fps\n";
      error.push_back(std::to_string(1000.0/(ms_double.count() / (total_steps))));
      error_logger.push_back(error);

      error_metric.add_values(error_logger);
      error_logger.clear();

//      std::ofstream myfile ("results.txt", std::ios_base::app);
//      if (myfile.is_open())
//      {
//        myfile << "run\tfeatures\tconnections\tthreads\tblocks\tgpu\tfps\n";
//        myfile << my_experiment.get_int_param("run");
//        myfile <<"\t";
//        myfile << my_experiment.get_int_param("features");
//        myfile <<"\t\t";
//        myfile << my_experiment.get_int_param("connections");
//        myfile <<"\t\t";
//        myfile << my_experiment.get_int_param("threads");
//        myfile <<"\t";
//        myfile << my_experiment.get_int_param("blocks");
//        myfile <<"\t";
//        myfile << my_experiment.get_string_param("gpu");
//        myfile <<"\t";
//        myfile << 1000.0/(ms_double.count() / (total_steps));
//        myfile << "\n";
//        myfile.close();
//      }
    }
  }
}