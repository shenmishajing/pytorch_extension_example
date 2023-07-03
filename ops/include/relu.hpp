#ifndef _RELU_HPP
#define _RELU_HPP

#include "cpp_helper.hpp"

at::Tensor relu_forward(const at::Tensor& input);

int test_relu_forward_cuda();

int test_relu_forward_cpu();

#endif // _RELU_HPP
