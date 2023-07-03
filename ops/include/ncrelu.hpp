#ifndef _NCRELU_HPP
#define _NCRELU_HPP

#include "cpp_helper.hpp"

at::Tensor ncrelu_forward(const at::Tensor& input);

int test_ncrelu_forward_cuda();

int test_ncrelu_forward_cpu();

#endif // _NCRELU_HPP
