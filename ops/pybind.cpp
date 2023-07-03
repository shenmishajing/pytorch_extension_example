#include "cpp_helper.hpp"
#include "ncrelu.hpp"
#include "relu.hpp"

TORCH_LIBRARY(ops, m)
{
    m.def("ncrelu_forward", &ncrelu_forward);
    m.def("relu_forward", &relu_forward);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("ncrelu_forward", &ncrelu_forward, "ncrelu forward");
    m.def("relu_forward", &relu_forward, "relu forward");
}

int main()
{
    // Any function you want to debug
    return test_ncrelu_forward_cuda();
}
