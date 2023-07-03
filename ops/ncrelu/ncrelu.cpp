#include "ncrelu.hpp"

at::Tensor ncrelu_forward(const at::Tensor& input)
{
    return DISPATCH_DEVICE_IMPL(ncrelu_forward, input);
}
