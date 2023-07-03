#include "relu.hpp"

at::Tensor relu_forward(const at::Tensor& input)
{
    return DISPATCH_DEVICE_IMPL(relu_forward, input);
}
