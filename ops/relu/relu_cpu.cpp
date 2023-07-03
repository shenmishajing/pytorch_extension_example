#include "cpp_helper.hpp"
#include "relu.hpp"

template <typename scalar_t>
void relu_forward_cpu_kernel(const std::size_t input_size,
    const int channels,
    const int height,
    const int width,
    const scalar_t* src_data,
    scalar_t* dst_data)
{

    const std::size_t chw = channels * height * width;

#pragma omp parallel for
    for (std::size_t index = 0; index < input_size; index++) {
        auto value = src_data[index]; // 寻找到原数据值
        dst_data[index] = value >= 0 ? value : scalar_t(0);
    }
}

torch::Tensor relu_forward_cpu(const torch::Tensor& src)
{

    CHECK_CPU_INPUT(src);
    torch::DeviceGuard guard(src.device());
    int batch = src.size(0);
    int channels = src.size(1);
    int height = src.size(2);
    int width = src.size(3);
    const std::size_t input_size = batch * channels * height * width;
    const std::size_t output_size = batch * channels * height * width;

    torch::Tensor dst = torch::empty({ batch, channels, height, width }, // 开辟一段存储空间
        src.options());

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(src.scalar_type(), "relu_forward_cpu_kernel", ([&] {
        relu_forward_cpu_kernel<scalar_t>(
            input_size, channels, height, width, src.data_ptr<scalar_t>(), dst.data_ptr<scalar_t>());
    }));
    return dst;
}

REGISTER_DEVICE_IMPL(relu_forward, CPU, relu_forward_cpu);

int test_relu_forward_cpu()
{
    torch::Tensor input = torch::randn({ 1, 2, 3, 4 }, torch::kCPU);
    torch::Tensor output = relu_forward_cpu(input);
    std::cout << input << std::endl;
    std::cout << output << std::endl;
    return 0;
}
