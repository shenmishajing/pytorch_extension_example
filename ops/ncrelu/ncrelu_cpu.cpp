#include "cpp_helper.hpp"
#include "ncrelu.hpp"

template <typename scalar_t>
void ncrelu_forward_cpu_kernel(const std::size_t input_size,
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
        dst_data[index + index / chw * chw] = value >= 0 ? value : scalar_t(0); // 前一半通道为正值
        dst_data[index + index / chw * chw + chw] = value >= 0 ? scalar_t(0) : value; // 后一半通道为负值
    }
}

torch::Tensor ncrelu_forward_cpu(const torch::Tensor& src)
{

    CHECK_CPU_INPUT(src);
    torch::DeviceGuard guard(src.device());
    int batch = src.size(0);
    int channels = src.size(1);
    int height = src.size(2);
    int width = src.size(3);
    const std::size_t input_size = batch * channels * height * width;
    const std::size_t output_size = batch * channels * height * width;

    torch::Tensor dst = torch::empty({ batch, 2 * channels, height, width }, // 开辟一段存储空间
        src.options());

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(src.scalar_type(), "ncrelu_forward_cpu_kernel", ([&] {
        ncrelu_forward_cpu_kernel<scalar_t>(
            input_size, channels, height, width, src.data_ptr<scalar_t>(), dst.data_ptr<scalar_t>());
    }));
    return dst;
}

REGISTER_DEVICE_IMPL(ncrelu_forward, CPU, ncrelu_forward_cpu);

int test_ncrelu_forward_cpu()
{
    torch::Tensor input = torch::randn({ 1, 2, 3, 4 }, torch::kCPU);
    torch::Tensor output = ncrelu_forward_cpu(input);
    std::cout << input << std::endl;
    std::cout << output << std::endl;
    return 0;
}
