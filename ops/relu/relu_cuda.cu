#include "cuda_helper.hpp"
#include "relu.hpp"

template <typename scalar_t>
__global__ void relu_forward_cuda_kernel(const std::size_t input_size,
    const int channels,
    const int height,
    const int width,
    const scalar_t* src_data,
    scalar_t* dst_data)
{

    std::size_t index = blockIdx.x * blockDim.x + threadIdx.x; // 计算绝对索引
    const std::size_t stride = gridDim.x * blockDim.x; // 计算总线程数
    const std::size_t chw = channels * height * width;

    for (; index < input_size; index += stride) {
        auto value = src_data[index]; // 寻找到原数据值
        dst_data[index] = value >= 0 ? value : scalar_t(0);
    }
}

at::Tensor relu_forward_cuda(const at::Tensor& src)
{

    // CHECK_INPUT(input);
    at::DeviceGuard guard(src.device());
    int batch = src.size(0);
    int channels = src.size(1);
    int height = src.size(2);
    int width = src.size(3);
    const std::size_t input_size = batch * channels * height * width;
    const std::size_t output_size = batch * channels * height * width;

    at::Tensor dst = at::empty({ batch, channels, height, width }, // 开辟一段存储空间
        src.options());

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(src.scalar_type(), "relu_forward_cuda_kernel", ([&] {
        relu_forward_cuda_kernel<scalar_t>
            <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK, 0, at::cuda::getCurrentCUDAStream()>>>(
                input_size, channels, height, width, src.data_ptr<scalar_t>(), dst.data_ptr<scalar_t>());
    }));
    AT_CUDA_CHECK(cudaGetLastError());
    return dst;
}

REGISTER_DEVICE_IMPL(relu_forward, CUDA, relu_forward_cuda);

int test_relu_forward_cuda()
{
    torch::Tensor input = torch::randn({ 1, 2, 3, 4 }, torch::kCUDA);
    torch::Tensor output = relu_forward_cuda(input);
    std::cout << input << std::endl;
    std::cout << output << std::endl;
    return 0;
}
