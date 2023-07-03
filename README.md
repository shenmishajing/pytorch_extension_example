# An example project of Pytorch C++/CUDA Extention

An example project of Pytorch C++/CUDA Extention, and a tutorial of how to build, use and debug it.

## Installation

This project works with

```txt
From system:
- gcc==7.5.0
- nvcc==11.6
From conda:
- python 3.10.11
- torch==1.13.1+cu116
- torchvision==0.14.1+cu116
- torchaudio==0.13.1
- cmake==3.26.4
- ninja==1.10.2
- cudnn==7.6.5.32
```

An environment with other versions is not guaranteed to work properly.

You can install the dependencies by

```bash
conda create -n <env_name> python=3.10 cmake ninja cudnn
conda activate <env_name>
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1
pip install -r requirments.txt
```

> Note that you have to install `pytorch` through `pip` instead of `conda`, because it will install another `nvcc` and will give you some trouble.

This project supports two methods to install the extensions: `cmake` and `setuptools`.

The `setuptools` method is easy. In the conda env, you can just run:

```bash
python setup.py install
```

and use the extension in Python:

```python
import torch
import ops

result = ops.<func_name>(args, ...)
```

> Note that ops is the extension name, and you have to import torch before import ops.

The `cmake` method is a little bit complicated. You have to build the extension by yourself. In the conda env, you can run:

```bash
cmake -DCMAKE_EXPORT_COMPILE_COMMANDS:BOOL=TRUE -S../ops -Bbuild -G Ninja
cmake --build build --config Debug --target all
```

> Note that if you are not in the conda env, like using cmake tools in vscode, you have to set the `CMAKE_PREFIX_PATH` to the conda env path. You can set `-DCMAKE_PREFIX_PATH=<conda_env_path>` in the configure args of cmake tools.
> You may need to set the compiler path manually, if you are in trouble about them. You can set `-DCMAKE_C_COMPILER=<gcc_path> -DCMAKE_CXX_COMPILER=<g++_path> -DCMAKE_CUDA_COMPILER=<nvcc_path>` to specify their path explicitly.
> You can omit the `-DCMAKE_EXPORT_COMPILE_COMMANDS:BOOL=TRUE` and `-G Ninja` args. They are used for lsp server. We will talk about them later.

Then you can use the extension in Python:

```python
import torch

torch.ops.load_library("build/libops.so")

result = torch.ops.ops.<func_name>(args, ...)
```

> Note that you have to recompile the extension again if you change the code, regradless of which method you use.

## Pytorch C++/CUDA Extention

Pytorch C++/CUDA Extention is a way to write C++/CUDA code and use it in Python. It is a good way to accelerate your code, especially when you have to write some loops in Python.

### compile method

There are three methods to compile such a Pytorch C++/CUDA Extention: `jit`, `setuptools` and `cmake`.

#### setuptools

By using `setuptools`, you can compile the extension before you run it. It leverages the `setup.py` to install the extension. Therefore, you have to write some hardcoded source paths and dirs to include. The advantage of this method is that we compile the extension by Python and can use the `torch` package to get the compile args, so there will be fewer path errors. The disadvantage is also that we compile the extension by Python, so we can not get any information for the LSP server to complete our code, check our code and debug our code. So, it's enough to compile code written by other people, but not enough to write something new.

You can get an example in the `setup.py` of this project.

#### jit

The `jit` method is very similar to the `setuptools` method, in addition to that you compile the extension in runtime. It requires source codes during the runtime and needs a hardcoded compile setting, which may be complex. Therefore, it is not a good way to deploy your code, usually.

The usage of `jit` is the same as `setuptools`, excluding the code location and the name of args. You can get an example in the `benchmark.py`.

#### cmake

Unlike the above two methods, which compile the extension by Python, the `cmake` method treats the extension as an independent project and the only interface between the extension and `Pytorch` is a dynamic library compiled by `cmake`. `cmake` can provide enough information for the LSP server and DAP server to enable all their features. So, if you want to write a new extension instead of just compiling some code written by other people, you have to use the `cmake` method. But, you get what you pay for, the `cmake` method is the most complex method and easy to get some path errors.

To use the `cmake` method, you have to write a `CMakeLists.txt` first. The `CMakeLists.txt` in this project is general enough, and you can just copy it in most cases.

##### debug

The `CMakeLists.txt` in this project will compile the extension into a dynamic library for import in Python and also an executable file for debugging. You are recommended to write some unit tests for every kernel function and call the unit test in the main function for debugging.

For debugging, you can use any tools you like, including the pure `gdb` or `lldb` and the debugger plugins as `codelldb` of `vscode`, to debug a C++ extension. But for the cuda part, you have to use the one from `nvidia`. You can use the pure `cuda-gdb` or `Nsight Visual Studio Code Edition` of `vscode`.

##### LSP server

`cmake` can generate the `compile_commands.json` file, which is required by the LSP servers such as `clangd`. You can enable this by set `-DCMAKE_EXPORT_COMPILE_COMMANDS:BOOL=TRUE` in the configure args of `cmake`. If you are using `cmake tools` and `clangd` in `vscode`, this arg will be set automatically. But, the default generator of `cmake` is `makefile`, which will use the `--option-files` to specify the extra include dirs for `nvcc`. But, `clangd` can not parse the `--option-files` arg. So, you will get an error of can not find header files in `cu` files and `cuh` files. To address this issue, you have to set `ninja` as the generateor of `cmake`, and `cmake` will use `-I` and `-isystem` instead of `--option-files`. Thus, you are out of the trouble of path.

In addition to the `--option-files` arg, there are many args of `nvcc` can not be parsed by `clangd`. Theoretically, you can create a config file for `clangd` in the project root or the home dir to set the `clangd` args (for more details about the location of the config file, see [doc](https://clangd.llvm.org/config#files)), and suppress those diagnostics by such a config file:

```yaml
Diagnostics:
  Suppress:
    - drv_unknown_argument
    - drv_unsupported_opt
```

But, there is an issue of `clangd` about this, and you have to remove all the args not supported by `clangd` manually. I have provided a config file to remove all the args introduced by my `CMakeLists.txt` and not unsupported by `clangd`. You can get it from [here](https://github.com/shenmishajing/Setting-for-Mac/blob/master/config/clangd/config.yaml), and copy it to the project root and rename it to `.clangd`  or just copy it to the `.config` folder under the home dir to address this issue.

## The project

### architecture

The architecture of this project is borrowed and improved from the [ops](https://github.com/open-mmlab/mmcv/tree/main/mmcv/ops/csrc) part of [mmcv](https://github.com/open-mmlab/mmcv). Every op has a folder, and a main cpp file with the name `<op_name`> in it. The main cpp file will dispatch the op to different implementation according to the device of input data. The implementation of every device is in a separate file with the name `<op_name>_<device>.<ext>`. The `<device>` can be `cpu`, `cuda` or other platforms. For `cuda` implementation, the `<ext>` is `cu`. For `cpu` implementation, the `<ext>` is `cpp`. For other platforms, the `<ext>` may be different. If you add any device with a different `<ext`>, you must add it to the `file(GLOB_RECURSE srcs *.cpp *.cu)` line in the `CMakeLists.txt` file to compile these files with the extension you use.

By the way, the [README](https://github.com/open-mmlab/mmcv/blob/main/mmcv/ops/csrc/README.md) of `mmcv` is a good tutorial for writing a new C++/CUDA extension.

### `<op_name>.cpp`

To add a new operator, you have to create a folder named `<op_name>` under the `ops` folder. Then you have to create a `<op_name>.cpp` in it with the context as follows:

```c++
// <op_name>/<op_name>.cpp
#include "<op_name>.hpp"

at::Tensor <op_name>_forward(const at::Tensor& input)
{
    return DISPATCH_DEVICE_IMPL(<op_name>_forward, input);
}
```

The `DISPATCH_DEVICE_IMPL` is a macro defined in the `device_registry.hpp` file, which will dispatch the op to the corresponding device implementation. It works with another macro `REGISTER_DEVICE_IMPL`, which we will talk about later.

The `hpp` file is just used to define the interface of the op. You can create it under the `include` folder with the context as follows:

```c++
#ifndef _<op_name>_HPP
#define _<op_name>_HPP

#include "cpp_helper.hpp"

at::Tensor <op_name>_forward(const at::Tensor& input);

// unit tests, you can omit them if you do not want to add unit tests

int test_<op_name>_forward_cuda();

int test_<op_name>_forward_cpu();

#endif // _<op_name>_HPP
```

### CUDA implementation

Then, if you want to add the ability to run on cuda to this op, you have to create a `<op_name>_cuda.cu` in the `<op_name>` folder. The context of the `<op_name>_cuda.cu` is as follows:

```c++
#include "cuda_helper.hpp"
#include "<op_name>.hpp"

// The kernel function run on the device (gpu).
// Must return void and can not recevie mutable args.
// Must use <<<blocks, threads, 0, cuda_stream>>> to specify the number of blocks and threads.
template <typename scalar_t>
__global__ void <op_name>_forward_cuda_kernel(args, ...)
{

    std::size_t index = blockIdx.x * blockDim.x + threadIdx.x; // current index
    const std::size_t stride = gridDim.x * blockDim.x; // stride is equal to the number of threads

    // stride-loop
    for (; index < input_size; index += stride) {
        auto value = src_data[index]; // get current value
        
        // calculate the output value
        ...
    }
}

// The cuda interface of the op.
// Do some checks and prepare data for the kernel function.
// Use the kernel function to do the calculation.
torch::Tensor <op_name>_forward_cuda(args, ...)
{

    // check the input
    CHECK_CUDA_INPUT(args, ...);
    
    // prepare data for the kernel function
    ...

    // malloc a tensor to store the output
    torch::Tensor dst = torch::empty({ ... }, src.options());

    // launch the kernel function
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(src.scalar_type(), "<op_name>_forward_cuda_kernel", ([&] {
        <op_name>_forward_cuda_kernel<scalar_t>
            <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK, 0, at::cuda::getCurrentCUDAStream()>>>(args, ...);
    }));

    // report any error from gpu
    AT_CUDA_CHECK(cudaGetLastError());

    // return the output
    return dst;
}

// register the cuda implementation of the op
REGISTER_DEVICE_IMPL(<op_name>_forward, CUDA, <op_name>_forward_cuda);

// unit test, you can omit it if you do not want to add an unit test
int test_<op_name>_forward_cuda()
{
    // some test code
    return 0;
}
```

The macro `REGISTER_DEVICE_IMPL` registers the cuda implementation of the op as the func `<op_name>_forward_cuda`. With the macro `DISPATCH_DEVICE_IMPL` in the `<op_name>.cpp`, the op will be dispatched to the cuda implementation when the input is on cuda.

The macro `AT_DISPATCH_FLOATING_TYPES_AND_HALF` dispatches the op to different implementations according to the type of data.

The `GET_BLOCKS` and `THREADS_PER_BLOCK` are defined in the `cuda_helper.hpp` file. They will get the minimum block num you need.

### CPU implementation

Though it will be too slow and probably not much faster than pure Python code, you can also add a cpu implementation to the op. The context of the `<op_name>_cpu.cpp` can be similar to the `<op_name>_cuda.cu`:

```c++
#include "cpp_helper.hpp"
#include "<op_name>.hpp"

// The kernel function.
template <typename scalar_t>
void <op_name>_forward_cpu_kernel(args, ...)
{

    // use openmp to parallel the for-loop
#pragma omp parallel for
    for (std::size_t index = 0; index < input_size; index++) {
        auto value = src_data[index]; // get current value
        
        // calculate the output value
        ...
    }
}

// The cpu interface of the op.
// Do some checks and prepare data for the kernel function.
// Use the kernel function to do the calculation.
torch::Tensor <op_name>_forward_cpu(args, ...)
{

    // check the input
    CHECK_CPU_INPUT(args, ...);
    
    // prepare data for the kernel function
    ...

    // malloc a tensor to store the output
    torch::Tensor dst = torch::empty({ ... }, src.options());

    // launch the kernel function
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(src.scalar_type(), "<op_name>_forward_cpu_kernel", ([&] {
        <op_name>_forward_cpu_kernel<scalar_t>(args, ...);
    }));

    // return the output
    return dst;
}

// register the cpu implementation of the op
REGISTER_DEVICE_IMPL(<op_name>_forward, CPU, <op_name>_forward_cpu);

// unit test, you can omit it if you do not want to add an unit test
int test_<op_name>_forward_cpu()
{
    // some test code
    return 0;
}
```

Similarly, we use the macro `REGISTER_DEVICE_IMPL` to register the cpu implementation of the op as the func `<op_name>_forward_cpu`. With the macro `DISPATCH_DEVICE_IMPL` in the `<op_name>.cpp`, the op will be dispatched to the cpu implementation when the input is on cpu.

Also, the macro `AT_DISPATCH_FLOATING_TYPES_AND_HALF` dispatches the op to different implementations according to the type of data.

Note that, unlike the cuda implementation, we do not need to use the kernel architecture for the cpu implementation. But, I recommend you to use this architecture, because you can implement the cpu part by just making a little modifycation like changing `cpu` to `cuda` and the `stride-loop` to a openmp loop.

Again, you do not need to add a cpu implementation to the op. If you do not want to add a cpu implementation, you can just omit the `<op_name>_cpu.cpp` file.
