import glob
import os

import torch
from speed_benchmark import speed_benchmark
from torch.utils.cpp_extension import load


def ncrelu(x):
    return torch.cat([x.clamp(min=0), x.clamp(max=0)], dim=1)


def main():
    b = 8
    h = 128
    w = 128

    experiment_name = "cmake"

    device = "cuda"

    if "setup" in experiment_name:
        import ops
    if "jit" in experiment_name:
        ops = load(
            name="ops",
            sources=glob.glob("ops/**/*.cpp", recursive=True)
            + glob.glob("ops/**/*.cu", recursive=True),
            extra_include_paths=[os.path.abspath("ops/include")],
        )
    elif "cmake" in experiment_name:
        torch.ops.load_library("build/libops.so")
        ops = torch.ops.ops

    funcs = [
        ncrelu,
        ops.ncrelu_forward,
        torch.nn.functional.relu,
        ops.relu_forward,
    ]

    args = {
        "main_arg_name": "c",
        "data": {
            c: {
                "args": [torch.randn((b, c, h, w), device=device)],
            }
            for c in [10, 100, 200, 300]
            + ([] if device == "cpu" else [500, 700, 800, 1000])
        },
    }
    speed_benchmark(
        funcs,
        args,
        repeat=10,
        num=10,
        experiment_name=experiment_name,
        check_result=False,
    )


if __name__ == "__main__":
    main()
