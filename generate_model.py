import torch
import argparse

import sys

parser = argparse.ArgumentParser()
parser.add_argument("iters", type=int)
parser.add_argument("matrix_size", type=int)
args = parser.parse_args(sys.argv[1:])

@torch.jit.script
def benchmark_model(input_tensor: torch.Tensor):  # input_tensor is a vector consisting of just the batch
    # create a modestly-large workload
    invented_tensor = torch.arange(args.matrix_size, device=input_tensor.device, dtype=input_tensor.dtype) # A N-wide vector
    matrix = torch.outer(invented_tensor, input_tensor)
    self_product = torch.matmul(matrix, torch.transpose(matrix, 0, 1))
    for _  in range(args.iters):
        self_product = torch.softmax(torch.matmul(self_product, torch.transpose(self_product, 0, 1)), dim=0)
    # collapse it down to the same shape as the input
    projection_back = torch.arange(args.matrix_size, device=input_tensor.device, dtype=input_tensor.dtype)
    return torch.matmul(self_product, projection_back).flatten()


torch.jit.save(benchmark_model, "src/resources/busywork.pt")
