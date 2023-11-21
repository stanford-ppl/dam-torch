import torch
import argparse

import sys

parser = argparse.ArgumentParser()
parser.add_argument("iters", type=int)
parser.add_argument("matrix_size", type=int)
args = parser.parse_args(sys.argv[1:])

class Foo(torch.nn.Module):
    # `Final` from the `typing_extensions` module can also be used
    matrix_size : torch.jit.Final[int]
    iters : torch.jit.Final[int]

    def __init__(self):
        super().__init__()
        self.matrix_size = args.matrix_size
        self.iters = args.iters

    def forward(self, input_tensor):
        # create a modestly-large workload
        invented_tensor = torch.arange(self.matrix_size, device=input_tensor.device, dtype=input_tensor.dtype) # A N-wide vector
        matrix = torch.outer(invented_tensor, input_tensor)
        self_product = torch.matmul(matrix, torch.transpose(matrix, 0, 1))
        for _  in range(self.iters):
            self_product = torch.softmax(torch.matmul(self_product, torch.transpose(self_product, 0, 1)), dim=0)
        # collapse it down to the same shape as the input
        projection_back = torch.arange(self.matrix_size, device=input_tensor.device, dtype=input_tensor.dtype)
        return torch.matmul(self_product, projection_back).flatten()

scripted = torch.jit.script(Foo())

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

start.record()
output = scripted.forward(torch.rand(1000, device="cuda", dtype=torch.float32))
end.record()

# Waits for everything to finish running
torch.cuda.synchronize()

print(start.elapsed_time(end))

torch.jit.save(scripted, "src/resources/busywork.pt")
