import torch

cuda_available = torch.cuda.is_available()


def to_best_device(tensor: torch.Tensor) -> torch.Tensor:
    if cuda_available:
        print('Sent tensor to CUDA')
        tensor.cuda()
    return tensor