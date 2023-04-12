TOLERANCE = {
    "float32": {"atol": 1e-6, "rtol": 1e-6},
    "float16": {"atol": 1e-4, "rtol": 1e-4},
    "bfloat16": {"atol": 1e-3, "rtol": 1e-3},
}

def convert_dtype_to_torch_type(dtype):
    import torch

    if dtype == 'float32':
        return torch.float32
    elif dtype == 'float16':
        return torch.float16
    elif dtype == 'bfloat16':
        return torch.bfloat16
    