import torch

if torch.cuda.is_available():
    print(f'PyTorch version: {torch.__version__}')
    print('*' * 10)
    print(f'CUDNN version: {torch.backends.cudnn.version()}')
    print(f'Available GPU devices: {torch.cuda.device_count()}')
    print(f'Device Name: {torch.cuda.get_device_name()}')
    device = "cuda"
else:
    device = "cpu"
print(f"Using {device} device")
