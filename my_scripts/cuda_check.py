import torch

print("Доступность CUDA: ", torch.cuda.is_available())

print("Список доступных устройств:")
for i in range(torch.cuda.device_count()):
    print(torch.cuda.get_device_name(i))