import torch
print(torch.cuda.is_available())


# Verifica se CUDA è disponibile
print("CUDA available:", torch.cuda.is_available())

# Mostra il nome della GPU
if torch.cuda.is_available():
    print("Device name:", torch.cuda.get_device_name(0))

# Esegui un'operazione semplice sulla GPU
x = torch.rand(3, 3).cuda()  # Crea un tensor casuale sulla GPU
y = torch.rand(3, 3).cuda()  # Crea un altro tensor casuale sulla GPU
z = x + y  # Somma i tensori sulla GPU

print("Computation successful! Tensor z on GPU:", z)