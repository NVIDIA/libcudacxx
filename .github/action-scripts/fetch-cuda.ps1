# Fetch CUDA Toolkit 11.7
Invoke-WebRequest -Uri "https://developer.download.nvidia.com/compute/cuda/12.1.0/network_installers/cuda_12.1.0_windows_network.exe" -OutFile "./cuda_network.exe"
./cuda_network.exe -s nvcc_12.1 cudart_12.1

$env:PATH="$env:PATH;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\bin;"
