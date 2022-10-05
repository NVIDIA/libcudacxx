---
parent: Setup
nav_order: 1
---

# Getting libcu++

## NVIDIA HPC SDK or CUDA Toolkit

libcu++ is included in the NVIDIA HPC SDK and the CUDA Toolkit.
It is on the default include path.
Today, there is no shared library component to libcu++; it's all header-only.
No additional compiler flags are needed.

## GitHub

libcu++ is an open source project developed on GitHub, which is where you'll
  find the latest versions and the development branch.
Our GitHub repository is [github.com/nvidia/libcudacxx](https://github.com/nvidia/libcudacxx).

You can clone the repository via
```bash
git clone -c core.symlinks=true https://github.com/nvidia/libcudacxx
```

After cloning the repository, one needs to initialize the llvm submodule:
```bash
git submodule init 
cd llvm-project
git init
git remote add origin https://github.com:miscco/llvm-project
git sparse-checkout init --cone
git sparse-checkout set --sparse-index libcxx libcxxabi libunwind
git fetch --depth=1 origin libcudacxx_base
git checkout origin/libcudacxx_base
```