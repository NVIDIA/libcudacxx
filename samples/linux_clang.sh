clang++ -I../include --cuda-gpu-arch=sm_70 -std=c++11 -O2 trie.cu -L/usr/local/cuda/lib64/ -lcudart_static -pthread -ldl -lrt -o trie
