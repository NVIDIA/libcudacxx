nvcc -I../include -gencode arch=compute_70,code=sm_70 -std=c++11 -O2 trie.cu --expt-relaxed-constexpr --expt-extended-lambda -o trie -rdc=true
