clang++ -std=c++14 -O2 trie_st.cpp -o trie_st
clang++ -std=c++14 -O2 trie_mt.cpp -o trie_mt
/Developer/NVIDIA/CUDA-9.2/bin/nvcc -I../includes -arch=compute_70 -std=c++11 -O2 trie.cu --compiler-bindir /usr/local/opt/llvm@5/bin/clang --expt-relaxed-constexpr -o trie
