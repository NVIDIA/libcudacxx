nvcc -I../includes -arch=compute_70 -std=c++11 -O2 trie.cu --expt-relaxed-constexpr -o trie
