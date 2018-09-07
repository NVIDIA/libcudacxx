nvcc -I../includes -arch=compute_70 -O2 trie.cu --expt-relaxed-constexpr -o trie
