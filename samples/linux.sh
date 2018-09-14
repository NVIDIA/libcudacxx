g++ trie_st.cpp -O2 -o trie_st
g++ trie_mt.cpp -O2 -o trie_mt -pthread
nvcc -I../includes -arch=compute_70 -std=c++11 -O2 trie.cu --expt-relaxed-constexpr -o trie
