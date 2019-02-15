#!/usr/bin/env bash
g++ -std=c++11 trie_st.cpp -O2 -o trie_st
g++ -std=c++11 trie_mt.cpp -O2 -o trie_mt -pthread
nvcc -I../include -arch=compute_70 -std=c++11 -O2 trie.cu --expt-relaxed-constexpr -o trie
