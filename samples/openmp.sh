clang++ -D_LIBCPP_BARRIER_BUTTERFLY -I../include -fopenmp=libomp -L../../llvm-project/build/lib/ -std=c++11 -O2 benchmark.cpp -lstdc++ -lpthread -lm -o benchmark
