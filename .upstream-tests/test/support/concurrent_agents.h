//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __CUDA_ARCH__
    #include <thread>
#endif

#if defined(_NV_TARGET_DEF) && _NV_TARGET_DEF < 350
    #error "This test requires CUDA dynamic parallelism to work."
#endif

template<typename... Fs>
__host__ __device__
void concurrent_agents_launch(Fs ...fs)
{
    NV_DISPATCH_TARGET(
        NV_IS_DEVICE, (
            assert(blockDim.x == sizeof...(Fs));

            using fptr = void (*)(void *);

            fptr device_threads[] = {
                [](void * data) {
                    (*reinterpret_cast<Fs *>(data))();
                }...
            };

            void * device_thread_data[] = {
                reinterpret_cast<void *>(&fs)...
            };

            __syncthreads();

            device_threads[threadIdx.x](device_thread_data[threadIdx.x]);

            __syncthreads();

        ),
        NV_IS_HOST, (
            std::thread threads[]{
                std::thread{ std::forward<Fs>(fs) }...
            };

            for (auto && thread : threads)
            {
                thread.join();
            }
        )
    )
}

