#pragma once

#include <cuda/memory_resource>

class derived1 : public cuda::memory_resource<cuda::memory_kind::managed> {
};

class derived2 : public derived1 {
};


class derived_async1 : public cuda::stream_ordered_memory_resource<cuda::memory_kind::managed> {
};

class derived_async2 : public derived_async1 {
};

/*
  D2  - derived2, host-accessible, oversubscribable, host-located
  D1  - derived1, host-accessible, host-located
  DA2 - derived_async2, host-accessible, oversubscribable, host-located
  DA1 - derived_async1, oversubscribable, host-located
  M   - memory_resource<managed>, host-accessible, host-located
  MA  - stream_ordered_memory_resource<managed>, host-accessible, host-located
  B   - memory_resource_base, host-located
  BA  - stream_ordered_memory_resource_base, host-located

The view compatibility is as follows:

from--->     DA2  DA1  D2  D1  MA  M  BA  B
to--v
    DA2      X
    DA1      X    X
    D2                 X
    D1                 X   X
    MA       X    X            X
    M        X    X    X   X   X   X
    BA       X    X            X      X
    B        X    X    X   X   X   X  X   X
*/


using view_D2 = cuda::basic_resource_view<
    derived2*,
    cuda::memory_access::host,
    cuda::oversubscribable,
    cuda::memory_location::host>;

using view_D1 = cuda::basic_resource_view<
    derived1*,
    cuda::memory_access::host,
    cuda::memory_location::host>;

using view_DA2 = cuda::basic_resource_view<
    derived_async2*,
    cuda::memory_access::host,
    cuda::oversubscribable,
    cuda::memory_location::host>;

using view_DA1 = cuda::basic_resource_view<
    derived_async1*,
    cuda::memory_access::host,
    cuda::memory_location::host>;

using view_M = cuda::basic_resource_view<
    cuda::memory_resource<cuda::memory_kind::managed>*,
    cuda::memory_access::host,
    cuda::memory_location::host>;

using view_MA = cuda::basic_resource_view<
    cuda::stream_ordered_memory_resource<cuda::memory_kind::managed>*,
    cuda::memory_access::host,
    cuda::memory_location::host>;

using view_B = cuda::resource_view<
    cuda::memory_location::host>;

using view_BA = cuda::stream_ordered_resource_view<
    cuda::memory_location::host>;
