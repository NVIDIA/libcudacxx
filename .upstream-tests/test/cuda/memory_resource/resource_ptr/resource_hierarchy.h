#pragma once

#include <cuda/memory_resource>

class derived1 : public cuda::pmr::memory_resource<cuda::pmr::memory_kind::managed> {
};

class derived2 : public derived1 {
};


class derived_async1 : public cuda::pmr::stream_ordered_resource<cuda::pmr::memory_kind::managed> {
};

class derived_async2 : public derived_async1 {
};

/*
  D2  - derived2, host-accessible, oversubscribable, host-located
  D1  - derived1, host-accessible, host-located
  DA2 - derived_async2, host-accessible, oversubscribable, host-located
  DA1 - derived_async1, oversubscribable, host-located
  M   - memory_resource<managed>, host-accessible, host-located
  MA  - stream_ordered_resource<managed>, host-accessible, host-located
  B   - memory_resource_base, host-located
  BA  - stream_ordered_resource_base, host-located

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

// NOTE - some views are parameterized with separate properties and some with property_list<...>

using view_D2 = cuda::pmr::basic_resource_ptr<
    derived2*,
    cuda::pmr::memory_access::host,
    cuda::pmr::oversubscribable,
    cuda::pmr::memory_location::host>;

using view_D1 = cuda::pmr::basic_resource_ptr<
    derived1*,
    cuda::pmr::memory_access::host,
    cuda::pmr::memory_location::host>;

using view_DA2 = cuda::pmr::basic_resource_ptr<
    derived_async2*,
    cuda::pmr::property_list<
        cuda::pmr::memory_access::host,
        cuda::pmr::oversubscribable,
        cuda::pmr::memory_location::host>>;

using view_DA1 = cuda::pmr::basic_resource_ptr<
    derived_async1*,
    cuda::pmr::property_list<
        cuda::pmr::memory_access::host,
        cuda::pmr::memory_location::host>>;

using view_M = cuda::pmr::basic_resource_ptr<
    cuda::pmr::memory_resource<cuda::pmr::memory_kind::managed>*,
    cuda::pmr::property_list<
        cuda::pmr::memory_access::host,
        cuda::pmr::memory_location::host>>;

using view_MA = cuda::pmr::basic_resource_ptr<
    cuda::pmr::stream_ordered_resource<cuda::pmr::memory_kind::managed>*,
    cuda::pmr::memory_access::host,
    cuda::pmr::memory_location::host>;

using view_B = cuda::pmr::resource_ptr<
    cuda::pmr::memory_location::host>;

using view_BA = cuda::pmr::stream_ordered_resource_ptr<
    cuda::pmr::memory_location::host>;
