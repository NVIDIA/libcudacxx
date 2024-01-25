## Memory access properties

:warning: **The libcudacxx repository has been archived and is now part of the unified [nvidia/cccl repository](https://github.com/nvidia/cccl). See the [announcement here](https://github.com/NVIDIA/cccl/discussions/520) for more information. Please visit the new repository for the latest updates. The new documentation can be found [here](https://nvidia.github.io/cccl/libcudacxx/).** :warning:

| [`cuda::annotated_ptr`]             | Binds an access property to a pointer. `(class template)` <br/><br/> 1.6.0 / CUDA 11.5 |
| [`cuda::access_property`]           | Represents a memory access property. `(class)` <br/><br/> 1.6.0 / CUDA 11.5 |
| [`cuda::apply_access_property`]     | Applies access property to memory location. `(function template)` <br/><br/> 1.6.0 / CUDA 11.5 |
| [`cuda::associate_access_property`] | Associates access property with raw pointer. `(function template)` <br/><br/> 1.6.0 / CUDA 11.5 |
| [`cuda::discard_memory`]            | Writes indeterminate values to memory. `(function)` <br/><br/> 1.6.0 / CUDA 11.5 |

[`cuda::annotated_ptr`]: {{ "extended_api/memory_access_properties/annotated_ptr.html" | relative_url }}
[`cuda::access_property`]: {{ "extended_api/memory_access_properties/access_property.html" | relative_url }}
[`cuda::associate_access_property`]: {{ "extended_api/memory_access_properties/associate_access_property.html" | relative_url }}
[`cuda::apply_access_property`]: {{ "extended_api/memory_access_properties/apply_access_property.html" | relative_url }}
[`cuda::discard_memory`]: {{ "extended_api/memory_access_properties/discard_memory.html" | relative_url }}
