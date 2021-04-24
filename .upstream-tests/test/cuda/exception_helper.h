#pragma once

#ifndef _LIBCUDACXX_NO_EXCEPTIONS
#define _LIBCUDACXX_TEST_TRY try
#define _LIBCUDACXX_TEST_CATCH(...) catch(__VA_ARGS__)
#else
#define _LIBCUDACXX_TEST_TRY if (1)
#define _LIBCUDACXX_TEST_CATCH(...) else if (0)
#endif // _LIBCUDACXX_NO_EXCEPTIONS
