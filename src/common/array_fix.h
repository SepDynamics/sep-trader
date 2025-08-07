#pragma once

// Explicit array header inclusion to fix std::array issues
#include <array>

// Force functional to see array
#ifndef _GLIBCXX_ARRAY
#define _GLIBCXX_ARRAY 1
#endif

#include <functional>
