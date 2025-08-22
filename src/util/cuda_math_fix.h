#pragma once

// CUDA math function compatibility fix
// This header must be included before any system math headers when using CUDA

#ifdef __CUDACC__
// Prevent glibc math function conflicts with CUDA
#define __MATHCALL(function,suffix, args) \
  extern __typeof__(function) __##function##suffix args __THROW
#define __MATHCALLX(function,suffix, args, attrib) \
  extern __typeof__(function) __##function##suffix args __THROW attrib

// Disable problematic math function redefinitions
#undef __USE_MISC
#undef __USE_ISOC99
#endif
