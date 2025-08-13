#ifndef CSIGNAL_COMPAT_H
#define CSIGNAL_COMPAT_H

#include <csignal>

// GCC 11+ csignal fix for CUDA
#include <signal.h>

#endif // CSIGNAL_COMPAT_H