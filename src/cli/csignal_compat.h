#ifndef CSIGNAL_COMPAT_H
#define CSIGNAL_COMPAT_H

#include <csignal>

#ifdef __GNUC__
#if __GNUC__ >= 11
// GCC 11+ csignal fix
#include <signal.h>
#endif
#endif

#endif // CSIGNAL_COMPAT_H