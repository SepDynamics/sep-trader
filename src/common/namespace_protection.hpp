#pragma once

// Protect standard library names from macro pollution.
// Include this header before and after third-party headers
// that may redefine standard symbols. The first inclusion
// pushes and undefines macros; the second inclusion restores them.

#ifndef SEP_NAMESPACE_PROTECTION_ACTIVE
#define SEP_NAMESPACE_PROTECTION_ACTIVE

#ifdef string
#pragma push_macro("string")
#undef string
#define SEP_NS_PROTECT_STRING
#endif

#ifdef cout
#pragma push_macro("cout")
#undef cout
#define SEP_NS_PROTECT_COUT
#endif

#ifdef ostream
#pragma push_macro("ostream")
#undef ostream
#define SEP_NS_PROTECT_OSTREAM
#endif

#ifdef std
#pragma push_macro("std")
#undef std
#define SEP_NS_PROTECT_STD
#endif

#else // SEP_NAMESPACE_PROTECTION_ACTIVE

#ifdef SEP_NS_PROTECT_STRING
#pragma pop_macro("string")
#undef SEP_NS_PROTECT_STRING
#endif

#ifdef SEP_NS_PROTECT_COUT
#pragma pop_macro("cout")
#undef SEP_NS_PROTECT_COUT
#endif

#ifdef SEP_NS_PROTECT_OSTREAM
#pragma pop_macro("ostream")
#undef SEP_NS_PROTECT_OSTREAM
#endif

#ifdef SEP_NS_PROTECT_STD
#pragma pop_macro("std")
#undef SEP_NS_PROTECT_STD
#endif

#undef SEP_NAMESPACE_PROTECTION_ACTIVE
#endif // SEP_NAMESPACE_PROTECTION_ACTIVE

