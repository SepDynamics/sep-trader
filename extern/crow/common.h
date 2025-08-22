#pragma once
// Common Crow definitions

#include "crow.h"

// HTTP status codes for compatibility
#ifndef HTTP_OK
#define HTTP_OK 200
#endif

#ifndef HTTP_BAD_REQUEST
#define HTTP_BAD_REQUEST 400
#endif

#ifndef HTTP_UNAUTHORIZED
#define HTTP_UNAUTHORIZED 401
#endif

#ifndef HTTP_FORBIDDEN
#define HTTP_FORBIDDEN 403
#endif

#ifndef HTTP_NOT_FOUND
#define HTTP_NOT_FOUND 404
#endif

#ifndef HTTP_INTERNAL_ERROR
#define HTTP_INTERNAL_ERROR 500
#endif
