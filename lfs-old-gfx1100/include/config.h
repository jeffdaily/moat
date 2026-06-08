#pragma once

// Version information
#define PROJECT_VERSION_MAJOR 0
#define PROJECT_VERSION_MINOR 5
#define PROJECT_VERSION_PATCH 2
#define PROJECT_VERSION_STRING "0.5.2"

// Git version (generated at build time)
#include "git_version.h"

// Build type
/* #undef DEBUG_BUILD */
/* #undef RELEASE_BUILD */

// Platform detection
#if defined(_WIN32) || defined(_WIN64)
#define PLATFORM_WINDOWS
#elif defined(__linux__)
#define PLATFORM_LINUX
#else
#define PLATFORM_UNKNOWN
#endif
