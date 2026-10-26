#pragma once

#include <cassert>

#ifdef NDEBUG
    #define BREW_ASSERT(cond, msg) ((void)0)
#else
    #define BREW_ASSERT(cond, msg) assert((cond) && (msg))
#endif
