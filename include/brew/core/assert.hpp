#pragma once

#include <cassert>

// BREW_ASSERT(cond, msg): precondition/invariant check that aborts on failure.
// Compiles out under NDEBUG like the standard assert. Replaces exception-based
// precondition checks so the code is usable on targets built with
// -fno-exceptions.
#ifdef NDEBUG
    #define BREW_ASSERT(cond, msg) ((void)0)
#else
    #define BREW_ASSERT(cond, msg) assert((cond) && (msg))
#endif
