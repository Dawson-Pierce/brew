#pragma once

namespace brew::filters {

// Maps a distribution/model type to the concrete filter used with it, so RFS
// filters can hold the filter BY VALUE (devirtualized) instead of through a
// polymorphic Filter<Dist> pointer. Specialized in each filter's header; the
// primary is intentionally left undefined so an unspecialized model is a clear
// compile error rather than silent fallback.
template <typename Dist>
struct default_filter;

template <typename Dist>
using default_filter_t = typename default_filter<Dist>::type;

}  // namespace brew::filters
