#pragma once

namespace brew::filters {

template <typename Dist>
struct default_filter;

template <typename Dist>
using default_filter_t = typename default_filter<Dist>::type;

}
