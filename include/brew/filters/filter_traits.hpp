#pragma once
#include <type_traits>
#include <utility>

namespace brew::filters {

template <typename Dist>
struct default_filter;

template <typename Dist>
using default_filter_t = typename default_filter<Dist>::type;

}

namespace brew::models {
template<class T>
decltype(auto) current_density(T& density) {
    if constexpr (requires { density.current(); }) return (density.current());
    else return (density);
}
template<class T>
using current_density_t = std::remove_cvref_t<decltype(current_density(std::declval<T&>()))>;
}
