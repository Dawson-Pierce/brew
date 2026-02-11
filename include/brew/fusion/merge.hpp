#pragma once

#include "brew/distributions/mixture.hpp"
#include "brew/distributions/gaussian.hpp"
#include "brew/distributions/ggiw.hpp"
#include "brew/distributions/trajectory_gaussian.hpp"
#include "brew/distributions/trajectory_ggiw.hpp"

namespace brew::fusion {

/// Merge close Gaussian components via moment matching.
void merge(distributions::Mixture<distributions::Gaussian>& mixture, double threshold);

/// Merge close GGIW components via weighted averaging (heaviest-first grouping).
void merge(distributions::Mixture<distributions::GGIW>& mixture, double threshold);

/// Merge close TrajectoryGaussian components (same-size, absorb into longer/heavier).
void merge(distributions::Mixture<distributions::TrajectoryGaussian>& mixture, double threshold);

/// Merge close TrajectoryGGIW components (same-size, absorb into longer/heavier).
void merge(distributions::Mixture<distributions::TrajectoryGGIW>& mixture, double threshold);

} // namespace brew::fusion
