#pragma once

#include "brew/models/mixture.hpp"
#include "brew/models/gaussian.hpp"
#include "brew/models/ggiw.hpp"
#include "brew/models/trajectory_gaussian.hpp"
#include "brew/models/trajectory_ggiw.hpp"

namespace brew::fusion {

/// Merge close Gaussian components via moment matching.
void merge(models::Mixture<models::Gaussian>& mixture, double threshold);

/// Merge close GGIW components via weighted averaging (heaviest-first grouping).
void merge(models::Mixture<models::GGIW>& mixture, double threshold);

/// Merge close TrajectoryGaussian components (same-size, absorb into longer/heavier).
void merge(models::Mixture<models::TrajectoryGaussian>& mixture, double threshold);

/// Merge close TrajectoryGGIW components (same-size, absorb into longer/heavier).
void merge(models::Mixture<models::TrajectoryGGIW>& mixture, double threshold);

} // namespace brew::fusion
