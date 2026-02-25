#pragma once

#include "brew/models/mixture.hpp"
#include "brew/models/gaussian.hpp"
#include "brew/models/ggiw.hpp"
#include "brew/models/ggiw_orientation.hpp"
#include "brew/models/trajectory_gaussian.hpp"
#include "brew/models/trajectory_ggiw.hpp"
#include "brew/models/trajectory_ggiw_orientation.hpp"

namespace brew::fusion {

/// Merge close Gaussian components via moment matching.
void merge(models::Mixture<models::Gaussian>& mixture, double threshold);

/// Merge close GGIW components via weighted averaging (heaviest-first grouping).
void merge(models::Mixture<models::GGIW>& mixture, double threshold);

/// Merge close GGIWOrientation components (same as GGIW merge; basis left empty for filter to recompute).
void merge(models::Mixture<models::GGIWOrientation>& mixture, double threshold);

/// Merge close TrajectoryGaussian components (same-size, absorb into longer/heavier).
void merge(models::Mixture<models::TrajectoryGaussian>& mixture, double threshold);

/// Merge close TrajectoryGGIW components (same-size, absorb into longer/heavier).
void merge(models::Mixture<models::TrajectoryGGIW>& mixture, double threshold);

/// Merge close TrajectoryGGIWOrientation components (same-size, absorb into longer/heavier; basis left empty).
void merge(models::Mixture<models::TrajectoryGGIWOrientation>& mixture, double threshold);

} // namespace brew::fusion
