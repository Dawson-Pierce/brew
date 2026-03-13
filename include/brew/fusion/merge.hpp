#pragma once

#include "brew/models/mixture.hpp"
#include "brew/models/gaussian.hpp"
#include "brew/models/ggiw.hpp"
#include "brew/models/ggiw_orientation.hpp"
#include "brew/models/template_pose.hpp"
#include "brew/models/trajectory.hpp"

namespace brew::fusion {

/// Merge close Gaussian components via moment matching.
void merge(models::Mixture<models::Gaussian>& mixture, double threshold);

/// Merge close GGIW components via weighted averaging (heaviest-first grouping).
void merge(models::Mixture<models::GGIW>& mixture, double threshold);

/// Merge close GGIWOrientation components (same as GGIW merge; basis left empty for filter to recompute).
void merge(models::Mixture<models::GGIWOrientation>& mixture, double threshold);

/// Merge close Trajectory<Gaussian> components (same-size, absorb into longer/heavier).
void merge(models::Mixture<models::Trajectory<models::Gaussian>>& mixture, double threshold);

/// Merge close Trajectory<GGIW> components (same-size, absorb into longer/heavier).
void merge(models::Mixture<models::Trajectory<models::GGIW>>& mixture, double threshold);

/// Merge close Trajectory<GGIWOrientation> components (same-size, absorb into longer/heavier; basis left empty).
void merge(models::Mixture<models::Trajectory<models::GGIWOrientation>>& mixture, double threshold);

/// Merge close TemplatePose components (same-template only, rotation averaged via SVD projection).
void merge(models::Mixture<models::TemplatePose>& mixture, double threshold);

/// Merge close Trajectory<TemplatePose> components (same-size, same-template, absorb into longer/heavier).
void merge(models::Mixture<models::Trajectory<models::TemplatePose>>& mixture, double threshold);

} // namespace brew::fusion
