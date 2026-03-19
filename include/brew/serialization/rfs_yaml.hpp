#pragma once

#include <yaml-cpp/yaml.h>
#include <Eigen/Dense>

#include "brew/models/gaussian.hpp"
#include "brew/models/ggiw.hpp"
#include "brew/models/ggiw_orientation.hpp"
#include "brew/models/trajectory.hpp"
#include "brew/models/mixture.hpp"
#include "brew/models/bernoulli.hpp"
#include "brew/multi_target/rfs_base.hpp"
#include "brew/multi_target/phd.hpp"
#include "brew/multi_target/cphd.hpp"
#include "brew/multi_target/glmb.hpp"
#include "brew/multi_target/jglmb.hpp"
#include "brew/multi_target/pmbm.hpp"
#include "brew/multi_target/mbm.hpp"
#include "brew/multi_target/mb.hpp"
#include "brew/multi_target/lmb.hpp"

#include <vector>
#include <map>
#include <string>

namespace brew::serialization {

// ============================================================
// Eigen helpers
// ============================================================

inline YAML::Node vector_to_yaml(const Eigen::VectorXd& v) {
    YAML::Node arr(YAML::NodeType::Sequence);
    for (Eigen::Index i = 0; i < v.size(); ++i) {
        arr.push_back(v(i));
    }
    return arr;
}

inline Eigen::VectorXd vector_from_yaml(const YAML::Node& j) {
    Eigen::VectorXd v(j.size());
    for (std::size_t i = 0; i < j.size(); ++i) {
        v(static_cast<Eigen::Index>(i)) = j[i].as<double>();
    }
    return v;
}

inline YAML::Node matrix_to_yaml(const Eigen::MatrixXd& m) {
    YAML::Node obj;
    obj["rows"] = static_cast<int>(m.rows());
    obj["cols"] = static_cast<int>(m.cols());
    YAML::Node data(YAML::NodeType::Sequence);
    for (Eigen::Index i = 0; i < m.rows(); ++i) {
        for (Eigen::Index j = 0; j < m.cols(); ++j) {
            data.push_back(m(i, j));
        }
    }
    obj["data"] = data;
    return obj;
}

inline Eigen::MatrixXd matrix_from_yaml(const YAML::Node& j) {
    int rows = j["rows"].as<int>();
    int cols = j["cols"].as<int>();
    Eigen::MatrixXd m(rows, cols);
    const auto& data = j["data"];
    int idx = 0;
    for (int i = 0; i < rows; ++i) {
        for (int jj = 0; jj < cols; ++jj) {
            m(i, jj) = data[idx++].as<double>();
        }
    }
    return m;
}

// ============================================================
// Gaussian
// ============================================================

inline YAML::Node to_yaml(const models::Gaussian& g) {
    YAML::Node j;
    j["type"] = "Gaussian";
    j["mean"] = vector_to_yaml(g.mean());
    j["covariance"] = matrix_to_yaml(g.covariance());
    return j;
}

inline models::Gaussian gaussian_from_yaml(const YAML::Node& j) {
    return models::Gaussian(
        vector_from_yaml(j["mean"]),
        matrix_from_yaml(j["covariance"])
    );
}

// ============================================================
// GGIW
// ============================================================

inline YAML::Node to_yaml(const models::GGIW& g) {
    YAML::Node j;
    j["type"] = "GGIW";
    j["mean"] = vector_to_yaml(g.mean());
    j["covariance"] = matrix_to_yaml(g.covariance());
    j["alpha"] = g.alpha();
    j["beta"] = g.beta();
    j["v"] = g.v();
    j["V"] = matrix_to_yaml(g.V());
    return j;
}

inline models::GGIW ggiw_from_yaml(const YAML::Node& j) {
    return models::GGIW(
        vector_from_yaml(j["mean"]),
        matrix_from_yaml(j["covariance"]),
        j["alpha"].as<double>(),
        j["beta"].as<double>(),
        j["v"].as<double>(),
        matrix_from_yaml(j["V"])
    );
}

// ============================================================
// GGIWOrientation
// ============================================================

inline YAML::Node to_yaml(const models::GGIWOrientation& g) {
    YAML::Node j;
    j["type"] = "GGIWOrientation";
    j["mean"] = vector_to_yaml(g.mean());
    j["covariance"] = matrix_to_yaml(g.covariance());
    j["alpha"] = g.alpha();
    j["beta"] = g.beta();
    j["v"] = g.v();
    j["V"] = matrix_to_yaml(g.V());
    if (g.basis().size() > 0) {
        j["basis"] = matrix_to_yaml(g.basis());
    }
    if (g.has_eigenvalues()) {
        j["eigenvalues"] = matrix_to_yaml(g.eigenvalues());
    }
    return j;
}

inline models::GGIWOrientation ggiw_orientation_from_yaml(const YAML::Node& j) {
    models::GGIWOrientation g(
        vector_from_yaml(j["mean"]),
        matrix_from_yaml(j["covariance"]),
        j["alpha"].as<double>(),
        j["beta"].as<double>(),
        j["v"].as<double>(),
        matrix_from_yaml(j["V"])
    );
    if (j["basis"].IsDefined()) {
        g.set_basis(matrix_from_yaml(j["basis"]));
    }
    return g;
}

// ============================================================
// Trajectory<Gaussian>
// ============================================================

inline YAML::Node to_yaml(const models::Trajectory<models::Gaussian>& g) {
    YAML::Node j;
    j["type"] = "TrajectoryGaussian";
    j["state_dim"] = g.state_dim;
    j["mean"] = vector_to_yaml(g.mean());
    j["covariance"] = matrix_to_yaml(g.covariance());
    return j;
}

inline models::Trajectory<models::Gaussian> trajectory_gaussian_from_yaml(const YAML::Node& j) {
    return models::Trajectory<models::Gaussian>(
        j["state_dim"].as<int>(),
        models::Gaussian(
            vector_from_yaml(j["mean"]),
            matrix_from_yaml(j["covariance"])
        )
    );
}

// ============================================================
// Trajectory<GGIW>
// ============================================================

inline YAML::Node to_yaml(const models::Trajectory<models::GGIW>& g) {
    YAML::Node j;
    j["type"] = "TrajectoryGGIW";
    j["state_dim"] = g.state_dim;
    j["mean"] = vector_to_yaml(g.mean());
    j["covariance"] = matrix_to_yaml(g.covariance());
    j["alpha"] = g.current().alpha();
    j["beta"] = g.current().beta();
    j["v"] = g.current().v();
    j["V"] = matrix_to_yaml(g.current().V());
    return j;
}

inline models::Trajectory<models::GGIW> trajectory_ggiw_from_yaml(const YAML::Node& j) {
    return models::Trajectory<models::GGIW>(
        j["state_dim"].as<int>(),
        models::GGIW(
            vector_from_yaml(j["mean"]),
            matrix_from_yaml(j["covariance"]),
            j["alpha"].as<double>(),
            j["beta"].as<double>(),
            j["v"].as<double>(),
            matrix_from_yaml(j["V"])
        )
    );
}

// ============================================================
// Trajectory<GGIWOrientation>
// ============================================================

inline YAML::Node to_yaml(const models::Trajectory<models::GGIWOrientation>& g) {
    YAML::Node j;
    j["type"] = "TrajectoryGGIWOrientation";
    j["state_dim"] = g.state_dim;
    j["mean"] = vector_to_yaml(g.mean());
    j["covariance"] = matrix_to_yaml(g.covariance());
    j["alpha"] = g.current().alpha();
    j["beta"] = g.current().beta();
    j["v"] = g.current().v();
    j["V"] = matrix_to_yaml(g.current().V());
    if (g.current().basis().size() > 0) {
        j["basis"] = matrix_to_yaml(g.current().basis());
    }
    if (g.current().has_eigenvalues()) {
        j["eigenvalues"] = matrix_to_yaml(g.current().eigenvalues());
    }
    return j;
}

inline models::Trajectory<models::GGIWOrientation> trajectory_ggiw_orientation_from_yaml(const YAML::Node& j) {
    models::Trajectory<models::GGIWOrientation> result(
        j["state_dim"].as<int>(),
        models::GGIWOrientation(
            vector_from_yaml(j["mean"]),
            matrix_from_yaml(j["covariance"]),
            j["alpha"].as<double>(),
            j["beta"].as<double>(),
            j["v"].as<double>(),
            matrix_from_yaml(j["V"])
        )
    );
    if (j["basis"].IsDefined()) {
        result.current().set_basis(matrix_from_yaml(j["basis"]));
    }
    return result;
}

// ============================================================
// Generic distribution from_yaml dispatcher
// ============================================================

template <typename T>
struct DistributionSerializer;

template <>
struct DistributionSerializer<models::Gaussian> {
    static YAML::Node serialize(const models::Gaussian& d) { return to_yaml(d); }
    static models::Gaussian deserialize(const YAML::Node& j) { return gaussian_from_yaml(j); }
};

template <>
struct DistributionSerializer<models::GGIW> {
    static YAML::Node serialize(const models::GGIW& d) { return to_yaml(d); }
    static models::GGIW deserialize(const YAML::Node& j) { return ggiw_from_yaml(j); }
};

template <>
struct DistributionSerializer<models::GGIWOrientation> {
    static YAML::Node serialize(const models::GGIWOrientation& d) { return to_yaml(d); }
    static models::GGIWOrientation deserialize(const YAML::Node& j) { return ggiw_orientation_from_yaml(j); }
};

template <>
struct DistributionSerializer<models::Trajectory<models::Gaussian>> {
    static YAML::Node serialize(const models::Trajectory<models::Gaussian>& d) { return to_yaml(d); }
    static models::Trajectory<models::Gaussian> deserialize(const YAML::Node& j) { return trajectory_gaussian_from_yaml(j); }
};

template <>
struct DistributionSerializer<models::Trajectory<models::GGIW>> {
    static YAML::Node serialize(const models::Trajectory<models::GGIW>& d) { return to_yaml(d); }
    static models::Trajectory<models::GGIW> deserialize(const YAML::Node& j) { return trajectory_ggiw_from_yaml(j); }
};

template <>
struct DistributionSerializer<models::Trajectory<models::GGIWOrientation>> {
    static YAML::Node serialize(const models::Trajectory<models::GGIWOrientation>& d) { return to_yaml(d); }
    static models::Trajectory<models::GGIWOrientation> deserialize(const YAML::Node& j) { return trajectory_ggiw_orientation_from_yaml(j); }
};

// ============================================================
// Mixture<T>
// ============================================================

template <typename T>
YAML::Node mixture_to_yaml(const models::Mixture<T>& mix) {
    YAML::Node j;
    j["weights"] = vector_to_yaml(mix.weights());
    YAML::Node comps(YAML::NodeType::Sequence);
    for (std::size_t i = 0; i < mix.size(); ++i) {
        comps.push_back(DistributionSerializer<T>::serialize(mix.component(i)));
    }
    j["components"] = comps;
    return j;
}

template <typename T>
std::unique_ptr<models::Mixture<T>> mixture_from_yaml(const YAML::Node& j) {
    auto mix = std::make_unique<models::Mixture<T>>();
    auto weights = vector_from_yaml(j["weights"]);
    const auto& comps = j["components"];
    for (std::size_t i = 0; i < comps.size(); ++i) {
        auto dist = DistributionSerializer<T>::deserialize(comps[i]);
        mix->add_component(std::make_unique<T>(std::move(dist)),
                          weights(static_cast<Eigen::Index>(i)));
    }
    return mix;
}

// ============================================================
// Bernoulli<T>
// ============================================================

template <typename T>
YAML::Node bernoulli_to_yaml(const models::Bernoulli<T>& b) {
    YAML::Node j;
    j["existence_probability"] = b.existence_probability();
    j["id"] = b.id();
    if (b.has_distribution()) {
        j["distribution"] = DistributionSerializer<T>::serialize(b.distribution());
    }
    return j;
}

template <typename T>
std::unique_ptr<models::Bernoulli<T>> bernoulli_from_yaml(const YAML::Node& j) {
    double r = j["existence_probability"].as<double>();
    int id = j["id"].as<int>();
    std::unique_ptr<T> dist;
    if (j["distribution"].IsDefined()) {
        auto d = DistributionSerializer<T>::deserialize(j["distribution"]);
        dist = std::make_unique<T>(std::move(d));
    }
    return std::make_unique<models::Bernoulli<T>>(r, std::move(dist), id);
}

// ============================================================
// Track histories helper
// ============================================================

inline YAML::Node track_histories_to_yaml(const std::map<int, std::vector<Eigen::VectorXd>>& histories) {
    YAML::Node j;
    for (const auto& [id, states] : histories) {
        YAML::Node arr(YAML::NodeType::Sequence);
        for (const auto& s : states) {
            arr.push_back(vector_to_yaml(s));
        }
        j[std::to_string(id)] = arr;
    }
    return j;
}

inline std::map<int, std::vector<Eigen::VectorXd>> track_histories_from_yaml(const YAML::Node& j) {
    std::map<int, std::vector<Eigen::VectorXd>> result;
    for (auto it = j.begin(); it != j.end(); ++it) {
        int id = std::stoi(it->first.as<std::string>());
        std::vector<Eigen::VectorXd> states;
        for (const auto& s : it->second) {
            states.push_back(vector_from_yaml(s));
        }
        result[id] = std::move(states);
    }
    return result;
}

// ============================================================
// RFS Base config (common to all filters)
// ============================================================

inline YAML::Node rfs_base_to_yaml(const multi_target::RFSBase& rfs) {
    YAML::Node j;
    // RFSBase members are protected, so we access them via the public setters
    // by serializing from the concrete filter types instead.
    // This function is a placeholder for common structure.
    return j;
}

// ============================================================
// PHD<T>
// ============================================================

template <typename T>
YAML::Node phd_to_yaml(const multi_target::PHD<T>& phd) {
    YAML::Node j;
    j["filter_type"] = "PHD";
    j["prob_detection"] = phd.prob_detection();
    j["prob_survive"] = phd.prob_survive();
    j["clutter_rate"] = phd.clutter_rate();
    j["clutter_density"] = phd.clutter_density();
    j["intensity"] = mixture_to_yaml<T>(phd.intensity());
    if (!phd.extracted_mixtures().empty()) {
        YAML::Node extracts(YAML::NodeType::Sequence);
        for (const auto& mix : phd.extracted_mixtures()) {
            extracts.push_back(mixture_to_yaml<T>(*mix));
        }
        j["extracted_mixtures"] = extracts;
    }
    return j;
}

// ============================================================
// CPHD<T>
// ============================================================

template <typename T>
YAML::Node cphd_to_yaml(const multi_target::CPHD<T>& cphd) {
    YAML::Node j;
    j["filter_type"] = "CPHD";
    j["prob_detection"] = cphd.prob_detection();
    j["prob_survive"] = cphd.prob_survive();
    j["clutter_rate"] = cphd.clutter_rate();
    j["clutter_density"] = cphd.clutter_density();
    j["intensity"] = mixture_to_yaml<T>(cphd.intensity());
    j["cardinality"] = vector_to_yaml(cphd.cardinality());
    if (!cphd.extracted_mixtures().empty()) {
        YAML::Node extracts(YAML::NodeType::Sequence);
        for (const auto& mix : cphd.extracted_mixtures()) {
            extracts.push_back(mixture_to_yaml<T>(*mix));
        }
        j["extracted_mixtures"] = extracts;
    }
    return j;
}

// ============================================================
// GLMB<T>
// ============================================================

template <typename T>
YAML::Node glmb_to_yaml(const multi_target::GLMB<T>& glmb) {
    YAML::Node j;
    j["filter_type"] = "GLMB";
    j["prob_detection"] = glmb.prob_detection();
    j["prob_survive"] = glmb.prob_survive();
    j["clutter_rate"] = glmb.clutter_rate();
    j["clutter_density"] = glmb.clutter_density();
    j["estimated_cardinality"] = glmb.estimated_cardinality();
    j["cardinality_pmf"] = vector_to_yaml(glmb.cardinality());
    j["track_histories"] = track_histories_to_yaml(glmb.track_histories());
    // Global hypotheses
    YAML::Node hyps(YAML::NodeType::Sequence);
    for (const auto& h : glmb.global_hypotheses()) {
        YAML::Node hyp;
        hyp["log_weight"] = h.log_weight;
        hyp["bernoulli_indices"] = h.bernoulli_indices;
        hyps.push_back(hyp);
    }
    j["global_hypotheses"] = hyps;
    if (!glmb.extracted_mixtures().empty()) {
        YAML::Node extracts(YAML::NodeType::Sequence);
        for (const auto& mix : glmb.extracted_mixtures()) {
            extracts.push_back(mixture_to_yaml<T>(*mix));
        }
        j["extracted_mixtures"] = extracts;
    }
    return j;
}

// ============================================================
// JGLMB<T>
// ============================================================

template <typename T>
YAML::Node jglmb_to_yaml(const multi_target::JGLMB<T>& jglmb) {
    YAML::Node j;
    j["filter_type"] = "JGLMB";
    j["prob_detection"] = jglmb.prob_detection();
    j["prob_survive"] = jglmb.prob_survive();
    j["clutter_rate"] = jglmb.clutter_rate();
    j["clutter_density"] = jglmb.clutter_density();
    j["estimated_cardinality"] = jglmb.estimated_cardinality();
    j["cardinality_pmf"] = vector_to_yaml(jglmb.cardinality());
    j["track_histories"] = track_histories_to_yaml(jglmb.track_histories());
    YAML::Node hyps(YAML::NodeType::Sequence);
    for (const auto& h : jglmb.global_hypotheses()) {
        YAML::Node hyp;
        hyp["log_weight"] = h.log_weight;
        hyp["bernoulli_indices"] = h.bernoulli_indices;
        hyps.push_back(hyp);
    }
    j["global_hypotheses"] = hyps;
    if (!jglmb.extracted_mixtures().empty()) {
        YAML::Node extracts(YAML::NodeType::Sequence);
        for (const auto& mix : jglmb.extracted_mixtures()) {
            extracts.push_back(mixture_to_yaml<T>(*mix));
        }
        j["extracted_mixtures"] = extracts;
    }
    return j;
}

// ============================================================
// PMBM<T>
// ============================================================

template <typename T>
YAML::Node pmbm_to_yaml(const multi_target::PMBM<T>& pmbm) {
    YAML::Node j;
    j["filter_type"] = "PMBM";
    j["prob_detection"] = pmbm.prob_detection();
    j["prob_survive"] = pmbm.prob_survive();
    j["clutter_rate"] = pmbm.clutter_rate();
    j["clutter_density"] = pmbm.clutter_density();
    j["estimated_cardinality"] = pmbm.estimated_cardinality();
    j["cardinality_pmf"] = vector_to_yaml(pmbm.cardinality());
    j["poisson_intensity"] = mixture_to_yaml<T>(pmbm.poisson_intensity());
    j["track_histories"] = track_histories_to_yaml(pmbm.track_histories());
    YAML::Node hyps(YAML::NodeType::Sequence);
    for (const auto& h : pmbm.global_hypotheses()) {
        YAML::Node hyp;
        hyp["log_weight"] = h.log_weight;
        hyp["bernoulli_indices"] = h.bernoulli_indices;
        hyps.push_back(hyp);
    }
    j["global_hypotheses"] = hyps;
    if (!pmbm.extracted_mixtures().empty()) {
        YAML::Node extracts(YAML::NodeType::Sequence);
        for (const auto& mix : pmbm.extracted_mixtures()) {
            extracts.push_back(mixture_to_yaml<T>(*mix));
        }
        j["extracted_mixtures"] = extracts;
    }
    return j;
}

// ============================================================
// MBM<T>
// ============================================================

template <typename T>
YAML::Node mbm_to_yaml(const multi_target::MBM<T>& mbm) {
    YAML::Node j;
    j["filter_type"] = "MBM";
    j["prob_detection"] = mbm.prob_detection();
    j["prob_survive"] = mbm.prob_survive();
    j["clutter_rate"] = mbm.clutter_rate();
    j["clutter_density"] = mbm.clutter_density();
    j["track_histories"] = track_histories_to_yaml(mbm.track_histories());
    YAML::Node hyps(YAML::NodeType::Sequence);
    for (const auto& h : mbm.global_hypotheses()) {
        YAML::Node hyp;
        hyp["log_weight"] = h.log_weight;
        hyp["bernoulli_indices"] = h.bernoulli_indices;
        hyps.push_back(hyp);
    }
    j["global_hypotheses"] = hyps;
    if (!mbm.extracted_mixtures().empty()) {
        YAML::Node extracts(YAML::NodeType::Sequence);
        for (const auto& mix : mbm.extracted_mixtures()) {
            extracts.push_back(mixture_to_yaml<T>(*mix));
        }
        j["extracted_mixtures"] = extracts;
    }
    return j;
}

// ============================================================
// MB<T>
// ============================================================

template <typename T>
YAML::Node mb_to_yaml(const multi_target::MB<T>& mb) {
    YAML::Node j;
    j["filter_type"] = "MB";
    j["prob_detection"] = mb.prob_detection();
    j["prob_survive"] = mb.prob_survive();
    j["clutter_rate"] = mb.clutter_rate();
    j["clutter_density"] = mb.clutter_density();
    if (!mb.extracted_mixtures().empty()) {
        YAML::Node extracts(YAML::NodeType::Sequence);
        for (const auto& mix : mb.extracted_mixtures()) {
            extracts.push_back(mixture_to_yaml<T>(*mix));
        }
        j["extracted_mixtures"] = extracts;
    }
    return j;
}

// ============================================================
// LMB<T>
// ============================================================

template <typename T>
YAML::Node lmb_to_yaml(const multi_target::LMB<T>& lmb) {
    YAML::Node j;
    j["filter_type"] = "LMB";
    j["prob_detection"] = lmb.prob_detection();
    j["prob_survive"] = lmb.prob_survive();
    j["clutter_rate"] = lmb.clutter_rate();
    j["clutter_density"] = lmb.clutter_density();
    j["track_histories"] = track_histories_to_yaml(lmb.track_histories());
    if (!lmb.extracted_mixtures().empty()) {
        YAML::Node extracts(YAML::NodeType::Sequence);
        for (const auto& mix : lmb.extracted_mixtures()) {
            extracts.push_back(mixture_to_yaml<T>(*mix));
        }
        j["extracted_mixtures"] = extracts;
    }
    return j;
}

} // namespace brew::serialization
