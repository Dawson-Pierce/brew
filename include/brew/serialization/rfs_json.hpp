#pragma once

#include <nlohmann/json.hpp>
#include <Eigen/Dense>

#include "brew/models/gaussian.hpp"
#include "brew/models/ggiw.hpp"
#include "brew/models/trajectory_gaussian.hpp"
#include "brew/models/trajectory_ggiw.hpp"
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

using json = nlohmann::json;

// ============================================================
// Eigen helpers
// ============================================================

inline json vector_to_json(const Eigen::VectorXd& v) {
    json arr = json::array();
    for (Eigen::Index i = 0; i < v.size(); ++i) {
        arr.push_back(v(i));
    }
    return arr;
}

inline Eigen::VectorXd vector_from_json(const json& j) {
    Eigen::VectorXd v(j.size());
    for (std::size_t i = 0; i < j.size(); ++i) {
        v(static_cast<Eigen::Index>(i)) = j[i].get<double>();
    }
    return v;
}

inline json matrix_to_json(const Eigen::MatrixXd& m) {
    json obj;
    obj["rows"] = m.rows();
    obj["cols"] = m.cols();
    json data = json::array();
    for (Eigen::Index i = 0; i < m.rows(); ++i) {
        for (Eigen::Index j = 0; j < m.cols(); ++j) {
            data.push_back(m(i, j));
        }
    }
    obj["data"] = std::move(data);
    return obj;
}

inline Eigen::MatrixXd matrix_from_json(const json& j) {
    int rows = j["rows"].get<int>();
    int cols = j["cols"].get<int>();
    Eigen::MatrixXd m(rows, cols);
    const auto& data = j["data"];
    int idx = 0;
    for (int i = 0; i < rows; ++i) {
        for (int jj = 0; jj < cols; ++jj) {
            m(i, jj) = data[idx++].get<double>();
        }
    }
    return m;
}

// ============================================================
// Gaussian
// ============================================================

inline json to_json(const models::Gaussian& g) {
    json j;
    j["type"] = "Gaussian";
    j["mean"] = vector_to_json(g.mean());
    j["covariance"] = matrix_to_json(g.covariance());
    return j;
}

inline models::Gaussian gaussian_from_json(const json& j) {
    return models::Gaussian(
        vector_from_json(j["mean"]),
        matrix_from_json(j["covariance"])
    );
}

// ============================================================
// GGIW
// ============================================================

inline json to_json(const models::GGIW& g) {
    json j;
    j["type"] = "GGIW";
    j["mean"] = vector_to_json(g.mean());
    j["covariance"] = matrix_to_json(g.covariance());
    j["alpha"] = g.alpha();
    j["beta"] = g.beta();
    j["v"] = g.v();
    j["V"] = matrix_to_json(g.V());
    return j;
}

inline models::GGIW ggiw_from_json(const json& j) {
    return models::GGIW(
        vector_from_json(j["mean"]),
        matrix_from_json(j["covariance"]),
        j["alpha"].get<double>(),
        j["beta"].get<double>(),
        j["v"].get<double>(),
        matrix_from_json(j["V"])
    );
}

// ============================================================
// TrajectoryGaussian
// ============================================================

inline json to_json(const models::TrajectoryGaussian& g) {
    json j;
    j["type"] = "TrajectoryGaussian";
    j["init_idx"] = g.init_idx;
    j["state_dim"] = g.state_dim;
    j["mean"] = vector_to_json(g.mean());
    j["covariance"] = matrix_to_json(g.covariance());
    return j;
}

inline models::TrajectoryGaussian trajectory_gaussian_from_json(const json& j) {
    return models::TrajectoryGaussian(
        j["init_idx"].get<int>(),
        j["state_dim"].get<int>(),
        vector_from_json(j["mean"]),
        matrix_from_json(j["covariance"])
    );
}

// ============================================================
// TrajectoryGGIW
// ============================================================

inline json to_json(const models::TrajectoryGGIW& g) {
    json j;
    j["type"] = "TrajectoryGGIW";
    j["init_idx"] = g.init_idx;
    j["state_dim"] = g.state_dim;
    j["mean"] = vector_to_json(g.mean());
    j["covariance"] = matrix_to_json(g.covariance());
    j["alpha"] = g.alpha();
    j["beta"] = g.beta();
    j["v"] = g.v();
    j["V"] = matrix_to_json(g.V());
    return j;
}

inline models::TrajectoryGGIW trajectory_ggiw_from_json(const json& j) {
    return models::TrajectoryGGIW(
        j["init_idx"].get<int>(),
        j["state_dim"].get<int>(),
        vector_from_json(j["mean"]),
        matrix_from_json(j["covariance"]),
        j["alpha"].get<double>(),
        j["beta"].get<double>(),
        j["v"].get<double>(),
        matrix_from_json(j["V"])
    );
}

// ============================================================
// Generic distribution from_json dispatcher
// ============================================================

template <typename T>
struct DistributionSerializer;

template <>
struct DistributionSerializer<models::Gaussian> {
    static json serialize(const models::Gaussian& d) { return to_json(d); }
    static models::Gaussian deserialize(const json& j) { return gaussian_from_json(j); }
};

template <>
struct DistributionSerializer<models::GGIW> {
    static json serialize(const models::GGIW& d) { return to_json(d); }
    static models::GGIW deserialize(const json& j) { return ggiw_from_json(j); }
};

template <>
struct DistributionSerializer<models::TrajectoryGaussian> {
    static json serialize(const models::TrajectoryGaussian& d) { return to_json(d); }
    static models::TrajectoryGaussian deserialize(const json& j) { return trajectory_gaussian_from_json(j); }
};

template <>
struct DistributionSerializer<models::TrajectoryGGIW> {
    static json serialize(const models::TrajectoryGGIW& d) { return to_json(d); }
    static models::TrajectoryGGIW deserialize(const json& j) { return trajectory_ggiw_from_json(j); }
};

// ============================================================
// Mixture<T>
// ============================================================

template <typename T>
json mixture_to_json(const models::Mixture<T>& mix) {
    json j;
    j["weights"] = vector_to_json(mix.weights());
    json comps = json::array();
    for (std::size_t i = 0; i < mix.size(); ++i) {
        comps.push_back(DistributionSerializer<T>::serialize(mix.component(i)));
    }
    j["components"] = std::move(comps);
    return j;
}

template <typename T>
std::unique_ptr<models::Mixture<T>> mixture_from_json(const json& j) {
    auto mix = std::make_unique<models::Mixture<T>>();
    auto weights = vector_from_json(j["weights"]);
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
json bernoulli_to_json(const models::Bernoulli<T>& b) {
    json j;
    j["existence_probability"] = b.existence_probability();
    j["id"] = b.id();
    if (b.has_distribution()) {
        j["distribution"] = DistributionSerializer<T>::serialize(b.distribution());
    }
    return j;
}

template <typename T>
std::unique_ptr<models::Bernoulli<T>> bernoulli_from_json(const json& j) {
    double r = j["existence_probability"].get<double>();
    int id = j["id"].get<int>();
    std::unique_ptr<T> dist;
    if (j.contains("distribution")) {
        auto d = DistributionSerializer<T>::deserialize(j["distribution"]);
        dist = std::make_unique<T>(std::move(d));
    }
    return std::make_unique<models::Bernoulli<T>>(r, std::move(dist), id);
}

// ============================================================
// Track histories helper
// ============================================================

inline json track_histories_to_json(const std::map<int, std::vector<Eigen::VectorXd>>& histories) {
    json j = json::object();
    for (const auto& [id, states] : histories) {
        json arr = json::array();
        for (const auto& s : states) {
            arr.push_back(vector_to_json(s));
        }
        j[std::to_string(id)] = std::move(arr);
    }
    return j;
}

inline std::map<int, std::vector<Eigen::VectorXd>> track_histories_from_json(const json& j) {
    std::map<int, std::vector<Eigen::VectorXd>> result;
    for (auto it = j.begin(); it != j.end(); ++it) {
        int id = std::stoi(it.key());
        std::vector<Eigen::VectorXd> states;
        for (const auto& s : it.value()) {
            states.push_back(vector_from_json(s));
        }
        result[id] = std::move(states);
    }
    return result;
}

// ============================================================
// RFS Base config (common to all filters)
// ============================================================

inline json rfs_base_to_json(const multi_target::RFSBase& rfs) {
    json j;
    // RFSBase members are protected, so we access them via the public setters
    // by serializing from the concrete filter types instead.
    // This function is a placeholder for common structure.
    return j;
}

// ============================================================
// PHD<T>
// ============================================================

template <typename T>
json phd_to_json(const multi_target::PHD<T>& phd) {
    json j;
    j["filter_type"] = "PHD";
    j["prob_detection"] = phd.prob_detection();
    j["prob_survive"] = phd.prob_survive();
    j["clutter_rate"] = phd.clutter_rate();
    j["clutter_density"] = phd.clutter_density();
    j["intensity"] = mixture_to_json<T>(phd.intensity());
    if (!phd.extracted_mixtures().empty()) {
        json extracts = json::array();
        for (const auto& mix : phd.extracted_mixtures()) {
            extracts.push_back(mixture_to_json<T>(*mix));
        }
        j["extracted_mixtures"] = std::move(extracts);
    }
    return j;
}

// ============================================================
// CPHD<T>
// ============================================================

template <typename T>
json cphd_to_json(const multi_target::CPHD<T>& cphd) {
    json j;
    j["filter_type"] = "CPHD";
    j["prob_detection"] = cphd.prob_detection();
    j["prob_survive"] = cphd.prob_survive();
    j["clutter_rate"] = cphd.clutter_rate();
    j["clutter_density"] = cphd.clutter_density();
    j["intensity"] = mixture_to_json<T>(cphd.intensity());
    j["cardinality"] = vector_to_json(cphd.cardinality());
    if (!cphd.extracted_mixtures().empty()) {
        json extracts = json::array();
        for (const auto& mix : cphd.extracted_mixtures()) {
            extracts.push_back(mixture_to_json<T>(*mix));
        }
        j["extracted_mixtures"] = std::move(extracts);
    }
    return j;
}

// ============================================================
// GLMB<T>
// ============================================================

template <typename T>
json glmb_to_json(const multi_target::GLMB<T>& glmb) {
    json j;
    j["filter_type"] = "GLMB";
    j["prob_detection"] = glmb.prob_detection();
    j["prob_survive"] = glmb.prob_survive();
    j["clutter_rate"] = glmb.clutter_rate();
    j["clutter_density"] = glmb.clutter_density();
    j["estimated_cardinality"] = glmb.estimated_cardinality();
    j["cardinality_pmf"] = vector_to_json(glmb.cardinality());
    j["track_histories"] = track_histories_to_json(glmb.track_histories());
    // Global hypotheses
    json hyps = json::array();
    for (const auto& h : glmb.global_hypotheses()) {
        json hyp;
        hyp["log_weight"] = h.log_weight;
        hyp["bernoulli_indices"] = h.bernoulli_indices;
        hyps.push_back(std::move(hyp));
    }
    j["global_hypotheses"] = std::move(hyps);
    if (!glmb.extracted_mixtures().empty()) {
        json extracts = json::array();
        for (const auto& mix : glmb.extracted_mixtures()) {
            extracts.push_back(mixture_to_json<T>(*mix));
        }
        j["extracted_mixtures"] = std::move(extracts);
    }
    return j;
}

// ============================================================
// JGLMB<T>
// ============================================================

template <typename T>
json jglmb_to_json(const multi_target::JGLMB<T>& jglmb) {
    json j;
    j["filter_type"] = "JGLMB";
    j["prob_detection"] = jglmb.prob_detection();
    j["prob_survive"] = jglmb.prob_survive();
    j["clutter_rate"] = jglmb.clutter_rate();
    j["clutter_density"] = jglmb.clutter_density();
    j["estimated_cardinality"] = jglmb.estimated_cardinality();
    j["cardinality_pmf"] = vector_to_json(jglmb.cardinality());
    j["track_histories"] = track_histories_to_json(jglmb.track_histories());
    json hyps = json::array();
    for (const auto& h : jglmb.global_hypotheses()) {
        json hyp;
        hyp["log_weight"] = h.log_weight;
        hyp["bernoulli_indices"] = h.bernoulli_indices;
        hyps.push_back(std::move(hyp));
    }
    j["global_hypotheses"] = std::move(hyps);
    if (!jglmb.extracted_mixtures().empty()) {
        json extracts = json::array();
        for (const auto& mix : jglmb.extracted_mixtures()) {
            extracts.push_back(mixture_to_json<T>(*mix));
        }
        j["extracted_mixtures"] = std::move(extracts);
    }
    return j;
}

// ============================================================
// PMBM<T>
// ============================================================

template <typename T>
json pmbm_to_json(const multi_target::PMBM<T>& pmbm) {
    json j;
    j["filter_type"] = "PMBM";
    j["prob_detection"] = pmbm.prob_detection();
    j["prob_survive"] = pmbm.prob_survive();
    j["clutter_rate"] = pmbm.clutter_rate();
    j["clutter_density"] = pmbm.clutter_density();
    j["estimated_cardinality"] = pmbm.estimated_cardinality();
    j["cardinality_pmf"] = vector_to_json(pmbm.cardinality());
    j["poisson_intensity"] = mixture_to_json<T>(pmbm.poisson_intensity());
    j["track_histories"] = track_histories_to_json(pmbm.track_histories());
    json hyps = json::array();
    for (const auto& h : pmbm.global_hypotheses()) {
        json hyp;
        hyp["log_weight"] = h.log_weight;
        hyp["bernoulli_indices"] = h.bernoulli_indices;
        hyps.push_back(std::move(hyp));
    }
    j["global_hypotheses"] = std::move(hyps);
    if (!pmbm.extracted_mixtures().empty()) {
        json extracts = json::array();
        for (const auto& mix : pmbm.extracted_mixtures()) {
            extracts.push_back(mixture_to_json<T>(*mix));
        }
        j["extracted_mixtures"] = std::move(extracts);
    }
    return j;
}

// ============================================================
// MBM<T>
// ============================================================

template <typename T>
json mbm_to_json(const multi_target::MBM<T>& mbm) {
    json j;
    j["filter_type"] = "MBM";
    j["prob_detection"] = mbm.prob_detection();
    j["prob_survive"] = mbm.prob_survive();
    j["clutter_rate"] = mbm.clutter_rate();
    j["clutter_density"] = mbm.clutter_density();
    j["track_histories"] = track_histories_to_json(mbm.track_histories());
    json hyps = json::array();
    for (const auto& h : mbm.global_hypotheses()) {
        json hyp;
        hyp["log_weight"] = h.log_weight;
        hyp["bernoulli_indices"] = h.bernoulli_indices;
        hyps.push_back(std::move(hyp));
    }
    j["global_hypotheses"] = std::move(hyps);
    if (!mbm.extracted_mixtures().empty()) {
        json extracts = json::array();
        for (const auto& mix : mbm.extracted_mixtures()) {
            extracts.push_back(mixture_to_json<T>(*mix));
        }
        j["extracted_mixtures"] = std::move(extracts);
    }
    return j;
}

// ============================================================
// MB<T>
// ============================================================

template <typename T>
json mb_to_json(const multi_target::MB<T>& mb) {
    json j;
    j["filter_type"] = "MB";
    j["prob_detection"] = mb.prob_detection();
    j["prob_survive"] = mb.prob_survive();
    j["clutter_rate"] = mb.clutter_rate();
    j["clutter_density"] = mb.clutter_density();
    if (!mb.extracted_mixtures().empty()) {
        json extracts = json::array();
        for (const auto& mix : mb.extracted_mixtures()) {
            extracts.push_back(mixture_to_json<T>(*mix));
        }
        j["extracted_mixtures"] = std::move(extracts);
    }
    return j;
}

// ============================================================
// LMB<T>
// ============================================================

template <typename T>
json lmb_to_json(const multi_target::LMB<T>& lmb) {
    json j;
    j["filter_type"] = "LMB";
    j["prob_detection"] = lmb.prob_detection();
    j["prob_survive"] = lmb.prob_survive();
    j["clutter_rate"] = lmb.clutter_rate();
    j["clutter_density"] = lmb.clutter_density();
    j["track_histories"] = track_histories_to_json(lmb.track_histories());
    if (!lmb.extracted_mixtures().empty()) {
        json extracts = json::array();
        for (const auto& mix : lmb.extracted_mixtures()) {
            extracts.push_back(mixture_to_json<T>(*mix));
        }
        j["extracted_mixtures"] = std::move(extracts);
    }
    return j;
}

} // namespace brew::serialization
