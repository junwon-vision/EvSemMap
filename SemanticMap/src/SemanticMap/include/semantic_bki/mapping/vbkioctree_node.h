#pragma once

#include <vector>

#include "dempster.h"

namespace vsemantic_bki {

    /// Occupancy state: before pruning: FREE, OCCUPIED, UNKNOWN; after pruning: PRUNED
    enum class State : char {
        FREE, OCCUPIED, UNKNOWN, PRUNED
    };

    /*
     * @brief Inference ouputs and occupancy state.
     *
     * Occupancy has member variables: m_A and m_B (kernel densities of positive
     * and negative class, respectively) and State.
     * Before using this class, set the static member variables first.
     */
    class Semantics {
      
      friend class SemanticBKIOctoMap;

    public:
        /*
         * @brief Constructors and destructor.
         */
        Semantics() : ms(std::vector<float>(num_class, prior)), state(State::UNKNOWN), edl_unc_avg(0.0), edl_unc_weight(0) { classified = false; }

        Semantics(const Semantics &other) : ms(other.ms), state(other.state), semantics(other.semantics), edl_unc_avg(other.edl_unc_avg), edl_unc_weight(other.edl_unc_weight) { }

        Semantics &operator=(const Semantics &other) {
          ms = other.ms;
          state = other.state;
          semantics = other.semantics;
          edl_unc_avg = other.edl_unc_avg;
          edl_unc_weight = other.edl_unc_weight;
          return *this;
        }

        ~Semantics() { }

        /*
         * @brief Exact updates for nonparametric Bayesian kernel inference
         * @param ybar kernel density estimate of positive class (occupied)
         * @param kbar kernel density of negative class (unoccupied)
         */
        void update(std::vector<float>& ybars);
        void update(std::vector<float>& ybars, double uncertainty, int data_number);
        
        void update_direct(std::vector<float>& ybars, double uncertainty, int data_number);

        /// Get probability of occupancy.
        void get_probs(std::vector<float>& probs) const;

        /// Get variance of occupancy (uncertainty)
	      void get_vars(std::vector<float>& vars) const;

        /// Get variance of occupancy (uncertainty)
	      void get_dempster_vars(std::vector<float>& vars) const;
        
        /*
         * @brief Get occupancy state of the node.
         * @return occupancy state (see State).
         */
        inline State get_state() const { return state; }

        inline int get_semantics() const { return semantics; }

        inline double get_uncertainty() const { return edl_unc_avg; }

        bool classified;

    private:
        std::vector<float> ms;
        State state;
        int semantics;
        double edl_unc_avg;       // average uncertainty
        int edl_unc_weight;
        static int num_class;     // number of classes
        
        static float sf2;
        static float ell;   // length-scale
        static float prior;  // prior on each class

        static float var_thresh;
        static float free_thresh;     // FREE occupancy threshold
        static float occupied_thresh; // OCCUPIED occupancy threshold
    };

    typedef Semantics SemanticOcTreeNode;
}
