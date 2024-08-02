#include <algorithm>
#include <assert.h>

#include <iostream>

#include "vbkioctree_node.h"

namespace vsemantic_bki {

    /// Default static values
    int Semantics::num_class = 2;
    float Semantics::sf2 = 1.0f;
    float Semantics::ell = 1.0f;
    float Semantics::prior = 0.5f;
    float Semantics::var_thresh = 1000.0f;
    float Semantics::free_thresh = 0.3f;
    float Semantics::occupied_thresh = 0.7f;

    void Semantics::get_probs(std::vector<float>& probs) const {
      assert (probs.size() == num_class);
      float sum = 0;
      for (auto m : ms)
        sum += m;
      for (int i = 0; i < num_class; ++i)
        probs[i] = ms[i] / sum;
    }

    void Semantics::get_vars(std::vector<float>& vars) const {
      assert (vars.size() == num_class);
      float sum = 0;
      for (auto m : ms){
        sum += m;
      }
      for (int i = 0; i < num_class; ++i){
        vars[i] = ((ms[i] / sum) - (ms[i] / sum) * (ms[i] / sum)) / (sum + 1);
      }
    }
    void Semantics::get_dempster_vars(std::vector<float>& vars) const {
      assert (vars.size() == num_class);
      float sum = 0;
      for (auto m : ms){
        sum += m;
      }
      for (int i = 0; i < num_class; ++i){
        vars[i] = 1.0 - sum;
      }
    }

    void Semantics::update(std::vector<float>& ybars) {
      assert(ybars.size() == num_class);
      classified = true;
      for (int i = 0; i < num_class; ++i)
        ms[i] += ybars[i];

      std::vector<float> probs(num_class);
      get_probs(probs);

      semantics = std::distance(probs.begin(), std::max_element(probs.begin(), probs.end()));

      // if (semantics == 12 || (semantics == 0 && ms[0] < 1.01))
      if (semantics == 0)
        state = State::FREE;
      else
        state = State::OCCUPIED;
    }

    void Semantics::update(std::vector<float>& ybars, double uncertainty, int data_number) {
      if(edl_unc_weight == 0){
        edl_unc_avg = uncertainty;
        edl_unc_weight = data_number;
      }else{
        edl_unc_avg = ((double) edl_unc_weight * edl_unc_avg + (double) data_number * uncertainty) / (double) (edl_unc_weight + data_number);
        edl_unc_weight += data_number;
      }

      update(ybars);
    }

    void Semantics::update_direct(std::vector<float>& ybars, double uncertainty, int data_number) {
      assert(ybars.size() == num_class);
      classified = true;

      dempster_combination(ybars, ms, 1);
      std::vector<float> probs(num_class);
      get_probs(probs);

      semantics = std::distance(probs.begin(), std::max_element(probs.begin(), probs.end()));

      if (semantics == 0)
        state = State::FREE;
      else
        state = State::OCCUPIED;
    }
}
