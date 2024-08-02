#ifndef VKJY_DEMPSTER_
#define VKJY_DEMPSTER_

#include <vector>
#include <iostream>

namespace vsemantic_bki {
    void dempster_combination(const std::vector<double> &a, std::vector<double> &result, int num);
    void dempster_combination(const std::vector<float> &a, std::vector<float> &result, int num);
}

#endif