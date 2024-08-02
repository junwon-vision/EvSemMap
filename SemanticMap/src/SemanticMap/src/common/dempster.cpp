#include "dempster.h"

namespace vsemantic_bki {
    void dempster_combination(const std::vector<double> &a, std::vector<double> &result, int num){
        if(num == 0){
            // First Merge
            std::copy( a.begin(), a.end(), result.begin() );
        }else{
            // Second~ Merge. Merge a, result => result.
            int num_class = a.size();
            std::vector<double> b;
            b.resize(num_class);
            std::copy( result.begin(), result.end(), b.begin() );
            
            // 1. Compute C
            double C = 0.0;
            for(int i = 0; i < num_class; i++){
                for(int j = 0; j < num_class; j++){
                if( i == j )
                    continue;
                else
                    C += a[i] * b[j];
                }
            }

            // 2. Normalize Factor
            double NormFactor = 1.0 - C;

            double S1 = 0.0;
            double S2 = 0.0;
            for(int i = 0 ; i < num_class; i++){
                S1 += a[i];
                S2 += b[i];
            }
            // if(S1 > 1.0 || S2 > 1.0){
            //     std::cout << "WTF!!\n" << S1 << " " << S2 << std::endl;
            // }
            double u1 = 1.0 - S1;
            double u2 = 1.0 - S2;

            // 3. Let's Merge!
            double res_sum = 0.0;
            for(int i = 0; i < num_class; i++){
                double res_temp = (a[i] * b[i] + a[i] * u2 + b[i] * u1) / NormFactor;
                result[i] = res_temp > 0.0 ? (res_temp < 1.0 ? res_temp : 1.0 - 1e-8) : 0.0;
                res_sum += result[i];
            }
            // if(res_sum >= 1.0){
            //     std::cout << "overflow belief! " << res_sum << std::endl;
            // }
        }
    };
    void dempster_combination(const std::vector<float> &a, std::vector<float> &result, int num){
        std::vector<double> a_double(a.begin(), a.end());
        std::vector<double> result_double(result.begin(), result.end());
        dempster_combination(a_double, result_double, num);
        for(int i = 0; i < result_double.size(); i++){
            result[i] = result_double[i];
        }
    };
}