#pragma once
#include <iostream>
#include <cmath>

#include "vbkioctomap.h"

namespace vsemantic_bki {

	/*
     * @brief Bayesian Generalized Kernel Inference on Bernoulli distribution
     * @param dim dimension of data (2, 3, etc.)
     * @param T data type (float, double, etc.)
     * @ref Nonparametric Bayesian inference on multivariate exponential families
     */
    template<int dim, typename T, int n_class>
    class SemanticBKInference {
    public:
      /// Eigen matrix type for training and test data and kernel
      using MatrixXType = Eigen::Matrix<T, -1, dim, Eigen::RowMajor>;
      using MatrixKType = Eigen::Matrix<T, -1, -1, Eigen::RowMajor>;
      using MatrixDKType = Eigen::Matrix<T, -1, 1>;
      using MatrixYType = Eigen::Matrix<T, -1, n_class>;

      SemanticBKInference(int nc, T sf2, T ell) : nc(nc), sf2(sf2), ell(ell), trained(false) { }

      /*
        * @brief Fit BGK Model
        * @param x input vector (3N, row major)
        * @param y target vector (N)
        */
      void train(const std::vector<T> &x, const std::vector<T> &y) {
        // std::cout << x.size() << " x | y " << y.size() << std::endl;
        assert(x.size() % dim == 0 && (int) (x.size() / dim) == (int) (y.size() / nc));
        MatrixXType _x = Eigen::Map<const MatrixXType>(x.data(), x.size() / dim, dim);
        MatrixYType _y = Eigen::Map<const MatrixYType>(y.data(), y.size(), nc);
        this->y_vec = y;
        train(_x, _y);
      }

      /*
        * @brief Fit BGK Model
        * @param x input matrix (NX3)
        * @param y target matrix (NX1)
        */
      void train(const MatrixXType &x, const MatrixYType &y) {
          this->x = MatrixXType(x);
          this->y = MatrixYType(y);
          trained = true;
      }

      void predict_bki_max(const MatrixXType &_xs, std::vector<std::vector<T>> &ybars, const MatrixKType &Ks) {
        int M = _xs.rows(), N = y_vec.size() / n_class, K = n_class;

        MatrixDKType _y_vec = MatrixDKType::Ones(N, 1);
        for (int k = 0; k < K; ++k) {
          for (int i = 0; i < N; ++i) {
            // y_vec [i * n_class + 0 ~ i * n_class + n_class - 1] max?
            int max_idx = -1;
            double max_value = -9999;
            for(int kk = 0; kk < K; kk++){
              if(y_vec[i * K + kk] > max_value){
                max_idx = kk;
                max_value = y_vec[i * K + kk];
              }
            }
            if(max_idx == k){
              _y_vec(i, 0) = 1;
            }else{
              _y_vec(i, 0) = 0;
            }
          }        
          MatrixDKType _ybar;
          _ybar = (Ks * _y_vec);
          
          for (int r = 0; r < M; ++r)
            ybars[r][k] = _ybar(r, 0);
        }
      }

      void predict_bki_vector(const MatrixXType &_xs, std::vector<std::vector<T>> &ybars, const MatrixKType &Ks, int int_mapping_media, int int_unc_policy, float uncThres) {
        int M = _xs.rows(), N = y_vec.size() / n_class, K = n_class;

        MatrixDKType _y_vec = MatrixDKType::Ones(N, 1);
        for (int k = 0; k < K; ++k) {
          // mapping_media: alpha, prob, prob_hat
          get__y_vec(_y_vec, N, K, k, int_mapping_media, int_unc_policy, uncThres);
        
          MatrixDKType _ybar;
          _ybar = (Ks * _y_vec);
          
          for (int r = 0; r < M; ++r)
            ybars[r][k] = _ybar(r, 0);
        }
        // _y_vec: N x 1
        // Ks    : M x N
        // _ybar : M x 1
        // ybars : M x K (hstack of _ybar)
      }


      // Max
      void predict_csm_max(const MatrixXType &_xs, std::vector<std::vector<T>> &ybars, const MatrixKType &Ks) {
        int M = _xs.rows(), N = y_vec.size() / n_class, K = n_class;

        MatrixDKType _y_vec = MatrixDKType::Ones(N, 1);
        for (int k = 0; k < K; ++k) {
          for (int i = 0; i < N; ++i) {
            // y_vec [i * n_class + 0 ~ i * n_class + n_class - 1] max?
            int max_idx = -1;
            double max_value = -9999;
            for(int kk = 0; kk < K; kk++){
              if(y_vec[i * K + kk] > max_value){
                max_idx = kk;
                max_value = y_vec[i * K + kk];
              }
            }
            if(max_idx == k){
              _y_vec(i, 0) = 1;
            }else{
              _y_vec(i, 0) = 0;
            }
          }
        
          MatrixDKType _ybar;
          _ybar = (Ks * _y_vec);
          
          for (int r = 0; r < M; ++r)
            ybars[r][k] = _ybar(r, 0);
        }
      }

      // Vectorized
      void predict_csm_vector(const MatrixXType &_xs, std::vector<std::vector<T>> &ybars, const MatrixKType &Ks, int int_mapping_media, int int_unc_policy) {
        int M = _xs.rows(), N = y_vec.size() / n_class, K = n_class;

        MatrixDKType _y_vec = MatrixDKType::Ones(N, 1);
        for (int k = 0; k < K; ++k) {
          // mapping_media: alpha, prob, prob_hat
          get__y_vec(_y_vec, N, K, k, int_mapping_media, int_unc_policy, -1.0f);
        
          MatrixDKType _ybar;
          _ybar = (Ks * _y_vec);
          
          for (int r = 0; r < M; ++r)
            ybars[r][k] = _ybar(r, 0);
        }
      }
      
      void predict_csm(const std::vector<T> &xs, std::vector<std::vector<T>> &ybars, int int_mapping_media, int int_unc_policy){
        // 1. Prepare Step
        assert(xs.size() % dim == 0);
        MatrixXType _xs = Eigen::Map<const MatrixXType>(xs.data(), xs.size() / dim, dim);
        assert(trained == true);
        MatrixKType Ks;

        covCountingSensorModel(_xs, x, Ks);

        int M = _xs.rows(), N = y_vec.size() / n_class, K = n_class;

        ybars.resize(M);
        for (int r = 0; r < M; ++r)
          ybars[r].resize(K);
        
        // 2. Calculate ybars
        if(int_mapping_media == MAP_MEDIA_ONEHOT){
          predict_csm_max(_xs, ybars, Ks);
        }else{
          predict_csm_vector(_xs, ybars, Ks, int_mapping_media, int_unc_policy);
        }
      }

      void predict_bki(const std::vector<T> &xs, std::vector<std::vector<T>> &ybars, int int_mapping_media, int int_unc_policy, double &uncertainty, int &data_number, float uncThres){
        // 1. Prepare Step
        assert(xs.size() % dim == 0);
        MatrixXType _xs = Eigen::Map<const MatrixXType>(xs.data(), xs.size() / dim, dim);
        assert(trained == true);
        MatrixKType Ks;

        int M = _xs.rows(), N = y_vec.size() / n_class, K = n_class;

        if(int_unc_policy == MAP_UNC_POLICY_ADAPTIVE_KERNEL_LENGTH || int_unc_policy == MAP_UNC_POLICY_ADLEN){
          covSparseUnc(_xs, x, Ks);
        }else{
          covSparse(_xs, x, Ks);
        }

        ybars.resize(M);
        for (int r = 0; r < M; ++r)
          ybars[r].resize(K);
        
        // 2. Calculate ybars
        if(int_mapping_media == MAP_MEDIA_ONEHOT){
          predict_bki_max(_xs, ybars, Ks);
        }else{
          predict_bki_vector(_xs, ybars, Ks, int_mapping_media, int_unc_policy, uncThres);
        }

        // 3. Calculate uncertainty
        std::vector<T> unc_vec;
        get_uncertainty(unc_vec, N, K);    // For Evidential + Vanilla
        
        double sum = 0.0f;
        for(int i =0; i < N; i++)
          sum += unc_vec[i];
        uncertainty = sum / (double) N;
        data_number = N;
      }

      void predict_direct_vector(const std::vector<T> &xs, std::vector<std::vector<T>> &ybars, double &uncertainty, int &data_number) {
        assert(xs.size() % dim == 0);
        MatrixXType _xs = Eigen::Map<const MatrixXType>(xs.data(), xs.size() / dim, dim);
        assert(trained == true);
        MatrixKType Ks;
        int K = n_class, N = y_vec.size() / K, M = xs.size() / dim;
        covSparseUnc(_xs, x, Ks); 
        // covSparse(_xs, x, Ks); //woAdaKernel

        // 1. Make a Belief Matrix (N x K)
        std::vector<std::vector<double>> beliefs(N, std::vector<double>(K, 0.0));
        for(int n = 0; n < N; n++){
          double alpha0 = 0.0;
          for (int k = 0; k < K; ++k){
            alpha0 += y_vec[n * K + k];
          }
          for(int k = 0; k < K; k++){
            double belief = (y_vec[n * K + k] - 1.0) / alpha0; 
            beliefs[n][k] = belief > 0.0 ? (belief < 1.0 ? belief : 1.0 - 1e-8) : 0.0;
          }
        }

        // 2. Apply Sparse Kernel and Fuse
        // Perform element-wise multiplication (Hadamard product) between beliefs and the current row of Ks
        ybars.resize(M);
        for(int m = 0; m < M; m++){
          // m-th map cell : Compute kernel multiplied Belief ( N by K matrix ).
          std::vector<std::vector<double>> kernel_belief(N, std::vector<double>(K, 0.0));
          for(int n = 0; n < N; n++){
            for(int k = 0; k < K; k++){
              // kernel_belief[n][k] = beliefs[n][k] * 20 * Ks(m, n); // Apply the kernel
              double multiplier = Ks(m, n);
              double kernel_belief_val = beliefs[n][k] * (multiplier > 1.0 ? 1.0 : multiplier); // Apply the kernel
              kernel_belief[n][k] = kernel_belief_val > 0.0 ? (kernel_belief_val < 1.0 ? kernel_belief_val : 1.0 - 1e-8) : 0.0;
            }
          }
          // n kernel applied belief => fusion => 1 x K belief.
          std::vector<double> merged_belief(K, 0.0);
          for(int n = 0; n < N; n++){
            dempster_combination(kernel_belief[n], merged_belief, n);
          }
          // Result: ybars[m] - Fused Belief Vector Saved.
          ybars[m].resize(K);
          for(int k = 0; k < K; k++)
            ybars[m][k] = merged_belief[k];
        }
      }

    private:
        /*
         * @brief Compute Euclid distances between two vectors.
         * @param x input vector
         * @param z input vecotr
         * @return d distance matrix
         */
        void dist(const MatrixXType &x, const MatrixXType &z, MatrixKType &d) const {
            d = MatrixKType::Zero(x.rows(), z.rows());
            for (int i = 0; i < x.rows(); ++i) {
                d.row(i) = (z.rowwise() - x.row(i)).rowwise().norm();
            }
        }

        /*
         * @brief Sparse kernel.
         * @param x input vector
         * @param z input vector
         * @return Kxz covariance matrix
         * @ref A sparse covariance function for exact gaussian process inference in large datasets.
         */
        void covSparse(const MatrixXType &x, const MatrixXType &z, MatrixKType &Kxz) const {
            dist(x / ell, z / ell, Kxz);
            Kxz = (((2.0f + (Kxz * 2.0f * 3.1415926f).array().cos()) * (1.0f - Kxz.array()) / 3.0f) +
                  (Kxz * 2.0f * 3.1415926f).array().sin() / (2.0f * 3.1415926f)).matrix() * sf2;

            // Clean up for values with distance outside length scale
            // Possible because Kxz <= 0 when dist >= ell
            for (int i = 0; i < Kxz.rows(); ++i)
            {
                for (int j = 0; j < Kxz.cols(); ++j)
                    if (Kxz(i,j) < 0.0)
                        Kxz(i,j) = 0.0f;
            }
        }

        void distAdaptive(const MatrixXType &x, const MatrixXType &z, MatrixKType &d, std::vector<T> &uncertainty) const {
          // 1. Calculate Real Distance First
          d = MatrixKType::Zero(x.rows(), z.rows());
          for (int i = 0; i < x.rows(); ++i) {
            d.row(i) = (z.rowwise() - x.row(i)).rowwise().norm();
          }

          // 2. Divide By Adaptive Kernel Length
          for (int i = 0; i < x.rows(); i++){
            for (int j = 0; j < z.rows(); j++) {
              double a = 1.0; // 5, 6
              double b = -3.0; // fn2
              // double b = -6.0; // fn3
              // double b = -1.5; // fn4
              
              // double a = 0.5; // 5, 6
              // double b = -10.0; // 6

              double adaptive_ =  ( a * std::exp(1 + b * uncertainty[j]));
              // double adaptive = adaptive_ > 1.0 ? adaptive_ : 1.0; 
              // double adaptive = adaptive_; 
              double adaptive = std::exp(1.0); // fn5
              // double adaptive = 1.0;  // fn6
              d(i, j) = d(i, j) / (ell * adaptive);
              // d(i, j) = d(i, j) / (ell * ( - a * std::log(uncertainty[j] - b) + a * std::log(1 - b) + 0.01 ));
              // if ( uncertainty[j] < 0.14 )
              //   d(i, j) = d(i, j) / (ell * (1.5)); 
              // else
              //   d(i, j) = d(i, j) / ell; 
            }
          }
        }

        void covSparseUnc(const MatrixXType &x, const MatrixXType &z, MatrixKType &Kxz) const {
          // 1. Get uncertainty from training data
          int N = y_vec.size() / n_class, K = n_class;
          std::vector<T> uncertainty;
          get_uncertainty(uncertainty, N, K);

          // 2. Calculate Adaptive Distance
          distAdaptive(x, z, Kxz, uncertainty);
          Kxz = (
                  ((2.0f + (Kxz * 2.0f * 3.1415926f).array().cos()) * (1.0f - Kxz.array()) / 3.0f)
                  +
                  (Kxz * 2.0f * 3.1415926f).array().sin() / (2.0f * 3.1415926f)
            ).matrix() * sf2;
          
          for (int i = 0; i < Kxz.rows(); ++i)
          {
            for (int j = 0; j < Kxz.cols(); ++j)
              // 3. Clean up for values with distance outside length scale
              if (Kxz(i,j) < 0.0) // Possible because Kxz <= 0 when dist >= ell
                Kxz(i,j) = 0.0f;
              else { // 3. Adaptive Kernel Signal Vairance
                // Not Modifying \sigma_0 *************************************************

                // double a = 0.6; 
                // double b = -5.0;  
                // Kxz(i, j) = Kxz(i, j) * ( a * std::exp(1 + b * uncertainty[j])); 
                // if ( uncertainty[j] < 0.14 )
                //   Kxz(i, j) = Kxz(i, j); 
                // else
                //   Kxz(i, j) = Kxz(i, j) * (1 - uncertainty[j]); 
              }
          }
        }

        void covCountingSensorModel(const MatrixXType &x, const MatrixXType &z, MatrixKType &Kxz) const {
          Kxz = MatrixKType::Ones(x.rows(), z.rows());
        }

        // Utilities ------------------------------------------------------------------------------------------------------------
        void get_uncertainty_from_evidential(std::vector<T> &unc, int N, int K) const {
          std::vector<T> alpha0;
          T max_alpha0 = get_alpha0(alpha0, N, K);

          for(int i = 0; i<N; i++){
            unc[i] = K / alpha0[i];
          }
        }

        void get_uncertainty_from_vanilla(std::vector<T> &unc, int N, int K) const {
          for (int i=0; i<N; i++){
            T maxValue = -99999;
            for (int j=0; j<K; j++){
              if ( y_vec[i * K + j] > maxValue )
                maxValue = y_vec[i * K + j];
            }
            unc[i] = 1.0f - maxValue;
          }
        }

        void get_uncertainty(std::vector<T> &unc, int N, int K) const {
          unc.resize(N);

          if(y_vec[0] < 1.0){
            get_uncertainty_from_vanilla(unc, N, K);
          }else{
            get_uncertainty_from_evidential(unc, N, K);
          }
        }

        void get_label(std::vector<int> &label, int N, int K) const {
          label.resize(N);

          for (int i=0; i<N; i++){
            T maxValue = -99999;
            int maxIndex = -1;
            for (int j=0; j<K; j++){
              if ( y_vec[i * K + j] > maxValue ){
                maxValue = y_vec[i * K + j];
                maxIndex = j;
              }
            }
            label[i] = maxIndex;
          }
        }

        // training_data(y_vec) => alpha0.
        T get_alpha0(std::vector<T> &alpha0, int N, int K) const {
          alpha0.resize(N);
          
          T maxValue = std::numeric_limits<T>::lowest();
          for (int i = 0; i<N; i++){
            T sum = 0.0;

            for(int j = 0; j < K; j++){
              sum += y_vec[i * K + j];
            }

            alpha0[i] = sum;
            if( alpha0[i] > maxValue )
              maxValue = alpha0[i];
          }
          return maxValue;
        }

        // Certain Class k, training data(y_vec) => _y_vec.
        void get__y_vec(MatrixDKType &_y_vec, int N, int K, int k, int int_mapping_media, int int_unc_policy, float uncThres){
          std::vector<T> alpha0;
          T max_alpha0 = get_alpha0(alpha0, N, K);

          // 1. Calculate appropriate media
          if(int_mapping_media == MAP_MEDIA_ALPHA){
            for (int i=0; i < N; i++){
              _y_vec(i, 0) = y_vec[i * K + k];
            }
          }else if(int_mapping_media == MAP_MEDIA_PROB){
            for (int i=0; i < N; i++){
              _y_vec(i, 0) = y_vec[i * K + k] / alpha0[i];
            }
          }else{
            std::cout << "No way @ get__y_vec\n";
          }

          std::vector<float> uncertainties;
          uncertainties.resize(N);
          for (int i=0; i<N; i++){
            double unc = (double) K / alpha0[i];
            uncertainties[i] = unc;
          }
        }

        // Properties ------------------------------------------------------------------------------------------------------------
        T sf2;    // signal variance
        T ell;    // length-scale
        int nc;   // number of classes

        MatrixXType x;   // temporary storage of training data
        MatrixYType y;   // temporary storage of training labels
        std::vector<T> y_vec;

        bool trained;    // true if bgkinference stored training data
    };

    typedef SemanticBKInference<3, float, NUM_CLASS_PCD> SemanticBKI3f;
}
