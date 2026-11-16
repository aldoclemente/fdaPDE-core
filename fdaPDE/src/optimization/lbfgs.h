// This file is part of fdaPDE, a C++ library for physics-informed
// spatial and functional data analysis.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#ifndef __FDAPDE_LBFGS_H__
#define __FDAPDE_LBFGS_H__

#include "header_check.h"

namespace fdapde {

template <int N> class LBFGS {
   private:
    using vector_t = std::conditional_t<N == Dynamic, Eigen::Matrix<double, Dynamic, 1>, Eigen::Matrix<double, N, 1>>;
    using matrix_t =
      std::conditional_t<N == Dynamic, Eigen::Matrix<double, Dynamic, Dynamic>, Eigen::Matrix<double, N, N>>;

    vector_t optimum_;
    double value_;                 // objective value at optimum
    int n_iter_ = 0;               // current iteration number
    std::vector<double> values_;   // explored objective values during optimization

    int max_iter_;        // maximum number of iterations before forced stop
    double tol_;          // tolerance on error before forced stop
    double step_;         // update step
    int mem_size_ = 10;   // number of vector used for approximating the objective hessian
    Eigen::Matrix<double, Dynamic, Dynamic> grad_mem_, x_mem_;
   public:
    static constexpr bool gradient_free = false;
    static constexpr int static_input_size = N;
    vector_t x_old, x_new, update, grad_old, grad_new;
    double h;
    // constructors
    LBFGS() : max_iter_(500), tol_(1e-5), step_(1e-2) { }
    LBFGS(int max_iter, double tol, double step, int mem_size) :
        max_iter_(max_iter), tol_(tol), step_(step), mem_size_(mem_size) {
        fdapde_assert(mem_size_ > 0);
    }
    LBFGS(const LBFGS& other) :
        max_iter_(other.max_iter_), tol_(other.tol_), step_(other.step_), mem_size_(other.mem_size_) { }
    LBFGS& operator=(const LBFGS& other) {
        max_iter_ = other.max_iter_;
        tol_ = other.tol_;
        step_ = other.step_;
        mem_size_ = other.mem_size_;
        return *this;
    }
    template <typename ObjectiveT, typename... Callbacks>
    vector_t optimize(ObjectiveT&& objective, const vector_t& x0, Callbacks&&... callbacks) {
        fdapde_static_assert(
          std::is_same<decltype(std::declval<ObjectiveT>().operator()(vector_t())) FDAPDE_COMMA double>::value,
          INVALID_CALL_TO_OPTIMIZE__OBJECTIVE_FUNCTOR_NOT_CALLABLE_AT_VECTOR_TYPE);
        constexpr double NaN = std::numeric_limits<double>::quiet_NaN();
        constexpr double curvature_tol = 1e-12;
        std::tuple<Callbacks...> callbacks_ {callbacks...};
        bool stop = false;   // asserted true in case of forced stop
        double error = std::numeric_limits<double>::max();
        int n_pairs = 0;
        double gamma = 1.0;
        int size = N == Dynamic ? x0.rows() : N;
        auto grad = objective.gradient();
        h = step_;
        n_iter_ = 0;
        values_.clear();
        x_old = x0, x_new = vector_t::Constant(size, NaN);
        grad_old = grad(x_old), grad_new = vector_t::Constant(size, NaN);
        if (!grad_old.allFinite()) {
            throw std::runtime_error("LBFGS: non-finite initial gradient");
        }

        update = -grad_old;
        stop |= internals::exec_grad_hooks(*this, objective, callbacks_);
        error = grad_old.norm();
        values_.push_back(objective(x_old));
        x_mem_.resize(x0.rows(), mem_size_);
        grad_mem_.resize(x0.rows(), mem_size_);

        while (n_iter_ < max_iter_ && error > tol_ && !stop) {
            // Ensure descent direction
            const double directional_derivative = grad_old.dot(update);
             if (!update.allFinite() || !std::isfinite(directional_derivative) || directional_derivative >= 0.0) {
                // Restart L-BFGS
                n_pairs = 0;
                gamma = 1.0;
                update = -grad_old;
            }

            //Linesearch
            stop |= internals::exec_adapt_hooks(*this, objective, callbacks_);
            if (stop) break;

            if (!std::isfinite(h) || h <= 0.0) {
                throw std::runtime_error("LBFGS: invalid step size");
            }

            // update along descent direction
            x_new = x_old + h * update;
            grad_new = grad(x_new);

            if (!grad_new.allFinite()) {
                throw std::runtime_error("LBFGS: non-finite gradient");
            }

             // Curvature pair
            vector_t s = x_new - x_old;
            if (s.norm() == 0.0) {
                throw std::runtime_error("LBFGS: numerical stagnation");
            }
            vector_t y = grad_new - grad_old;
            const double sy = s.dot(y);
            const double yy = y.squaredNorm();
	        const double snorm = s.norm();
            const double ynorm = y.norm();
            
            const bool valid_pair = std::isfinite(sy) && std::isfinite(snorm) && std::isfinite(ynorm) && 
                                    snorm > 0.0 && ynorm > 0.0 && 
                                    sy > curvature_tol * snorm * ynorm;
            
            if (valid_pair) {
                // Circular buffer indexed by accepted pairs
                const int col_idx = n_pairs % mem_size_;

                x_mem_.col(col_idx) = s;
                grad_mem_.col(col_idx) = y;

                ++n_pairs;

                const double candidate = sy / yy;

                if (std::isfinite(candidate) && candidate > 0.0) { 
                    gamma = candidate;
                } else {
                    gamma = 1.0;
                }
            }
            
            int current_mem = std::min(n_pairs, mem_size_);
            
            // Two-loop recursion
            vector_t q = grad_new;
	        
            std::vector<double> alpha(current_mem, 0);
            // First loop
            for (int i = 0; i < current_mem; ++i) {
                int k = (n_pairs - i - 1) % mem_size_;
                alpha[i] = x_mem_.col(k).dot(q) / grad_mem_.col(k).dot(x_mem_.col(k));
                q -= alpha[i] * grad_mem_.col(k);
            }
            // Initial inverse Hessian approximation: H_0^k = gamma * I
            vector_t r = gamma * q;
            //Second loop
            for (int i = current_mem - 1; i >= 0; --i) {

                int k = (n_pairs - i - 1) % mem_size_;
                double beta = grad_mem_.col(k).dot(r) / grad_mem_.col(k).dot(x_mem_.col(k));
                r += x_mem_.col(k) * (alpha[i] - beta);
            }
            update = -r;


            const double dphi_new = grad_new.dot(update);
            if (!update.allFinite() || !std::isfinite(dphi_new) || dphi_new >= 0.0) {
                n_pairs = 0;
                gamma = 1.0;
                update = -grad_new;
            }

            ++n_iter_;

            stop |=
              (internals::exec_grad_hooks(*this, objective, callbacks_) || internals::exec_stop_if(*this, objective));
            x_old = x_new;
            grad_old = grad_new;
            error = grad_new.norm();
	        values_.push_back(objective(x_old));
        }
        optimum_ = x_old;
        value_ = values_.back();
        return optimum_;
    }
    // observers
    vector_t optimum() const { return optimum_; }
    double value() const { return value_; }
    int n_iter() const { return n_iter_; }

    const std::vector<double>& values() const { return values_; }
};

}   // namespace fdapde

#endif   // __FDAPDE_LBFGS_H__
