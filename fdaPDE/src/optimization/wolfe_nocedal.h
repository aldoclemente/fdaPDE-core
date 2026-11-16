
#ifndef __FDAPDE_WOLFE_LINE_SEARCH_NOCEDAL_H__
#define __FDAPDE_WOLFE_LINE_SEARCH_NOCEDAL_H__

#include "header_check.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace fdapde {

class WolfeLineSearchNocedal {
   private:
    static constexpr int max_iter_ = 30;
    static constexpr int max_zoom_iter_ = 40;

    double alpha_ = 1.0;
    double alpha_max_ = 1e6;
    double c1_ = 1e-4;
    double c2_ = 0.9;

   public:
    WolfeLineSearchNocedal() = default;

    WolfeLineSearchNocedal(double alpha, double c1, double c2)
        : alpha_(alpha), c1_(c1), c2_(c2) { }

    template <typename Opt, typename Obj>
    bool adapt_hook(Opt& opt, Obj& obj) {
        fdapde_static_assert(
            is_gradient_based_opt_v<Opt>,
            THIS_METHOD_IS_FOR_GRADIENT_BASED_OPTIMIZATION_ONLY
        );

        if (!(alpha_ > 0.0) || 
            !std::isfinite(alpha_) ||
            !std::isfinite(alpha_max_) ||
            !(alpha_max_ >= alpha_) ||
            !(0.0 < c1_ && c1_ < c2_ && c2_ < 1.0)) {
            throw std::invalid_argument(
                "WolfeLineSearch: invalid parameters"
            );
        }

        const double phi0 = obj(opt.x_old);
        const double dphi0 = opt.grad_old.dot(opt.update);

        if (!std::isfinite(phi0) ||
            !std::isfinite(dphi0) || dphi0 >= 0.0) {
            throw std::runtime_error(
                "WolfeLineSearch: invalid descent direction"
            );
        }

        auto gradient = obj.gradient();

        auto phi = [&](double a) {
            return obj(opt.x_old + a * opt.update);
        };

        auto dphi = [&](double a) {
            return gradient(opt.x_old + a * opt.update)
                .dot(opt.update);
        };

        auto armijo = [&](double a, double f) {
            return std::isfinite(f) &&
                   f <= phi0 + c1_ * a * dphi0;
        };

        auto curvature = [&](double df) {
            return std::isfinite(df) &&
                   std::abs(df) <= -c2_ * dphi0;
        };

        // Zoom over a bracket satisfying the conditions
        // of Algorithm 3.6 (Nocedal & Wright).
        // Endpoints are not necessarily numerically ordered.
        // The lo endpoint satisfies the Armijo condition.
        auto zoom = [&](double lo, double hi,
                        double phi_lo) -> double {
            for (int j = 0; j < max_zoom_iter_; ++j) {
                const double a = lo + 0.5 * (hi - lo);

                if (a == lo || a == hi) {
                    break;
                }

                const double f = phi(a);

                if (!armijo(a, f) || f >= phi_lo) {
                    hi = a;
                    continue;
                }

                const double df = dphi(a);

                if (curvature(df)) {
                    return a;
                }

                if (!std::isfinite(df)) {
                    hi = a;
                    continue;
                }

                if (df * (hi - lo) >= 0.0) {
                    hi = lo;
                }

                lo = a;
                phi_lo = f;
            }

            throw std::runtime_error(
                "WolfeLineSearch: zoom failed"
            );
        };

        auto accept = [&](double a) {
          const double f = phi(a);
          const double df = dphi(a);

          if (!armijo(a, f) || !curvature(df)) {
              throw std::runtime_error(
                  "WolfeLineSearch: Wolfe conditions violated"
              );
          }
      
          opt.h = a;
        };

        double alpha_prev = 0.0;
        double phi_prev = phi0;
        double alpha = alpha_;

        for (int i = 0; i < max_iter_; ++i) {
            const double f = phi(alpha);

            if (!armijo(alpha, f) ||
                (i > 0 && f >= phi_prev)) {
                accept(zoom(alpha_prev, alpha, phi_prev));
                return false;
            }

            const double df = dphi(alpha);

            if (curvature(df)) {
                accept(alpha);
                return false;
            }

            if (!std::isfinite(df)) {
                accept(zoom(alpha_prev, alpha, phi_prev));
                return false;
            }

            if (df >= 0.0) {
                accept(zoom(alpha, alpha_prev, f));
                return false;
            }

            alpha_prev = alpha;
            phi_prev = f;

            const double next = std::min(2.0 * alpha, alpha_max_);

            if (next <= alpha) {
                break;
            }

            alpha = next;
        }

        throw std::runtime_error(
            "WolfeLineSearch: bracketing failed"
        );
    }
};

}  // namespace fdapde

#endif
