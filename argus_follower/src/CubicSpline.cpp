#include <argus_follower/CubicSpline.hpp>

#include <algorithm>
#include <stdexcept>

namespace argus_follower {

CubicSpline::CubicSpline(std::vector<double> x, std::vector<double> y) : x_(std::move(x))
{
    if (x_.size() != y.size() || x_.size() < 2) {
        throw std::invalid_argument("CubicSpline: need matching x/y with at least 2 points");
    }
    const std::size_t n = x_.size();
    std::vector<double> h(n - 1);
    for (std::size_t i = 0; i < n - 1; ++i) {
        h[i] = x_[i + 1] - x_[i];
        if (h[i] <= 0.0) {
            throw std::invalid_argument("CubicSpline: x must be strictly increasing");
        }
    }

    // Natural cubic spline: solve the tridiagonal system for the second
    // derivatives m_i (m_0 = m_{n-1} = 0) via the Thomas algorithm.
    std::vector<double> m(n, 0.0);
    if (n > 2) {
        const std::size_t ni = n - 2;  // number of interior unknowns
        std::vector<double> sub(ni), diag(ni), sup(ni), rhs(ni);
        for (std::size_t i = 1; i < n - 1; ++i) {
            const std::size_t k = i - 1;
            sub[k]  = h[i - 1];
            diag[k] = 2.0 * (h[i - 1] + h[i]);
            sup[k]  = h[i];
            rhs[k]  = 6.0 * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1]);
        }
        for (std::size_t k = 1; k < ni; ++k) {
            const double factor = sub[k] / diag[k - 1];
            diag[k] -= factor * sup[k - 1];
            rhs[k]  -= factor * rhs[k - 1];
        }
        std::vector<double> inner(ni);
        inner[ni - 1] = rhs[ni - 1] / diag[ni - 1];
        for (long k = static_cast<long>(ni) - 2; k >= 0; --k) {
            const std::size_t uk = static_cast<std::size_t>(k);
            inner[uk] = (rhs[uk] - sup[uk] * inner[uk + 1]) / diag[uk];
        }
        for (std::size_t i = 1; i < n - 1; ++i) {
            m[i] = inner[i - 1];
        }
    }

    segments_.resize(n - 1);
    for (std::size_t i = 0; i < n - 1; ++i) {
        Segment& s = segments_[i];
        s.x0 = x_[i];
        s.a  = y[i];
        s.b  = (y[i + 1] - y[i]) / h[i] - h[i] * (2.0 * m[i] + m[i + 1]) / 6.0;
        s.c  = m[i] / 2.0;
        s.d  = (m[i + 1] - m[i]) / (6.0 * h[i]);
    }
}

const CubicSpline::Segment& CubicSpline::segmentFor(double t) const
{
    // First knot strictly greater than t; the segment starting at the knot
    // before it covers t. Clamped to the first/last segment for t outside
    // [x_.front(), x_.back()) -- extrapolation via that boundary cubic.
    auto it = std::upper_bound(x_.begin(), x_.end(), t);
    std::size_t idx;
    if (it == x_.begin()) {
        idx = 0;
    } else if (it == x_.end()) {
        idx = segments_.size() - 1;
    } else {
        idx = static_cast<std::size_t>(it - x_.begin()) - 1;
    }
    return segments_[idx];
}

double CubicSpline::operator()(double t) const
{
    const Segment& s = segmentFor(t);
    const double dx = t - s.x0;
    return s.a + dx * (s.b + dx * (s.c + dx * s.d));
}

double CubicSpline::derivative(double t) const
{
    const Segment& s = segmentFor(t);
    const double dx = t - s.x0;
    return s.b + dx * (2.0 * s.c + 3.0 * dx * s.d);
}

}  // namespace argus_follower
