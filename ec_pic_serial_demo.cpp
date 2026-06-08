// -----------------------------------------------------------------------------
// ec_pic_serial_demo.cpp
// -----------------------------------------------------------------------------
//
// Serial teaching implementation of the energy-conserving unstaggered potential
// particle-in-cell (PIC) idea described in the accompanying paper draft.
//
// This file is intentionally verbose and intentionally serial.  The goal is to
// make the numerical ideas easy to read, teach, modify, and audit.  There is no
// MPI, no OpenMP, and no external FFT dependency.  A parallel implementation can
// be released later without obscuring this reference version.
//
// What the program demonstrates
// -----------------------------
//
// The code advances the nonrelativistic Vlasov-Maxwell system in a Lorenz-gauge
// potential formulation.  The evolved mesh variables are
//
//     phi  : scalar potential
//     psi  : time derivative of phi
//     A    : vector potential
//     U    : time derivative of A
//     rho  : charge density, advanced from continuity rather than directly
//            deposited after the initial condition
//     J    : orbit-averaged current at the time-centered step
//
// The particle variable used by the pusher is the canonical momentum
//
//     P = m v + q A_h(x),
//
// where A_h is the mesh-interpolated vector potential.  The update follows the
// paper's orbit-discrete-gradient construction:
//
//     1. Guess the end-of-step particle velocity.
//     2. Trace a straight time-centered particle orbit over the step.
//     3. Deposit the orbit-averaged current J^{n+1/2}.
//     4. Update rho from the same J using the discrete continuity equation.
//     5. Solve the Crank-Nicolson wave equations for phi, psi, A, and U.
//     6. Gather the midpoint electric force with the same orbit weights.
//     7. Push canonical momentum using an orbit-discrete-gradient D_A.
//     8. Iterate these steps with Picard iteration until J and v converge.
//
// The important identity is the finite-difference chain rule along each particle
// orbit,
//
//     A_h^{n+1}(x^{n+1}) - A_h^n(x^n)
//       = dt * U_bar + dt * D_A * v_bar.
//
// A pointwise midpoint gradient of A is second order accurate, but it does not
// satisfy this identity exactly for a finite orbit.  The orbit-discrete-gradient
// D_A is built from the same interpolant A_h used in P = m v + q A_h.  When the
// nonlinear solve is converged and the orbit integrals are evaluated accurately,
// this gives particle work = mesh work and total energy is conserved up to
// Picard tolerance, quadrature error, and roundoff.
//
// Why the path is split at cell/spline knots
// ------------------------------------------
//
// The particle shape is a compactly supported B-spline.  Along a particle path,
// the pulled-back shape is only a single polynomial until the particle crosses a
// spline knot.  The conservative implementation therefore splits every orbit at
// all crossed knots, then applies Gauss quadrature on each subinterval.  The
// input option
//
//     split_orbit_at_knots = true
//
// should be left true for conservative runs.  Setting it false is included only
// as a teaching comparison: Gauss and Lorenz residuals may remain small, but the
// vector-potential chain-rule residual and total-energy drift expose the
// inconsistency.
//
// Included examples
// -----------------
//
//     two_stream      : cold two-stream instability, matching the paper setup in
//                       a serial demonstration form
//     landau_weak    : weak Landau damping with a linear-theory envelope
//     landau_strong  : strong Landau damping with the same linear theory used
//                       only as an early-time reference
//
// See README.md for build/run instructions and for the exact input keys used to
// switch between these examples.
//
// -----------------------------------------------------------------------------

#include <algorithm>
#include <array>
#include <cassert>
#include <cctype>
#include <cmath>
#include <complex>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using Real = double;
using Complex = std::complex<Real>;
using Field = std::vector<Real>;
using CField = std::vector<Complex>;
using Vec3 = std::array<Real, 3>;
using Mat3 = std::array<std::array<Real, 3>, 3>;
using VecField = std::array<Field, 3>;

static constexpr Real PI = 3.141592653589793238462643383279502884;

// -----------------------------------------------------------------------------
// Small helpers
// -----------------------------------------------------------------------------

static std::string trim(const std::string& s) {
    const std::string ws = " \t\r\n";
    const auto b = s.find_first_not_of(ws);
    if (b == std::string::npos) return "";
    const auto e = s.find_last_not_of(ws);
    return s.substr(b, e - b + 1);
}

static std::string lowercase(std::string s) {
    for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

static bool is_power_of_two(int n) {
    return n > 0 && (n & (n - 1)) == 0;
}

[[maybe_unused]] static Real sqr(Real x) { return x * x; }

[[maybe_unused]] static Vec3 make_vec3(Real x, Real y, Real z) { return Vec3{x, y, z}; }

static Vec3 add3(const Vec3& a, const Vec3& b) {
    return Vec3{a[0] + b[0], a[1] + b[1], a[2] + b[2]};
}

static Vec3 sub3(const Vec3& a, const Vec3& b) {
    return Vec3{a[0] - b[0], a[1] - b[1], a[2] - b[2]};
}

static Vec3 mul3(Real c, const Vec3& a) {
    return Vec3{c * a[0], c * a[1], c * a[2]};
}

static Real dot3(const Vec3& a, const Vec3& b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

static Real norm3(const Vec3& a) {
    return std::sqrt(dot3(a, a));
}

static Real wrap1(Real x, Real L) {
    Real y = std::fmod(x + 0.5 * L, L);
    if (y < 0.0) y += L;
    return y - 0.5 * L;
}

static Vec3 wrap_pos(const Vec3& x, const Vec3& L) {
    return Vec3{wrap1(x[0], L[0]), wrap1(x[1], L[1]), wrap1(x[2], L[2])};
}

static Real rms_field(const Field& f) {
    long double s = 0.0L;
    for (Real x : f) s += static_cast<long double>(x) * static_cast<long double>(x);
    return std::sqrt(static_cast<Real>(s / std::max<std::size_t>(1, f.size())));
}

static Real rms_vec_particles(const std::vector<Vec3>& a) {
    long double s = 0.0L;
    std::size_t count = 0;
    for (const auto& v : a) {
        for (int d = 0; d < 3; ++d) {
            s += static_cast<long double>(v[d]) * static_cast<long double>(v[d]);
            ++count;
        }
    }
    return std::sqrt(static_cast<Real>(s / std::max<std::size_t>(1, count)));
}

static Real relative_rms_error_vec_particles(const std::vector<Vec3>& a, const std::vector<Vec3>& b) {
    if (a.size() != b.size()) throw std::runtime_error("particle vector size mismatch");
    std::vector<Vec3> diff(a.size());
    for (std::size_t p = 0; p < a.size(); ++p) diff[p] = sub3(a[p], b[p]);
    const Real denom = 0.5 * (rms_vec_particles(a) + rms_vec_particles(b)) + 1.0e-300;
    return rms_vec_particles(diff) / denom;
}

static Real relative_rms_error_vecfield(const VecField& a, const VecField& b) {
    long double sd = 0.0L, sa = 0.0L, sb = 0.0L;
    std::size_t count = 0;
    for (int m = 0; m < 3; ++m) {
        if (a[m].size() != b[m].size()) throw std::runtime_error("vecfield size mismatch");
        for (std::size_t i = 0; i < a[m].size(); ++i) {
            const long double da = static_cast<long double>(a[m][i] - b[m][i]);
            sd += da * da;
            sa += static_cast<long double>(a[m][i]) * static_cast<long double>(a[m][i]);
            sb += static_cast<long double>(b[m][i]) * static_cast<long double>(b[m][i]);
            ++count;
        }
    }
    const Real rd = std::sqrt(static_cast<Real>(sd / std::max<std::size_t>(1, count)));
    const Real ra = std::sqrt(static_cast<Real>(sa / std::max<std::size_t>(1, count)));
    const Real rb = std::sqrt(static_cast<Real>(sb / std::max<std::size_t>(1, count)));
    return rd / (0.5 * (ra + rb) + 1.0e-300);
}

static Real mean_field(const Field& f) {
    long double s = 0.0L;
    for (Real x : f) s += x;
    return static_cast<Real>(s / std::max<std::size_t>(1, f.size()));
}

static void subtract_mean(Field& f) {
    const Real m = mean_field(f);
    for (Real& x : f) x -= m;
}

[[maybe_unused]] static Field zero_field(std::size_t n) { return Field(n, 0.0); }
static VecField zero_vecfield(std::size_t n) { return VecField{Field(n, 0.0), Field(n, 0.0), Field(n, 0.0)}; }

// -----------------------------------------------------------------------------
// Input deck
// -----------------------------------------------------------------------------

struct Config {
    // ------------------------------------------------------------------
    // Mesh and time-step controls.
    // ------------------------------------------------------------------
    // The in-file FFT is radix-2, so nx, ny, and nz must be powers of two.
    // ny=nz=1 is allowed and is useful for 1D teaching examples embedded in
    // the 3D code path.
    int nx = 4, ny = 4, nz = 4;
    Real Lx = 2.0 * PI, Ly = 2.0 * PI, Lz = 2.0 * PI;
    int n_steps = 6;
    Real dt = 0.00625;

    // ------------------------------------------------------------------
    // Initial condition selector.
    // ------------------------------------------------------------------
    // Supported values:
    //   two_stream    - cold two-stream instability
    //   landau_weak   - weak Landau damping demonstration
    //   landau_strong - strong Landau damping demonstration
    // The Landau labels mostly document intent; both use the same initializer
    // with different perturbation amplitudes.
    std::string test_case = "two_stream";

    // For two_stream this is the number of counter-streaming particle pairs per
    // mesh cell.  For Landau examples it is the number of quiet-start velocity
    // samples per spatial mesh cell.
    int particles_per_cell_pair = 4;

    // Particle shape degree.  The paper's production numerical examples use
    // first-order tensor-product B-splines.  The code supports degrees 1..4 for
    // experimentation, but degree 1 is the clearest teaching choice.
    int spline_order = 2;

    // Number of Gauss-Legendre points used on each split orbit segment.  For a
    // B-spline degree r, the paper notes that 2*nq-1 >= 3*r is sufficient for
    // exact integration on each segment.  nq=16 is intentionally generous.
    int orbit_quad_order = 16;
    bool split_orbit_at_knots = true;

    // Picard nonlinear iteration controls.  Conservation is a property of the
    // converged nonlinear equations, so tighter tolerances generally reduce the
    // energy residuals until roundoff/quadrature limits dominate.
    Real picard_tol = 1.0e-11;
    int picard_max_iter = 16;
    bool final_consistency_sweep = true;
    bool run_unit_tests = true;

    // Nondimensional field/source coefficients in SI-like notation.  The
    // compatibility c^2 * mu0 = 1/eps0 should be preserved when changing these.
    Real c = 1.0, eps0 = 1.0, mu0 = 1.0;

    // Plasma and perturbation parameters.
    Real n0 = 1.0;                 // fixed ion background density
    Real v0 = 0.3;                 // two-stream drift speed
    Real perturbation = 0.02;      // velocity perturbation for two-stream,
                                   // density perturbation fallback for Landau
    Real landau_alpha = -1.0;      // density amplitude; negative means use perturbation
    Real thermal_velocity = 1.0;   // 1D Maxwellian thermal speed for Landau
    int perturbation_mode = 1;     // integer Fourier mode in x

    // Optional linear Landau reference.  The weak and strong Landau input decks
    // set these for the standard k=0.5, v_th=1, omega_p=1 electrostatic root:
    // omega ~= 1.4156618886 and gamma ~= -0.1533594669.
    // The strong case is nonlinear, so the same values are only an early-time
    // reference envelope, not a prediction of the full nonlinear evolution.
    Real theory_omega = 0.0;
    Real theory_gamma = 0.0;

    unsigned int seed = 12345;     // retained for future randomized examples
    std::string output_prefix = "orbit_smoke";
};

static bool parse_bool(std::string v) {
    v = lowercase(trim(v));
    return v == "1" || v == "true" || v == "yes" || v == "on";
}

static bool is_landau_case(const Config& cfg) {
    const std::string tc = lowercase(cfg.test_case);
    return tc == "landau" || tc == "landau_weak" || tc == "landau_strong" ||
           tc == "weak_landau" || tc == "strong_landau";
}

static Real landau_density_amplitude(const Config& cfg) {
    return (cfg.landau_alpha >= 0.0) ? cfg.landau_alpha : cfg.perturbation;
}

static Config read_config(const std::string& path) {
    Config cfg;
    std::ifstream in(path);
    if (!in) throw std::runtime_error("could not open input file: " + path);
    std::string line;
    while (std::getline(in, line)) {
        const auto sharp = line.find('#');
        if (sharp != std::string::npos) line = line.substr(0, sharp);
        const auto eq = line.find('=');
        if (eq == std::string::npos) continue;
        const std::string key = trim(line.substr(0, eq));
        const std::string val = trim(line.substr(eq + 1));
        if (key.empty()) continue;
        std::istringstream ss(val);
        if      (key == "nx") ss >> cfg.nx;
        else if (key == "ny") ss >> cfg.ny;
        else if (key == "nz") ss >> cfg.nz;
        else if (key == "Lx") ss >> cfg.Lx;
        else if (key == "Ly") ss >> cfg.Ly;
        else if (key == "Lz") ss >> cfg.Lz;
        else if (key == "n_steps") ss >> cfg.n_steps;
        else if (key == "dt") ss >> cfg.dt;
        else if (key == "test_case" || key == "initial_condition" || key == "example") cfg.test_case = lowercase(val);
        else if (key == "particles_per_cell_pair" || key == "particles_per_cell" || key == "velocity_samples_per_cell") ss >> cfg.particles_per_cell_pair;
        else if (key == "spline_order") ss >> cfg.spline_order;
        else if (key == "orbit_quad_order") ss >> cfg.orbit_quad_order;
        else if (key == "split_orbit_at_knots") cfg.split_orbit_at_knots = parse_bool(val);
        else if (key == "picard_tol") ss >> cfg.picard_tol;
        else if (key == "picard_max_iter") ss >> cfg.picard_max_iter;
        else if (key == "final_consistency_sweep") cfg.final_consistency_sweep = parse_bool(val);
        else if (key == "run_unit_tests") cfg.run_unit_tests = parse_bool(val);
        else if (key == "c") ss >> cfg.c;
        else if (key == "eps0") ss >> cfg.eps0;
        else if (key == "mu0") ss >> cfg.mu0;
        else if (key == "n0") ss >> cfg.n0;
        else if (key == "v0") ss >> cfg.v0;
        else if (key == "perturbation") ss >> cfg.perturbation;
        else if (key == "landau_alpha" || key == "landau_density_perturbation") ss >> cfg.landau_alpha;
        else if (key == "thermal_velocity" || key == "vth") ss >> cfg.thermal_velocity;
        else if (key == "perturbation_mode") ss >> cfg.perturbation_mode;
        else if (key == "theory_omega" || key == "landau_theory_omega") ss >> cfg.theory_omega;
        else if (key == "theory_gamma" || key == "landau_theory_gamma") ss >> cfg.theory_gamma;
        else if (key == "seed") ss >> cfg.seed;
        else if (key == "output_prefix") cfg.output_prefix = val;
        else std::cerr << "warning: unknown input key ignored: " << key << "\n";
    }
    return cfg;
}

// -----------------------------------------------------------------------------
// Grid and Fourier modes
// -----------------------------------------------------------------------------

struct Grid {
    int nx = 0, ny = 0, nz = 0;
    int N = 0;
    Vec3 L{};
    Vec3 dx{};
    Real dV = 0.0;
    Field kx, ky, kz, k2;

    int index(int i, int j, int k) const {
        return (i * ny + j) * nz + k;
    }
};

static int fft_mode_number(int i, int n) {
    // Map an FFT array index to its integer Fourier mode.  The n==1 case is
    // important for the 1D Landau examples embedded in a 3D array: the only
    // available mode must be zero, not -1.
    if (n == 1) return 0;
    return (i <= n / 2) ? i : i - n;
}

static Grid make_grid(const Config& cfg) {
    if (!is_power_of_two(cfg.nx) || !is_power_of_two(cfg.ny) || !is_power_of_two(cfg.nz)) {
        throw std::runtime_error("the in-file radix-2 FFT requires nx, ny, nz to be powers of two");
    }
    Grid g;
    g.nx = cfg.nx; g.ny = cfg.ny; g.nz = cfg.nz;
    g.N = g.nx * g.ny * g.nz;
    g.L = Vec3{cfg.Lx, cfg.Ly, cfg.Lz};
    g.dx = Vec3{cfg.Lx / cfg.nx, cfg.Ly / cfg.ny, cfg.Lz / cfg.nz};
    g.dV = g.dx[0] * g.dx[1] * g.dx[2];
    g.kx.assign(g.N, 0.0); g.ky.assign(g.N, 0.0); g.kz.assign(g.N, 0.0); g.k2.assign(g.N, 0.0);
    for (int i = 0; i < g.nx; ++i) {
        const Real kxi = 2.0 * PI * static_cast<Real>(fft_mode_number(i, g.nx)) / g.L[0];
        for (int j = 0; j < g.ny; ++j) {
            const Real kyj = 2.0 * PI * static_cast<Real>(fft_mode_number(j, g.ny)) / g.L[1];
            for (int k = 0; k < g.nz; ++k) {
                const Real kzk = 2.0 * PI * static_cast<Real>(fft_mode_number(k, g.nz)) / g.L[2];
                const int id = g.index(i, j, k);
                g.kx[id] = kxi; g.ky[id] = kyj; g.kz[id] = kzk;
                g.k2[id] = kxi * kxi + kyj * kyj + kzk * kzk;
            }
        }
    }
    return g;
}

// -----------------------------------------------------------------------------
// Radix-2 FFT
// -----------------------------------------------------------------------------

static void fft1d(std::vector<Complex>& a, bool inverse) {
    const int n = static_cast<int>(a.size());
    if (!is_power_of_two(n)) throw std::runtime_error("fft1d length must be a power of two");

    for (int i = 1, j = 0; i < n; ++i) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) std::swap(a[i], a[j]);
    }

    for (int len = 2; len <= n; len <<= 1) {
        const Real angle = 2.0 * PI / static_cast<Real>(len) * (inverse ? 1.0 : -1.0);
        const Complex wlen(std::cos(angle), std::sin(angle));
        for (int i = 0; i < n; i += len) {
            Complex w(1.0, 0.0);
            for (int j = 0; j < len / 2; ++j) {
                const Complex u = a[i + j];
                const Complex v = a[i + j + len / 2] * w;
                a[i + j] = u + v;
                a[i + j + len / 2] = u - v;
                w *= wlen;
            }
        }
    }

    if (inverse) {
        const Real inv_n = 1.0 / static_cast<Real>(n);
        for (auto& z : a) z *= inv_n;
    }
}

static void fft3d(CField& data, const Grid& g, bool inverse) {
    std::vector<Complex> tmp;

    tmp.resize(g.nz);
    for (int i = 0; i < g.nx; ++i) {
        for (int j = 0; j < g.ny; ++j) {
            for (int k = 0; k < g.nz; ++k) tmp[k] = data[g.index(i, j, k)];
            fft1d(tmp, inverse);
            for (int k = 0; k < g.nz; ++k) data[g.index(i, j, k)] = tmp[k];
        }
    }

    tmp.resize(g.ny);
    for (int i = 0; i < g.nx; ++i) {
        for (int k = 0; k < g.nz; ++k) {
            for (int j = 0; j < g.ny; ++j) tmp[j] = data[g.index(i, j, k)];
            fft1d(tmp, inverse);
            for (int j = 0; j < g.ny; ++j) data[g.index(i, j, k)] = tmp[j];
        }
    }

    tmp.resize(g.nx);
    for (int j = 0; j < g.ny; ++j) {
        for (int k = 0; k < g.nz; ++k) {
            for (int i = 0; i < g.nx; ++i) tmp[i] = data[g.index(i, j, k)];
            fft1d(tmp, inverse);
            for (int i = 0; i < g.nx; ++i) data[g.index(i, j, k)] = tmp[i];
        }
    }
}

static CField fft_real(const Field& f, const Grid& g) {
    CField out(g.N);
    for (int id = 0; id < g.N; ++id) out[id] = Complex(f[id], 0.0);
    fft3d(out, g, false);
    return out;
}

static Field ifft_real(CField fhat, const Grid& g) {
    fft3d(fhat, g, true);
    Field f(g.N);
    for (int id = 0; id < g.N; ++id) f[id] = fhat[id].real();
    return f;
}

// -----------------------------------------------------------------------------
// Spectral operators
// -----------------------------------------------------------------------------

static VecField spectral_gradient_scalar(const Field& f, const Grid& g) {
    const Complex I(0.0, 1.0);
    const CField fhat = fft_real(f, g);
    VecField out = zero_vecfield(g.N);
    for (int d = 0; d < 3; ++d) {
        CField dhat(g.N);
        for (int id = 0; id < g.N; ++id) {
            const Real kk = (d == 0) ? g.kx[id] : (d == 1) ? g.ky[id] : g.kz[id];
            dhat[id] = I * kk * fhat[id];
        }
        out[d] = ifft_real(std::move(dhat), g);
    }
    return out;
}

static Field spectral_divergence_vector(const VecField& V, const Grid& g) {
    const Complex I(0.0, 1.0);
    const CField hx = fft_real(V[0], g);
    const CField hy = fft_real(V[1], g);
    const CField hz = fft_real(V[2], g);
    CField divhat(g.N);
    for (int id = 0; id < g.N; ++id) {
        divhat[id] = I * (g.kx[id] * hx[id] + g.ky[id] * hy[id] + g.kz[id] * hz[id]);
    }
    return ifft_real(std::move(divhat), g);
}

static VecField spectral_curl(const VecField& A, const Grid& g) {
    const Complex I(0.0, 1.0);
    const CField ax = fft_real(A[0], g);
    const CField ay = fft_real(A[1], g);
    const CField az = fft_real(A[2], g);
    std::array<CField, 3> ch{CField(g.N), CField(g.N), CField(g.N)};
    for (int id = 0; id < g.N; ++id) {
        ch[0][id] = I * (g.ky[id] * az[id] - g.kz[id] * ay[id]);
        ch[1][id] = I * (g.kz[id] * ax[id] - g.kx[id] * az[id]);
        ch[2][id] = I * (g.kx[id] * ay[id] - g.ky[id] * ax[id]);
    }
    VecField curl = zero_vecfield(g.N);
    for (int d = 0; d < 3; ++d) curl[d] = ifft_real(std::move(ch[d]), g);
    return curl;
}

// -----------------------------------------------------------------------------
// B-spline shape functions and derivatives
// -----------------------------------------------------------------------------

static Real cardinal_bspline(int order, Real t) {
    if (order < 1) throw std::runtime_error("cardinal_bspline order must be positive");
    if (order == 1) return (t >= 0.0 && t < 1.0) ? 1.0 : 0.0;
    return (t / static_cast<Real>(order - 1)) * cardinal_bspline(order - 1, t)
         + ((static_cast<Real>(order) - t) / static_cast<Real>(order - 1)) * cardinal_bspline(order - 1, t - 1.0);
}

static Real cardinal_bspline_derivative(int order, Real t) {
    if (order <= 1) return 0.0;
    return cardinal_bspline(order - 1, t) - cardinal_bspline(order - 1, t - 1.0);
}

static Real centered_bspline(int degree, Real r) {
    if (degree < 1 || degree > 4) throw std::runtime_error("spline_order must be 1, 2, 3, or 4");
    return cardinal_bspline(degree + 1, r + 0.5 * static_cast<Real>(degree + 1));
}

static Real centered_bspline_derivative(int degree, Real r) {
    if (degree < 1 || degree > 4) throw std::runtime_error("spline_order must be 1, 2, 3, or 4");
    return cardinal_bspline_derivative(degree + 1, r + 0.5 * static_cast<Real>(degree + 1));
}

struct AxisStencil {
    std::array<int, 5> idx{};
    std::array<Real, 5> w{};
    std::array<Real, 5> dw_dx{};
};

static AxisStencil axis_stencil(Real x, int n, Real L, int order) {
    AxisStencil s;
    const Real dx = L / static_cast<Real>(n);
    const Real u = (x + 0.5 * L) / dx;
    const int i_left = static_cast<int>(std::floor(u - 0.5 * static_cast<Real>(order - 1)));
    for (int a = 0; a <= order; ++a) {
        const int raw = i_left + a;
        int ii = raw % n;
        if (ii < 0) ii += n;
        const Real r = u - static_cast<Real>(raw);
        s.idx[a] = ii;
        s.w[a] = centered_bspline(order, r);
        s.dw_dx[a] = centered_bspline_derivative(order, r) / dx;
    }
    return s;
}

// -----------------------------------------------------------------------------
// Scatter/gather operations
// -----------------------------------------------------------------------------

static void deposit_scalar_add(Field& mesh, const Vec3& pos, Real value, const Grid& g, int order, bool divide_by_dV) {
    const Vec3 x = wrap_pos(pos, g.L);
    const AxisStencil sx = axis_stencil(x[0], g.nx, g.L[0], order);
    const AxisStencil sy = axis_stencil(x[1], g.ny, g.L[1], order);
    const AxisStencil sz = axis_stencil(x[2], g.nz, g.L[2], order);
    const Real scale = divide_by_dV ? (1.0 / g.dV) : 1.0;
    for (int a = 0; a <= order; ++a) {
        for (int b = 0; b <= order; ++b) {
            for (int c = 0; c <= order; ++c) {
                const int id = g.index(sx.idx[a], sy.idx[b], sz.idx[c]);
                mesh[id] += value * sx.w[a] * sy.w[b] * sz.w[c] * scale;
            }
        }
    }
}

static Field deposit_scalar_to_mesh(const std::vector<Vec3>& pos, const std::vector<Real>& values,
                                    const Grid& g, int order, bool divide_by_dV) {
    if (pos.size() != values.size()) throw std::runtime_error("deposit_scalar size mismatch");
    Field mesh(g.N, 0.0);
    for (std::size_t p = 0; p < pos.size(); ++p) deposit_scalar_add(mesh, pos[p], values[p], g, order, divide_by_dV);
    return mesh;
}

static void deposit_current_add(VecField& J, const Vec3& pos, const Vec3& vel, Real charge,
                                const Grid& g, int order, Real prefactor) {
    const Vec3 x = wrap_pos(pos, g.L);
    const AxisStencil sx = axis_stencil(x[0], g.nx, g.L[0], order);
    const AxisStencil sy = axis_stencil(x[1], g.ny, g.L[1], order);
    const AxisStencil sz = axis_stencil(x[2], g.nz, g.L[2], order);
    for (int a = 0; a <= order; ++a) {
        for (int b = 0; b <= order; ++b) {
            for (int c = 0; c <= order; ++c) {
                const int id = g.index(sx.idx[a], sy.idx[b], sz.idx[c]);
                const Real w = prefactor * charge * sx.w[a] * sy.w[b] * sz.w[c] / g.dV;
                for (int d = 0; d < 3; ++d) J[d][id] += w * vel[d];
            }
        }
    }
}

static VecField deposit_current_to_mesh(const std::vector<Vec3>& pos, const std::vector<Vec3>& vel,
                                        const std::vector<Real>& q, const Grid& g, int order) {
    VecField J = zero_vecfield(g.N);
    for (std::size_t p = 0; p < pos.size(); ++p) deposit_current_add(J, pos[p], vel[p], q[p], g, order, 1.0);
    return J;
}

static Real gather_scalar_at(const Field& mesh, const Vec3& pos, const Grid& g, int order) {
    const Vec3 x = wrap_pos(pos, g.L);
    const AxisStencil sx = axis_stencil(x[0], g.nx, g.L[0], order);
    const AxisStencil sy = axis_stencil(x[1], g.ny, g.L[1], order);
    const AxisStencil sz = axis_stencil(x[2], g.nz, g.L[2], order);
    Real out = 0.0;
    for (int a = 0; a <= order; ++a) {
        for (int b = 0; b <= order; ++b) {
            for (int c = 0; c <= order; ++c) {
                const int id = g.index(sx.idx[a], sy.idx[b], sz.idx[c]);
                out += mesh[id] * sx.w[a] * sy.w[b] * sz.w[c];
            }
        }
    }
    return out;
}

static Vec3 gather_vector_at(const VecField& mesh, const Vec3& pos, const Grid& g, int order) {
    return Vec3{
        gather_scalar_at(mesh[0], pos, g, order),
        gather_scalar_at(mesh[1], pos, g, order),
        gather_scalar_at(mesh[2], pos, g, order)
    };
}

static std::vector<Vec3> gather_vector_all(const std::vector<Vec3>& pos, const VecField& mesh, const Grid& g, int order) {
    std::vector<Vec3> out(pos.size());
    for (std::size_t p = 0; p < pos.size(); ++p) out[p] = gather_vector_at(mesh, pos[p], g, order);
    return out;
}

static Mat3 zero_mat3() {
    Mat3 M{};
    for (int a = 0; a < 3; ++a) for (int b = 0; b < 3; ++b) M[a][b] = 0.0;
    return M;
}

static Mat3 gather_shape_gradient_vector_interp_at(const VecField& A0, const VecField& A1,
                                                   Real s_interp, const Vec3& pos,
                                                   const Grid& g, int order) {
    const Vec3 x = wrap_pos(pos, g.L);
    const AxisStencil sx = axis_stencil(x[0], g.nx, g.L[0], order);
    const AxisStencil sy = axis_stencil(x[1], g.ny, g.L[1], order);
    const AxisStencil sz = axis_stencil(x[2], g.nz, g.L[2], order);
    Mat3 out = zero_mat3();

    for (int a = 0; a <= order; ++a) {
        for (int b = 0; b <= order; ++b) {
            for (int c = 0; c <= order; ++c) {
                const int id = g.index(sx.idx[a], sy.idx[b], sz.idx[c]);
                const Real wx = sx.w[a], wy = sy.w[b], wz = sz.w[c];
                const Real dwx = sx.dw_dx[a], dwy = sy.dw_dx[b], dwz = sz.dw_dx[c];
                const Real dSdx[3] = {dwx * wy * wz, wx * dwy * wz, wx * wy * dwz};
                for (int comp = 0; comp < 3; ++comp) {
                    const Real Aval = (1.0 - s_interp) * A0[comp][id] + s_interp * A1[comp][id];
                    for (int d = 0; d < 3; ++d) out[comp][d] += Aval * dSdx[d];
                }
            }
        }
    }
    return out;
}

[[maybe_unused]] static Mat3 gather_shape_gradient_vector_at(const VecField& A, const Vec3& pos, const Grid& g, int order) {
    return gather_shape_gradient_vector_interp_at(A, A, 0.0, pos, g, order);
}

// -----------------------------------------------------------------------------
// Gauss-Legendre quadrature on [0,1]
// -----------------------------------------------------------------------------

struct Quadrature {
    std::vector<Real> s;
    std::vector<Real> w;
};

static std::pair<Real, Real> legendre_value_and_derivative(int n, Real x) {
    Real p0 = 1.0;
    Real p1 = x;
    if (n == 0) return {p0, 0.0};
    if (n == 1) return {p1, 1.0};
    for (int k = 2; k <= n; ++k) {
        const Real pk = ((2.0 * k - 1.0) * x * p1 - (k - 1.0) * p0) / static_cast<Real>(k);
        p0 = p1;
        p1 = pk;
    }
    const Real dp = static_cast<Real>(n) * (x * p1 - p0) / (x * x - 1.0);
    return {p1, dp};
}

static Quadrature gauss_legendre_01(int n) {
    Quadrature q;
    q.s.assign(n, 0.0);
    q.w.assign(n, 0.0);
    const int m = (n + 1) / 2;
    for (int i = 0; i < m; ++i) {
        Real x = std::cos(PI * (static_cast<Real>(i) + 0.75) / (static_cast<Real>(n) + 0.5));
        for (int it = 0; it < 80; ++it) {
            const auto pd = legendre_value_and_derivative(n, x);
            const Real dx = -pd.first / pd.second;
            x += dx;
            if (std::abs(dx) < 1.0e-15) break;
        }
        const auto pd = legendre_value_and_derivative(n, x);
        const Real w = 2.0 / ((1.0 - x * x) * pd.second * pd.second);
        const Real s_left = 0.5 * (1.0 - x);
        const Real s_right = 0.5 * (1.0 + x);
        const Real w01 = 0.5 * w;
        q.s[i] = s_left;
        q.w[i] = w01;
        q.s[n - 1 - i] = s_right;
        q.w[n - 1 - i] = w01;
    }
    return q;
}

// -----------------------------------------------------------------------------
// Orbit integration with particle crossing of spline knots
// -----------------------------------------------------------------------------

struct PathStats {
    std::size_t n_particles = 0;
    std::size_t crossing_particles = 0;
    std::size_t total_segments = 0;
    int max_segments = 1;
};

static Vec3 path_position(const Vec3& x0, const Vec3& dx_path, Real s, const Grid& g) {
    // The path is advanced in unwrapped coordinates, then wrapped only for
    // evaluating the periodic mesh shape functions.
    return wrap_pos(add3(x0, mul3(s, dx_path)), g.L);
}

static std::vector<Real> path_breakpoints_for_particle(const Vec3& x0, const Vec3& dx_path,
                                                       const Grid& g, int order,
                                                       bool split_at_knots) {
    std::vector<Real> b;
    b.reserve(16);
    b.push_back(0.0);
    b.push_back(1.0);

    if (!split_at_knots) {
        return b;
    }

    // A tensor-product centered B-spline is a different polynomial branch when
    // any coordinate crosses a one-dimensional spline knot.  The active stencil
    // in axis_stencil() changes when
    //
    //     u(s) - 0.5*(order-1) is an integer,
    //
    // where u=(x+L/2)/dx is the grid coordinate.  The following loop adds all
    // such crossing times s in the open interval (0,1).
    const Real tol = 1.0e-13;
    for (int d = 0; d < 3; ++d) {
        const Real Ld = g.L[d];
        const Real hd = g.dx[d];
        const Real u0 = (wrap1(x0[d], Ld) + 0.5 * Ld) / hd;
        const Real du = dx_path[d] / hd;
        if (std::abs(du) < 1.0e-300) continue;

        const Real offset = 0.5 * static_cast<Real>(order - 1);
        const Real a0 = u0 - offset;
        const Real a1 = a0 + du;
        const Real lo = std::min(a0, a1);
        const Real hi = std::max(a0, a1);

        const long long first = static_cast<long long>(std::floor(lo)) + 1;
        const long long last  = static_cast<long long>(std::ceil(hi)) - 1;
        for (long long m = first; m <= last; ++m) {
            const Real s = (static_cast<Real>(m) - a0) / du;
            if (s > tol && s < 1.0 - tol) b.push_back(s);
        }
    }

    std::sort(b.begin(), b.end());
    std::vector<Real> out;
    out.reserve(b.size());
    for (Real x : b) {
        if (out.empty() || std::abs(x - out.back()) > 1.0e-12) out.push_back(x);
    }
    if (out.front() != 0.0) out.insert(out.begin(), 0.0);
    if (out.back() != 1.0) out.push_back(1.0);
    return out;
}

static PathStats compute_path_stats(const std::vector<Vec3>& x0, const std::vector<Vec3>& dx_path,
                                    const Grid& g, int order, bool split_at_knots) {
    PathStats stats;
    stats.n_particles = x0.size();
    stats.max_segments = 1;
    for (std::size_t p = 0; p < x0.size(); ++p) {
        const auto br = path_breakpoints_for_particle(x0[p], dx_path[p], g, order, split_at_knots);
        const int nseg = std::max<int>(1, static_cast<int>(br.size()) - 1);
        stats.total_segments += static_cast<std::size_t>(nseg);
        stats.max_segments = std::max(stats.max_segments, nseg);
        if (nseg > 1) ++stats.crossing_particles;
    }
    return stats;
}

static void update_path_stats(PathStats& stats, const std::vector<Real>& br) {
    const int nseg = std::max<int>(1, static_cast<int>(br.size()) - 1);
    ++stats.n_particles;
    stats.total_segments += static_cast<std::size_t>(nseg);
    stats.max_segments = std::max(stats.max_segments, nseg);
    if (nseg > 1) ++stats.crossing_particles;
}

static std::vector<Vec3> path_average_gather_vector(const std::vector<Vec3>& x0,
                                                    const std::vector<Vec3>& dx_path,
                                                    const VecField& mesh, const Grid& g,
                                                    int order, const Quadrature& quad,
                                                    bool split_at_knots = true) {
    std::vector<Vec3> out(x0.size(), Vec3{0.0, 0.0, 0.0});
    for (std::size_t p = 0; p < x0.size(); ++p) {
        const auto br = path_breakpoints_for_particle(x0[p], dx_path[p], g, order, split_at_knots);
        for (std::size_t seg = 0; seg + 1 < br.size(); ++seg) {
            const Real a = br[seg];
            const Real b = br[seg + 1];
            const Real len = b - a;
            if (len <= 0.0) continue;
            for (std::size_t iq = 0; iq < quad.s.size(); ++iq) {
                const Real s = a + len * quad.s[iq];
                const Real w = len * quad.w[iq];
                const Vec3 xs = path_position(x0[p], dx_path[p], s, g);
                const Vec3 val = gather_vector_at(mesh, xs, g, order);
                for (int d = 0; d < 3; ++d) out[p][d] += w * val[d];
            }
        }
    }
    return out;
}

static VecField path_average_deposit_current(const std::vector<Vec3>& x0, const std::vector<Vec3>& dx_path,
                                             const std::vector<Vec3>& vbar, const std::vector<Real>& q,
                                             const Grid& g, int order, const Quadrature& quad,
                                             bool split_at_knots = true, PathStats* stats = nullptr) {
    VecField J = zero_vecfield(g.N);
    if (stats) *stats = PathStats{};
    for (std::size_t p = 0; p < x0.size(); ++p) {
        const auto br = path_breakpoints_for_particle(x0[p], dx_path[p], g, order, split_at_knots);
        if (stats) update_path_stats(*stats, br);
        for (std::size_t seg = 0; seg + 1 < br.size(); ++seg) {
            const Real a = br[seg];
            const Real b = br[seg + 1];
            const Real len = b - a;
            if (len <= 0.0) continue;
            for (std::size_t iq = 0; iq < quad.s.size(); ++iq) {
                const Real s = a + len * quad.s[iq];
                const Real w = len * quad.w[iq];
                const Vec3 xs = path_position(x0[p], dx_path[p], s, g);
                deposit_current_add(J, xs, vbar[p], q[p], g, order, w);
            }
        }
    }
    return J;
}

static std::vector<Mat3> orbit_discrete_gradient_A(const std::vector<Vec3>& x0, const std::vector<Vec3>& dx_path,
                                                   const VecField& A_n, const VecField& A_np1,
                                                   const Grid& g, int order, const Quadrature& quad,
                                                   bool split_at_knots = true) {
    std::vector<Mat3> D(x0.size(), zero_mat3());
    for (std::size_t p = 0; p < x0.size(); ++p) {
        const auto br = path_breakpoints_for_particle(x0[p], dx_path[p], g, order, split_at_knots);
        for (std::size_t seg = 0; seg + 1 < br.size(); ++seg) {
            const Real a = br[seg];
            const Real b = br[seg + 1];
            const Real len = b - a;
            if (len <= 0.0) continue;
            for (std::size_t iq = 0; iq < quad.s.size(); ++iq) {
                const Real s = a + len * quad.s[iq];
                const Real w = len * quad.w[iq];
                const Vec3 xs = path_position(x0[p], dx_path[p], s, g);
                const Mat3 G = gather_shape_gradient_vector_interp_at(A_n, A_np1, s, xs, g, order);
                for (int a0 = 0; a0 < 3; ++a0) {
                    for (int b0 = 0; b0 < 3; ++b0) {
                        D[p][a0][b0] += w * G[a0][b0];
                    }
                }
            }
        }
    }
    return D;
}

// -----------------------------------------------------------------------------
// Field solve and diagnostics
// -----------------------------------------------------------------------------

static std::pair<Field, Field> cn_fft_wave_step(const Field& u, const Field& ut, const Field& S_mid,
                                                Real dt, const Grid& g, Real c, Real source_scale) {
    const CField uhat = fft_real(u, g);
    const CField vhat = fft_real(ut, g);
    const CField Shat = fft_real(S_mid, g);
    CField unew_hat(g.N), vnew_hat(g.N);
    const Real c2 = c * c;
    const Real source_coeff = c2 * source_scale;
    for (int id = 0; id < g.N; ++id) {
        const Real theta = 0.25 * c2 * dt * dt * g.k2[id];
        const Real denom = 1.0 + theta;
        unew_hat[id] = ((1.0 - theta) * uhat[id] + dt * vhat[id] + 0.5 * source_coeff * dt * dt * Shat[id]) / denom;
        vnew_hat[id] = ((1.0 - theta) * vhat[id] - c2 * dt * g.k2[id] * uhat[id] + source_coeff * dt * Shat[id]) / denom;
    }
    Field unew = ifft_real(std::move(unew_hat), g);
    Field vnew = ifft_real(std::move(vnew_hat), g);
    return {std::move(unew), std::move(vnew)};
}

static Field solve_periodic_poisson_for_phi(const Field& rho, const Grid& g, Real eps0) {
    CField rhohat = fft_real(rho, g);
    CField phihat(g.N, Complex(0.0, 0.0));
    for (int id = 0; id < g.N; ++id) {
        if (g.k2[id] > 0.0) phihat[id] = rhohat[id] / (eps0 * g.k2[id]);
    }
    Field phi = ifft_real(std::move(phihat), g);
    subtract_mean(phi);
    return phi;
}

struct State {
    std::vector<Vec3> x, v, v_prev, P;
    std::vector<Real> q, m;
    Field phi, psi, rho;
    VecField A, U, J;
};

static void cn_potential_update(const State& st, const Field& rho_mid, const VecField& J_mid,
                                Real dt, const Grid& g, Real c, Real eps0, Real mu0,
                                Field& phi_new, Field& psi_new, VecField& A_new, VecField& U_new) {
    auto pp = cn_fft_wave_step(st.phi, st.psi, rho_mid, dt, g, c, 1.0 / eps0);
    phi_new = std::move(pp.first);
    psi_new = std::move(pp.second);
    subtract_mean(phi_new);
    subtract_mean(psi_new);

    A_new = zero_vecfield(g.N);
    U_new = zero_vecfield(g.N);
    for (int d = 0; d < 3; ++d) {
        auto au = cn_fft_wave_step(st.A[d], st.U[d], J_mid[d], dt, g, c, mu0);
        A_new[d] = std::move(au.first);
        U_new[d] = std::move(au.second);
    }
}

static Field continuity_charge_update(const Field& rho_n, const VecField& J_mid, Real dt, const Grid& g) {
    Field divJ = spectral_divergence_vector(J_mid, g);
    Field rho(g.N);
    for (int id = 0; id < g.N; ++id) rho[id] = rho_n[id] - dt * divJ[id];
    subtract_mean(rho);
    return rho;
}

static VecField add_vecfield(const VecField& a, const VecField& b, Real ca, Real cb) {
    VecField out = zero_vecfield(a[0].size());
    for (int d = 0; d < 3; ++d) {
        for (std::size_t i = 0; i < a[d].size(); ++i) out[d][i] = ca * a[d][i] + cb * b[d][i];
    }
    return out;
}

static VecField compute_E(const State& st, const Grid& g) {
    VecField grad_phi = spectral_gradient_scalar(st.phi, g);
    VecField E = zero_vecfield(g.N);
    for (int d = 0; d < 3; ++d) for (int id = 0; id < g.N; ++id) E[d][id] = -grad_phi[d][id] - st.U[d][id];
    return E;
}

static Real compute_field_energy(const State& st, const Grid& g, Real eps0, Real mu0) {
    const VecField E = compute_E(st, g);
    const VecField B = spectral_curl(st.A, g);
    long double se = 0.0L, sb = 0.0L;
    for (int d = 0; d < 3; ++d) {
        for (int id = 0; id < g.N; ++id) {
            se += static_cast<long double>(E[d][id]) * static_cast<long double>(E[d][id]);
            sb += static_cast<long double>(B[d][id]) * static_cast<long double>(B[d][id]);
        }
    }
    return static_cast<Real>((0.5 * eps0 * se + 0.5 / mu0 * sb) * g.dV);
}

static Real compute_kinetic_energy(const State& st) {
    long double s = 0.0L;
    for (std::size_t p = 0; p < st.v.size(); ++p) s += 0.5L * st.m[p] * dot3(st.v[p], st.v[p]);
    return static_cast<Real>(s);
}

struct Diagnostics {
    Real gauss_rms = 0.0;
    Real gauss_max = 0.0;
    Real gauge_rms = 0.0;
    Real field_energy = 0.0;
    Real kinetic_energy = 0.0;
    Real total_energy = 0.0;

    // Fourier-mode diagnostics for the requested x mode.  rho_mode is the
    // charge-density mode amplitude.  E_mode_* are the complex Fourier
    // coefficient of E_x at (mode,0,0), normalized by the number of grid points.
    // Landau examples compare E_mode_abs with a linear-theory envelope in the
    // CSV writer.
    Real rho_mode = 0.0;
    Real E_mode_re = 0.0;
    Real E_mode_im = 0.0;
    Real E_mode_abs = 0.0;
    Real mean_rho = 0.0;
};

static Diagnostics compute_diagnostics(const State& st, const Grid& g, Real eps0, Real mu0, Real c, int mode) {
    Diagnostics d;
    const VecField E = compute_E(st, g);
    Field divE = spectral_divergence_vector(E, g);
    Field gauss(g.N);
    for (int id = 0; id < g.N; ++id) gauss[id] = divE[id] - st.rho[id] / eps0;
    d.gauss_rms = rms_field(gauss);
    d.gauss_max = 0.0;
    for (Real x : gauss) d.gauss_max = std::max(d.gauss_max, std::abs(x));

    Field divA = spectral_divergence_vector(st.A, g);
    Field gauge(g.N);
    for (int id = 0; id < g.N; ++id) gauge[id] = st.psi[id] / (c * c) + divA[id];
    d.gauge_rms = rms_field(gauge);

    d.field_energy = compute_field_energy(st, g, eps0, mu0);
    d.kinetic_energy = compute_kinetic_energy(st);
    d.total_energy = d.field_energy + d.kinetic_energy;

    CField rhohat = fft_real(st.rho, g);
    CField Exhat = fft_real(E[0], g);
    const int imode = ((mode % g.nx) + g.nx) % g.nx;
    const int id = g.index(imode, 0, 0);
    const Real invN = 1.0 / static_cast<Real>(g.N);
    d.rho_mode = std::abs(rhohat[id]) * invN;
    d.E_mode_re = Exhat[id].real() * invN;
    d.E_mode_im = Exhat[id].imag() * invN;
    d.E_mode_abs = std::abs(Exhat[id]) * invN;
    d.mean_rho = mean_field(st.rho);
    return d;
}

// -----------------------------------------------------------------------------
// Initial condition: 3D embedding of linearly spaced 1D two-stream setup
// -----------------------------------------------------------------------------

static std::vector<Real> cell_centered_grid(int n, Real L, bool centered) {
    std::vector<Real> x(n);
    const Real dx = L / static_cast<Real>(n);
    for (int i = 0; i < n; ++i) {
        x[i] = (static_cast<Real>(i) + 0.5) * dx;
        if (centered) x[i] -= 0.5 * L;
    }
    return x;
}

static Real normal_quantile(Real p) {
    // Acklam's rational approximation for the inverse standard-normal CDF.
    // It is accurate enough for quiet-start velocity sampling and avoids a
    // dependency on special-function libraries.  The Landau examples use
    // p=(j+1/2)/Nv, so p is never exactly 0 or 1.
    if (!(p > 0.0 && p < 1.0)) throw std::runtime_error("normal_quantile requires 0<p<1");

    static const Real a[] = {
        -3.969683028665376e+01,  2.209460984245205e+02,
        -2.759285104469687e+02,  1.383577518672690e+02,
        -3.066479806614716e+01,  2.506628277459239e+00
    };
    static const Real b[] = {
        -5.447609879822406e+01,  1.615858368580409e+02,
        -1.556989798598866e+02,  6.680131188771972e+01,
        -1.328068155288572e+01
    };
    static const Real c[] = {
        -7.784894002430293e-03, -3.223964580411365e-01,
        -2.400758277161838e+00, -2.549732539343734e+00,
         4.374664141464968e+00,  2.938163982698783e+00
    };
    static const Real d[] = {
         7.784695709041462e-03,  3.224671290700398e-01,
         2.445134137142996e+00,  3.754408661907416e+00
    };

    const Real plow = 0.02425;
    const Real phigh = 1.0 - plow;
    Real q, r, x;

    if (p < plow) {
        q = std::sqrt(-2.0 * std::log(p));
        x = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0);
    } else if (p <= phigh) {
        q = p - 0.5;
        r = q * q;
        x = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q /
            (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0);
    } else {
        q = std::sqrt(-2.0 * std::log(1.0 - p));
        x = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
             ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0);
    }
    return x;
}

static State initialize_two_stream_3d_linear_x(const Config& cfg, const Grid& g) {
    State st;
    const int n_x_per_yz = g.nx * cfg.particles_per_cell_pair;
    const std::vector<Real> xline = cell_centered_grid(n_x_per_yz, g.L[0], true);
    const std::vector<Real> yline = cell_centered_grid(g.ny, g.L[1], true);
    const std::vector<Real> zline = cell_centered_grid(g.nz, g.L[2], true);

    std::vector<Vec3> base;
    base.reserve(static_cast<std::size_t>(n_x_per_yz) * g.ny * g.nz);
    for (Real y : yline) for (Real z : zline) for (Real x : xline) base.push_back(Vec3{x, y, z});

    const std::size_t nbase = base.size();
    const std::size_t np = 2 * nbase;
    st.x.assign(np, Vec3{0.0, 0.0, 0.0});
    st.v.assign(np, Vec3{0.0, 0.0, 0.0});
    st.v_prev.assign(np, Vec3{0.0, 0.0, 0.0});
    st.P.assign(np, Vec3{0.0, 0.0, 0.0});
    st.q.assign(np, 0.0);
    st.m.assign(np, 0.0);

    const Real kx = 2.0 * PI * static_cast<Real>(cfg.perturbation_mode) / g.L[0];
    for (std::size_t p = 0; p < nbase; ++p) {
        const Real seed_v = cfg.perturbation * std::sin(kx * base[p][0]);
        st.x[p] = wrap_pos(base[p], g.L);
        st.x[nbase + p] = wrap_pos(base[p], g.L);
        st.v[p] = Vec3{+cfg.v0 + seed_v, 0.0, 0.0};
        st.v[nbase + p] = Vec3{-cfg.v0 + seed_v, 0.0, 0.0};
    }

    const Real volume = g.L[0] * g.L[1] * g.L[2];
    const Real q_macro = -cfg.n0 * volume / static_cast<Real>(np);
    const Real m_macro = std::abs(q_macro);
    for (std::size_t p = 0; p < np; ++p) {
        st.q[p] = q_macro;
        st.m[p] = m_macro;
    }

    Field rho_e = deposit_scalar_to_mesh(st.x, st.q, g, cfg.spline_order, true);
    st.rho.assign(g.N, 0.0);
    for (int id = 0; id < g.N; ++id) st.rho[id] = rho_e[id] + cfg.n0;
    subtract_mean(st.rho);

    st.J = deposit_current_to_mesh(st.x, st.v, st.q, g, cfg.spline_order);
    st.phi = solve_periodic_poisson_for_phi(st.rho, g, cfg.eps0);
    st.psi.assign(g.N, 0.0);
    st.A = zero_vecfield(g.N);
    st.U = zero_vecfield(g.N);

    for (std::size_t p = 0; p < np; ++p) st.P[p] = mul3(st.m[p], st.v[p]);

    // Taylor startup for v^{n-1}.  Since A=0 initially,
    // a^n = (q/m) E^n.
    State tmp = st;
    const VecField E0 = compute_E(tmp, g);
    const auto E0p = gather_vector_all(st.x, E0, g, cfg.spline_order);
    for (std::size_t p = 0; p < np; ++p) {
        const Vec3 acc = mul3(st.q[p] / st.m[p], E0p[p]);
        st.v_prev[p] = sub3(st.v[p], mul3(cfg.dt, acc));
    }

    return st;
}

static State initialize_landau_damping_1d3d(const Config& cfg, const Grid& g) {
    // ------------------------------------------------------------------
    // Landau damping initial condition.
    // ------------------------------------------------------------------
    // This is a 1D-1V electrostatic-style Landau test embedded in the same 3D
    // field and particle infrastructure used by the two-stream example.  The
    // perturbation is in x only.  y and z are included so the same scatter,
    // gather, FFT, Picard, and energy-diagnostic code paths are exercised.
    //
    // A quiet-start velocity grid is generated by sampling the inverse CDF of a
    // unit Maxwellian.  Each velocity sample has equal statistical weight.  The
    // density perturbation is represented by modulating the electron macrocharge
    // as
    //
    //     q_p(x) = q0 * [1 + alpha cos(k x)],
    //
    // while setting m_p = |q_p|.  Therefore q_p/m_p = -1 for every electron,
    // matching the nondimensional electron charge-to-mass ratio used in the
    // paper examples.  The fixed ion background +n0 is then added on the mesh.
    const int nv = cfg.particles_per_cell_pair;
    if (nv < 2) throw std::runtime_error("Landau examples need at least two velocity samples per cell");

    const Real alpha = landau_density_amplitude(cfg);
    if (std::abs(alpha) >= 1.0) {
        throw std::runtime_error("Landau density amplitude must satisfy |alpha| < 1 so all macro weights stay positive");
    }

    State st;
    const std::vector<Real> xline = cell_centered_grid(g.nx, g.L[0], true);
    const std::vector<Real> yline = cell_centered_grid(g.ny, g.L[1], true);
    const std::vector<Real> zline = cell_centered_grid(g.nz, g.L[2], true);

    std::vector<Real> vquiet(nv);
    for (int j = 0; j < nv; ++j) {
        const Real p_cdf = (static_cast<Real>(j) + 0.5) / static_cast<Real>(nv);
        vquiet[j] = cfg.thermal_velocity * normal_quantile(p_cdf);
    }

    const std::size_t np = static_cast<std::size_t>(g.nx) * g.ny * g.nz * nv;
    st.x.assign(np, Vec3{0.0, 0.0, 0.0});
    st.v.assign(np, Vec3{0.0, 0.0, 0.0});
    st.v_prev.assign(np, Vec3{0.0, 0.0, 0.0});
    st.P.assign(np, Vec3{0.0, 0.0, 0.0});
    st.q.assign(np, 0.0);
    st.m.assign(np, 0.0);

    const Real kx = 2.0 * PI * static_cast<Real>(cfg.perturbation_mode) / g.L[0];
    const Real volume = g.L[0] * g.L[1] * g.L[2];
    const Real q0 = -cfg.n0 * volume / static_cast<Real>(np);

    std::size_t pidx = 0;
    for (Real y : yline) {
        for (Real z : zline) {
            for (Real x : xline) {
                const Real density_weight = 1.0 + alpha * std::cos(kx * x);
                for (int jv = 0; jv < nv; ++jv) {
                    st.x[pidx] = wrap_pos(Vec3{x, y, z}, g.L);
                    st.v[pidx] = Vec3{vquiet[jv], 0.0, 0.0};
                    st.q[pidx] = q0 * density_weight;
                    st.m[pidx] = std::abs(st.q[pidx]);
                    ++pidx;
                }
            }
        }
    }

    Field rho_e = deposit_scalar_to_mesh(st.x, st.q, g, cfg.spline_order, true);
    st.rho.assign(g.N, 0.0);
    for (int id = 0; id < g.N; ++id) st.rho[id] = rho_e[id] + cfg.n0;
    subtract_mean(st.rho);

    st.J = deposit_current_to_mesh(st.x, st.v, st.q, g, cfg.spline_order);
    st.phi = solve_periodic_poisson_for_phi(st.rho, g, cfg.eps0);
    st.psi.assign(g.N, 0.0);
    st.A = zero_vecfield(g.N);
    st.U = zero_vecfield(g.N);

    for (std::size_t p = 0; p < np; ++p) st.P[p] = mul3(st.m[p], st.v[p]);

    // Same Taylor startup as the two-stream case.  The history velocity is used
    // only to form the first Picard extrapolation v^{n+1,0}=2v^n-v^{n-1}.
    State tmp = st;
    const VecField E0 = compute_E(tmp, g);
    const auto E0p = gather_vector_all(st.x, E0, g, cfg.spline_order);
    for (std::size_t p = 0; p < np; ++p) {
        const Vec3 acc = mul3(st.q[p] / st.m[p], E0p[p]);
        st.v_prev[p] = sub3(st.v[p], mul3(cfg.dt, acc));
    }

    return st;
}

static State initialize_state(const Config& cfg, const Grid& g) {
    // Centralized switch for teaching examples.  To add a new initial condition,
    // implement another initialize_* function and add one branch here.  The rest
    // of the program intentionally does not know which IC was used.
    const std::string tc = lowercase(cfg.test_case);
    if (tc == "two_stream" || tc == "two-stream" || tc == "twostream") {
        return initialize_two_stream_3d_linear_x(cfg, g);
    }
    if (is_landau_case(cfg)) {
        return initialize_landau_damping_1d3d(cfg, g);
    }
    throw std::runtime_error("unknown test_case/initial_condition: " + cfg.test_case);
}

// -----------------------------------------------------------------------------
// R_A,p residual and step diagnostics
// -----------------------------------------------------------------------------

struct RAResult {
    std::vector<Vec3> R;
    std::vector<Vec3> A_delta;
    std::vector<Vec3> rhs;
};

static RAResult compute_RA_orbit_residual(const std::vector<Vec3>& x0, const std::vector<Vec3>& vbar,
                                          const VecField& A_n, const VecField& A_np1,
                                          const VecField& U_mid, Real dt, const Grid& g,
                                          int order, const Quadrature& quad,
                                          bool split_at_knots = true) {
    const std::size_t np = x0.size();
    std::vector<Vec3> dx_path(np), x1(np);
    for (std::size_t p = 0; p < np; ++p) {
        dx_path[p] = mul3(dt, vbar[p]);
        x1[p] = wrap_pos(add3(x0[p], dx_path[p]), g.L);
    }
    RAResult rr;
    rr.R.assign(np, Vec3{0.0, 0.0, 0.0});
    rr.A_delta.assign(np, Vec3{0.0, 0.0, 0.0});
    rr.rhs.assign(np, Vec3{0.0, 0.0, 0.0});

    const auto Ubar = path_average_gather_vector(x0, dx_path, U_mid, g, order, quad, split_at_knots);
    const auto DA = orbit_discrete_gradient_A(x0, dx_path, A_n, A_np1, g, order, quad, split_at_knots);
    for (std::size_t p = 0; p < np; ++p) {
        const Vec3 A0p = gather_vector_at(A_n, x0[p], g, order);
        const Vec3 A1p = gather_vector_at(A_np1, x1[p], g, order);
        rr.A_delta[p] = sub3(A1p, A0p);
        Vec3 DAv{0.0, 0.0, 0.0};
        for (int a = 0; a < 3; ++a) {
            DAv[a] = DA[p][a][0] * vbar[p][0] + DA[p][a][1] * vbar[p][1] + DA[p][a][2] * vbar[p][2];
        }
        rr.rhs[p] = mul3(dt, add3(Ubar[p], DAv));
        rr.R[p] = sub3(rr.A_delta[p], rr.rhs[p]);
    }
    return rr;
}

struct StepDiagnostics {
    Real A_chain_abs_rms = 0.0;
    Real A_chain_rel_rms = 0.0;
    Real A_chain_max_norm = 0.0;
    Real A_chain_mean_norm = 0.0;
    Real A_chain_work_defect = 0.0;
    Real delta_K = 0.0;
    Real delta_W = 0.0;
    Real delta_total_energy = 0.0;
    Real particle_work = 0.0;
    Real grid_work = 0.0;
    Real deposit_gather_work_residual = 0.0;
    Real particle_energy_residual = 0.0;
    Real field_energy_residual = 0.0;
    std::size_t crossing_particles = 0;
    std::size_t total_crossings = 0;
    Real average_path_segments = 1.0;
    int max_path_segments = 1;
};

static Real rms_from_vec3s(const std::vector<Vec3>& a) { return rms_vec_particles(a); }

static StepDiagnostics compute_step_diagnostics(const State& s0, const State& s1, const Grid& g,
                                                Real dt, int order, const Quadrature& quad,
                                                Real eps0, Real mu0, bool split_at_knots = true) {
    StepDiagnostics d;
    const std::size_t np = s0.x.size();
    std::vector<Vec3> vbar(np), dx_path(np);
    for (std::size_t p = 0; p < np; ++p) {
        vbar[p] = mul3(0.5, add3(s1.v[p], s0.v[p]));
        dx_path[p] = mul3(dt, vbar[p]);
    }

    const PathStats ps = compute_path_stats(s0.x, dx_path, g, order, split_at_knots);
    d.crossing_particles = ps.crossing_particles;
    d.total_crossings = (ps.total_segments >= ps.n_particles) ? (ps.total_segments - ps.n_particles) : 0;
    d.average_path_segments = ps.n_particles ? static_cast<Real>(ps.total_segments) / static_cast<Real>(ps.n_particles) : 1.0;
    d.max_path_segments = ps.max_segments;

    VecField U_mid = add_vecfield(s0.U, s1.U, 0.5, 0.5);
    RAResult rr = compute_RA_orbit_residual(s0.x, vbar, s0.A, s1.A, U_mid, dt, g, order, quad, split_at_knots);
    d.A_chain_abs_rms = rms_from_vec3s(rr.R);
    d.A_chain_rel_rms = d.A_chain_abs_rms / (0.5 * (rms_from_vec3s(rr.A_delta) + rms_from_vec3s(rr.rhs)) + 1.0e-300);
    long double norm_sum = 0.0L;
    d.A_chain_max_norm = 0.0;
    long double work_defect = 0.0L;
    for (std::size_t p = 0; p < np; ++p) {
        const Real nr = norm3(rr.R[p]);
        d.A_chain_max_norm = std::max(d.A_chain_max_norm, nr);
        norm_sum += nr;
        work_defect += -static_cast<long double>(s0.q[p]) * dot3(vbar[p], rr.R[p]);
    }
    d.A_chain_mean_norm = static_cast<Real>(norm_sum / std::max<std::size_t>(1, np));
    d.A_chain_work_defect = static_cast<Real>(work_defect);

    VecField grad_phi0 = spectral_gradient_scalar(s0.phi, g);
    VecField grad_phi1 = spectral_gradient_scalar(s1.phi, g);
    VecField E_mid = zero_vecfield(g.N);
    for (int m = 0; m < 3; ++m) {
        for (int id = 0; id < g.N; ++id) E_mid[m][id] = -0.5 * (grad_phi0[m][id] + grad_phi1[m][id]) - U_mid[m][id];
    }
    const auto Ebar_p = path_average_gather_vector(s0.x, dx_path, E_mid, g, order, quad, split_at_knots);
    const VecField Jbar = path_average_deposit_current(s0.x, dx_path, vbar, s0.q, g, order, quad, split_at_knots);

    long double pw = 0.0L, gw = 0.0L;
    for (std::size_t p = 0; p < np; ++p) pw += static_cast<long double>(s0.q[p]) * dot3(vbar[p], Ebar_p[p]);
    for (int m = 0; m < 3; ++m) for (int id = 0; id < g.N; ++id) gw += static_cast<long double>(Jbar[m][id]) * E_mid[m][id];
    d.particle_work = static_cast<Real>(dt * pw);
    d.grid_work = static_cast<Real>(dt * gw * g.dV);

    d.delta_K = compute_kinetic_energy(s1) - compute_kinetic_energy(s0);
    d.delta_W = compute_field_energy(s1, g, eps0, mu0) - compute_field_energy(s0, g, eps0, mu0);
    d.delta_total_energy = d.delta_K + d.delta_W;
    d.deposit_gather_work_residual = d.particle_work - d.grid_work;
    d.particle_energy_residual = d.delta_K - d.particle_work;
    d.field_energy_residual = d.delta_W + d.grid_work;
    return d;
}

// -----------------------------------------------------------------------------
// Picard map and time step
// -----------------------------------------------------------------------------

struct PicardTrial {
    std::vector<Vec3> vbar, dx_path, x_end;
    VecField Jbar;
    Field rho_end, rho_mid;
    Field phi_end, psi_end;
    VecField A_end, U_end;
    std::vector<Vec3> P_end, v_next;
    PathStats path_stats;
};

static PicardTrial evaluate_picard_map_orbit(const std::vector<Vec3>& v_guess,
                                             const State& st, const Grid& g, const Config& cfg,
                                             const VecField& grad_phi_n,
                                             const Quadrature& quad) {
    const std::size_t np = st.x.size();
    PicardTrial tr;
    tr.vbar.assign(np, Vec3{0.0, 0.0, 0.0});
    tr.dx_path.assign(np, Vec3{0.0, 0.0, 0.0});
    tr.x_end.assign(np, Vec3{0.0, 0.0, 0.0});
    tr.P_end.assign(np, Vec3{0.0, 0.0, 0.0});
    tr.v_next.assign(np, Vec3{0.0, 0.0, 0.0});

    for (std::size_t p = 0; p < np; ++p) {
        tr.vbar[p] = mul3(0.5, add3(v_guess[p], st.v[p]));
        tr.dx_path[p] = mul3(cfg.dt, tr.vbar[p]);
        tr.x_end[p] = wrap_pos(add3(st.x[p], tr.dx_path[p]), g.L);
    }

    tr.Jbar = path_average_deposit_current(st.x, tr.dx_path, tr.vbar, st.q, g, cfg.spline_order, quad, cfg.split_orbit_at_knots, &tr.path_stats);
    tr.rho_end = continuity_charge_update(st.rho, tr.Jbar, cfg.dt, g);
    tr.rho_mid.assign(g.N, 0.0);
    for (int id = 0; id < g.N; ++id) tr.rho_mid[id] = 0.5 * (tr.rho_end[id] + st.rho[id]);

    cn_potential_update(st, tr.rho_mid, tr.Jbar, cfg.dt, g, cfg.c, cfg.eps0, cfg.mu0,
                        tr.phi_end, tr.psi_end, tr.A_end, tr.U_end);

    VecField grad_phi_end = spectral_gradient_scalar(tr.phi_end, g);
    VecField grad_phi_mid = zero_vecfield(g.N);
    for (int d = 0; d < 3; ++d) for (int id = 0; id < g.N; ++id) grad_phi_mid[d][id] = 0.5 * (grad_phi_n[d][id] + grad_phi_end[d][id]);

    const auto grad_phi_bar_p = path_average_gather_vector(st.x, tr.dx_path, grad_phi_mid, g, cfg.spline_order, quad, cfg.split_orbit_at_knots);
    const auto DA = orbit_discrete_gradient_A(st.x, tr.dx_path, st.A, tr.A_end, g, cfg.spline_order, quad, cfg.split_orbit_at_knots);

    for (std::size_t p = 0; p < np; ++p) {
        const Vec3 A_end_p = gather_vector_at(tr.A_end, tr.x_end[p], g, cfg.spline_order);
        Vec3 DATv{0.0, 0.0, 0.0};
        // D_A[a][b] = d A_a / d x_b.  The canonical term is D_A^T vbar.
        for (int b = 0; b < 3; ++b) {
            DATv[b] = DA[p][0][b] * tr.vbar[p][0] + DA[p][1][b] * tr.vbar[p][1] + DA[p][2][b] * tr.vbar[p][2];
        }
        Vec3 forceP{0.0, 0.0, 0.0};
        for (int d = 0; d < 3; ++d) forceP[d] = (-st.q[p]) * grad_phi_bar_p[p][d] + st.q[p] * DATv[d];
        tr.P_end[p] = add3(st.P[p], mul3(cfg.dt, forceP));
        for (int d = 0; d < 3; ++d) tr.v_next[p][d] = (tr.P_end[p][d] - st.q[p] * A_end_p[d]) / st.m[p];
    }
    return tr;
}

struct PicardHistoryEntry {
    int iter = 0;
    Real current_error = std::numeric_limits<Real>::infinity();
    Real velocity_error = std::numeric_limits<Real>::infinity();
};

static State picard_step_orbit(const State& st, const Grid& g, const Config& cfg, const Quadrature& quad,
                               std::vector<PicardHistoryEntry>& hist) {
    const std::size_t np = st.x.size();
    VecField grad_phi_n = spectral_gradient_scalar(st.phi, g);
    std::vector<Vec3> v_guess(np);
    for (std::size_t p = 0; p < np; ++p) v_guess[p] = sub3(mul3(2.0, st.v[p]), st.v_prev[p]);

    VecField J_prev = zero_vecfield(g.N);
    bool have_J_prev = false;
    PicardTrial tr;
    hist.clear();

    for (int it = 0; it < cfg.picard_max_iter; ++it) {
        tr = evaluate_picard_map_orbit(v_guess, st, g, cfg, grad_phi_n, quad);
        PicardHistoryEntry he;
        he.iter = it;
        he.velocity_error = relative_rms_error_vec_particles(tr.v_next, v_guess);
        he.current_error = have_J_prev ? relative_rms_error_vecfield(tr.Jbar, J_prev) : std::numeric_limits<Real>::infinity();
        hist.push_back(he);
        if (it > 0 && std::max(he.current_error, he.velocity_error) < cfg.picard_tol) break;
        J_prev = tr.Jbar;
        have_J_prev = true;
        v_guess = tr.v_next;
    }

    if (cfg.final_consistency_sweep) {
        tr = evaluate_picard_map_orbit(tr.v_next, st, g, cfg, grad_phi_n, quad);
    }

    State out;
    out.x = std::move(tr.x_end);
    out.v = std::move(tr.v_next);
    out.v_prev = st.v;
    out.P = std::move(tr.P_end);
    out.q = st.q;
    out.m = st.m;
    out.phi = std::move(tr.phi_end);
    out.psi = std::move(tr.psi_end);
    out.A = std::move(tr.A_end);
    out.U = std::move(tr.U_end);
    out.rho = std::move(tr.rho_end);
    out.J = std::move(tr.Jbar);
    return out;
}

// -----------------------------------------------------------------------------
// Unit tests
// -----------------------------------------------------------------------------

static bool test_RA_orbit_chain_rule(std::ostream& os) {
    Config cfg;
    cfg.nx = cfg.ny = cfg.nz = 8;
    cfg.Lx = cfg.Ly = cfg.Lz = 2.0 * PI;
    cfg.dt = 0.01;
    cfg.spline_order = 2;
    cfg.orbit_quad_order = 20;
    Grid g = make_grid(cfg);
    Quadrature quad = gauss_legendre_01(cfg.orbit_quad_order);

    std::mt19937_64 rng(11);
    std::normal_distribution<Real> normal(0.0, 1.0);
    std::uniform_int_distribution<int> cell_dist(0, g.nx - 1);
    const std::size_t np = 128;
    std::vector<Vec3> x0(np), vbar(np);
    for (std::size_t p = 0; p < np; ++p) {
        const int ix = cell_dist(rng), iy = cell_dist(rng), iz = cell_dist(rng);
        x0[p] = Vec3{-0.5 * g.L[0] + (ix + 0.31) * g.dx[0],
                     -0.5 * g.L[1] + (iy + 0.37) * g.dx[1],
                     -0.5 * g.L[2] + (iz + 0.43) * g.dx[2]};
        vbar[p] = Vec3{0.03 * normal(rng), 0.03 * normal(rng), 0.03 * normal(rng)};
    }

    VecField A0 = zero_vecfield(g.N);
    VecField Umid = zero_vecfield(g.N);
    VecField A1 = zero_vecfield(g.N);
    for (int m = 0; m < 3; ++m) {
        for (int id = 0; id < g.N; ++id) {
            A0[m][id] = normal(rng);
            Umid[m][id] = normal(rng);
            A1[m][id] = A0[m][id] + cfg.dt * Umid[m][id];
        }
    }

    RAResult rr = compute_RA_orbit_residual(x0, vbar, A0, A1, Umid, cfg.dt, g, cfg.spline_order, quad);
    const Real abs_rms = rms_from_vec3s(rr.R);
    const Real rel_rms = abs_rms / (0.5 * (rms_from_vec3s(rr.A_delta) + rms_from_vec3s(rr.rhs)) + 1.0e-300);
    os << std::scientific << std::setprecision(6)
       << "R_A,p chain-rule unit test: abs_rms=" << abs_rms
       << " rel_rms=" << rel_rms << "\n";
    if (!(abs_rms < 5.0e-12 && rel_rms < 5.0e-10)) {
        os << "FAILED: R_A,p residual is too large\n";
        return false;
    }
    return true;
}


static bool test_RA_orbit_chain_rule_with_crossings(std::ostream& os) {
    Config cfg;
    cfg.nx = cfg.ny = cfg.nz = 8;
    cfg.Lx = cfg.Ly = cfg.Lz = 2.0 * PI;
    cfg.dt = 1.0;
    cfg.spline_order = 1;
    cfg.orbit_quad_order = 16;
    Grid g = make_grid(cfg);
    Quadrature quad = gauss_legendre_01(cfg.orbit_quad_order);

    std::mt19937_64 rng(17);
    std::normal_distribution<Real> normal(0.0, 1.0);
    const std::size_t np = 96;
    std::vector<Vec3> x0(np), vbar(np);
    for (std::size_t p = 0; p < np; ++p) {
        // Put each particle near a spline knot and force it to cross that knot.
        const int ix = static_cast<int>(p % static_cast<std::size_t>(g.nx));
        const int iy = static_cast<int>((p / g.nx) % static_cast<std::size_t>(g.ny));
        const int iz = static_cast<int>((p / (g.nx * g.ny)) % static_cast<std::size_t>(g.nz));
        const Real sx = (p % 2 == 0) ? 0.49 : -0.49;
        const Real sy = ((p / 2) % 2 == 0) ? 0.45 : -0.45;
        const Real sz = ((p / 4) % 2 == 0) ? 0.42 : -0.42;
        x0[p] = Vec3{-0.5 * g.L[0] + (ix + 0.5 + sx) * g.dx[0],
                     -0.5 * g.L[1] + (iy + 0.5 + sy) * g.dx[1],
                     -0.5 * g.L[2] + (iz + 0.5 + sz) * g.dx[2]};
        vbar[p] = Vec3{0.35 * g.dx[0] * ((sx > 0.0) ? 1.0 : -1.0),
                       0.25 * g.dx[1] * ((sy > 0.0) ? 1.0 : -1.0),
                       0.20 * g.dx[2] * ((sz > 0.0) ? 1.0 : -1.0)};
    }

    VecField A0 = zero_vecfield(g.N);
    VecField Umid = zero_vecfield(g.N);
    VecField A1 = zero_vecfield(g.N);
    for (int m = 0; m < 3; ++m) {
        for (int id = 0; id < g.N; ++id) {
            A0[m][id] = normal(rng);
            Umid[m][id] = normal(rng);
            A1[m][id] = A0[m][id] + cfg.dt * Umid[m][id];
        }
    }

    RAResult rr_split = compute_RA_orbit_residual(x0, vbar, A0, A1, Umid, cfg.dt, g, cfg.spline_order, quad, true);
    RAResult rr_unsplit = compute_RA_orbit_residual(x0, vbar, A0, A1, Umid, cfg.dt, g, cfg.spline_order, quad, false);
    const Real split_abs = rms_from_vec3s(rr_split.R);
    const Real unsplit_abs = rms_from_vec3s(rr_unsplit.R);
    PathStats stats = compute_path_stats(x0, std::vector<Vec3>(vbar.begin(), vbar.end()), g, cfg.spline_order, true);

    os << std::scientific << std::setprecision(6)
       << "R_A,p crossing unit test: split_abs_rms=" << split_abs
       << " unsplit_abs_rms=" << unsplit_abs
       << " crossing_particles=" << stats.crossing_particles
       << " max_segments=" << stats.max_segments << "\n";
    if (!(split_abs < 5.0e-12 && unsplit_abs > 1.0e-10 && stats.crossing_particles > 0)) {
        os << "FAILED: split-path crossing test did not behave as expected\n";
        return false;
    }
    return true;
}

static bool test_initial_gauss_law(std::ostream& os) {
    Config cfg;
    cfg.nx = cfg.ny = cfg.nz = 8;
    cfg.Lx = cfg.Ly = cfg.Lz = 2.0 * PI;
    cfg.dt = 0.00625;
    cfg.spline_order = 2;
    cfg.particles_per_cell_pair = 2;
    Grid g = make_grid(cfg);
    State st = initialize_two_stream_3d_linear_x(cfg, g);
    Diagnostics d = compute_diagnostics(st, g, cfg.eps0, cfg.mu0, cfg.c, cfg.perturbation_mode);
    os << std::scientific << std::setprecision(6)
       << "Initial Gauss-law unit test: gauss_rms=" << d.gauss_rms << "\n";
    if (!(d.gauss_rms < 5.0e-11)) {
        os << "FAILED: initial Gauss residual is too large\n";
        return false;
    }
    return true;
}

static bool test_one_step_gauss_law_and_RA(std::ostream& os) {
    Config cfg;
    cfg.nx = cfg.ny = cfg.nz = 4;
    cfg.Lx = cfg.Ly = cfg.Lz = 2.0 * PI;
    cfg.dt = 0.00625;
    cfg.n_steps = 1;
    cfg.spline_order = 2;
    cfg.particles_per_cell_pair = 2;
    cfg.orbit_quad_order = 16;
    cfg.picard_tol = 1.0e-11;
    cfg.picard_max_iter = 16;
    Grid g = make_grid(cfg);
    Quadrature quad = gauss_legendre_01(cfg.orbit_quad_order);
    State s0 = initialize_two_stream_3d_linear_x(cfg, g);
    std::vector<PicardHistoryEntry> hist;
    State s1 = picard_step_orbit(s0, g, cfg, quad, hist);
    Diagnostics d1 = compute_diagnostics(s1, g, cfg.eps0, cfg.mu0, cfg.c, cfg.perturbation_mode);
    StepDiagnostics sd = compute_step_diagnostics(s0, s1, g, cfg.dt, cfg.spline_order, quad, cfg.eps0, cfg.mu0);
    os << std::scientific << std::setprecision(6)
       << "One-step Gauss-law unit test: gauss_rms=" << d1.gauss_rms << "\n"
       << "One-step R_A,p unit test: A_chain_abs_rms=" << sd.A_chain_abs_rms << "\n"
       << "One-step deposit/gather work residual=" << sd.deposit_gather_work_residual << "\n";
    if (!(d1.gauss_rms < 5.0e-10 && sd.A_chain_abs_rms < 5.0e-10)) {
        os << "FAILED: one-step Gauss or R_A,p test failed\n";
        return false;
    }
    return true;
}

// -----------------------------------------------------------------------------
// Output helpers
// -----------------------------------------------------------------------------

static void write_particle_sample(const std::string& path, const State& st, std::size_t max_particles) {
    std::ofstream out(path);
    out << "particle,x,y,z,vx,vy,vz,q,m\n";
    const std::size_t np = st.x.size();
    const std::size_t nwrite = std::min(max_particles, np);
    for (std::size_t r = 0; r < nwrite; ++r) {
        const std::size_t p = (nwrite == 1) ? 0 : (r * (np - 1) / (nwrite - 1));
        out << p << "," << std::setprecision(17)
            << st.x[p][0] << "," << st.x[p][1] << "," << st.x[p][2] << ","
            << st.v[p][0] << "," << st.v[p][1] << "," << st.v[p][2] << ","
            << st.q[p] << "," << st.m[p] << "\n";
    }
}

static void write_config_echo(const std::string& path, const Config& cfg, const Grid& g, std::size_t np) {
    std::ofstream out(path);
    out << "test_case=" << cfg.test_case << "\n";
    out << "nx=" << cfg.nx << "\n";
    out << "ny=" << cfg.ny << "\n";
    out << "nz=" << cfg.nz << "\n";
    out << "Lx=" << std::setprecision(17) << cfg.Lx << "\n";
    out << "Ly=" << std::setprecision(17) << cfg.Ly << "\n";
    out << "Lz=" << std::setprecision(17) << cfg.Lz << "\n";
    out << "n_steps=" << cfg.n_steps << "\n";
    out << "dt=" << std::setprecision(17) << cfg.dt << "\n";
    out << "particles=" << np << "\n";
    out << "particles_per_cell_pair_or_velocity_samples=" << cfg.particles_per_cell_pair << "\n";
    out << "spline_order=" << cfg.spline_order << "\n";
    out << "orbit_quad_order=" << cfg.orbit_quad_order << "\n";
    out << "split_orbit_at_knots=" << (cfg.split_orbit_at_knots ? "true" : "false") << "\n";
    out << "picard_tol=" << cfg.picard_tol << "\n";
    out << "picard_max_iter=" << cfg.picard_max_iter << "\n";
    out << "final_consistency_sweep=" << (cfg.final_consistency_sweep ? "true" : "false") << "\n";
    out << "c=" << cfg.c << "\n";
    out << "eps0=" << cfg.eps0 << "\n";
    out << "mu0=" << cfg.mu0 << "\n";
    out << "n0=" << cfg.n0 << "\n";
    out << "v0=" << cfg.v0 << "\n";
    out << "perturbation=" << cfg.perturbation << "\n";
    out << "landau_alpha=" << landau_density_amplitude(cfg) << "\n";
    out << "thermal_velocity=" << cfg.thermal_velocity << "\n";
    out << "perturbation_mode=" << cfg.perturbation_mode << "\n";
    out << "theory_omega=" << cfg.theory_omega << "\n";
    out << "theory_gamma=" << cfg.theory_gamma << "\n";
    out << "dV=" << g.dV << "\n";
}

// -----------------------------------------------------------------------------
// Main driver
// -----------------------------------------------------------------------------

int main(int argc, char** argv) {
    try {
        if (argc < 2) {
            std::cerr << "usage: " << argv[0] << " input.txt\n";
            return 2;
        }
        const Config cfg = read_config(argv[1]);
        const Grid g = make_grid(cfg);
        const Quadrature quad = gauss_legendre_01(cfg.orbit_quad_order);

        std::cout << "Serial teaching code: orbit-discrete-gradient Picard CN potential PIC\n";
        std::cout << "test_case = " << cfg.test_case << " (serial, no MPI/OpenMP)\n";
        std::cout << "grid = " << cfg.nx << " x " << cfg.ny << " x " << cfg.nz
                  << ", dt = " << cfg.dt << ", steps = " << cfg.n_steps << "\n";

        std::ofstream unit_log(cfg.output_prefix + "_unit_tests.txt");
        bool tests_ok = true;
        if (cfg.run_unit_tests) {
            tests_ok = test_RA_orbit_chain_rule(std::cout) && tests_ok;
            tests_ok = test_RA_orbit_chain_rule_with_crossings(std::cout) && tests_ok;
            tests_ok = test_initial_gauss_law(std::cout) && tests_ok;
            tests_ok = test_one_step_gauss_law_and_RA(std::cout) && tests_ok;
            test_RA_orbit_chain_rule(unit_log);
            test_RA_orbit_chain_rule_with_crossings(unit_log);
            test_initial_gauss_law(unit_log);
            test_one_step_gauss_law_and_RA(unit_log);
        }
        if (!tests_ok) {
            std::cerr << "one or more unit tests failed\n";
            return 3;
        }
        if (cfg.run_unit_tests) std::cout << "Unit tests passed.\n";

        State state = initialize_state(cfg, g);
        write_config_echo(cfg.output_prefix + "_config_echo.txt", cfg, g, state.x.size());

        // Store the initial electric-field Fourier amplitude so Landau runs can
        // write a simple linear-theory envelope |E_k(0)| exp(gamma t).  For the
        // strong Landau case this is only a reference curve, because nonlinear
        // trapping invalidates the linear theory at later times.
        const Diagnostics initial_diag = compute_diagnostics(state, g, cfg.eps0, cfg.mu0, cfg.c, cfg.perturbation_mode);
        const Real theory_E0 = initial_diag.E_mode_abs;
        const bool has_theory_reference = (std::abs(cfg.theory_gamma) > 0.0 || std::abs(cfg.theory_omega) > 0.0) && theory_E0 > 0.0;

        std::ofstream diag(cfg.output_prefix + "_diagnostics.csv");
        diag << "step,time,gauss_rms,gauss_max,gauge_rms,field_energy,kinetic_energy,total_energy,"
             << "rho_mode,E_mode_re,E_mode_im,E_mode_abs,theory_E_abs,E_over_theory,mean_rho,"
             << "picard_iterations,picard_current_error,picard_velocity_error,"
             << "crossing_particles,avg_path_segments,max_path_segments\n";

        std::ofstream stepcsv(cfg.output_prefix + "_step_diagnostics.csv");
        stepcsv << "step,time,A_chain_abs_rms,A_chain_rel_rms,A_chain_max_norm,A_chain_mean_norm,A_chain_work_defect,delta_K,delta_W,delta_total_energy,particle_work,grid_work,deposit_gather_work_residual,particle_energy_residual,field_energy_residual,crossing_particles,total_crossings,avg_path_segments,max_path_segments\n";

        auto write_diag = [&](int step, Real time, const State& st, int pic_iters, Real cur_err, Real vel_err, const PathStats& ps) {
            const Diagnostics d = compute_diagnostics(st, g, cfg.eps0, cfg.mu0, cfg.c, cfg.perturbation_mode);
            Real theory_E_abs = std::numeric_limits<Real>::quiet_NaN();
            Real E_over_theory = std::numeric_limits<Real>::quiet_NaN();
            if (has_theory_reference) {
                theory_E_abs = theory_E0 * std::exp(cfg.theory_gamma * time);
                E_over_theory = d.E_mode_abs / (theory_E_abs + 1.0e-300);
            }
            diag << step << "," << std::setprecision(17) << time << ","
                 << d.gauss_rms << "," << d.gauss_max << "," << d.gauge_rms << ","
                 << d.field_energy << "," << d.kinetic_energy << "," << d.total_energy << ","
                 << d.rho_mode << "," << d.E_mode_re << "," << d.E_mode_im << ","
                 << d.E_mode_abs << "," << theory_E_abs << "," << E_over_theory << ","
                 << d.mean_rho << ","
                 << pic_iters << "," << cur_err << "," << vel_err << ","
                 << ps.crossing_particles << ","
                 << (ps.n_particles ? static_cast<Real>(ps.total_segments) / static_cast<Real>(ps.n_particles) : 0.0) << ","
                 << ps.max_segments << "\n";
        };

        write_diag(0, 0.0, state, 0, std::numeric_limits<Real>::quiet_NaN(), std::numeric_limits<Real>::quiet_NaN(), PathStats{});

        for (int n = 0; n < cfg.n_steps; ++n) {
            State state_n = state;
            std::vector<PicardHistoryEntry> hist;
            state = picard_step_orbit(state_n, g, cfg, quad, hist);
            const StepDiagnostics sd = compute_step_diagnostics(state_n, state, g, cfg.dt, cfg.spline_order, quad, cfg.eps0, cfg.mu0, cfg.split_orbit_at_knots);
            const Real time = (n + 1) * cfg.dt;
            const int pic_iters = static_cast<int>(hist.size());
            const Real cur_err = hist.empty() ? std::numeric_limits<Real>::quiet_NaN() : hist.back().current_error;
            const Real vel_err = hist.empty() ? std::numeric_limits<Real>::quiet_NaN() : hist.back().velocity_error;
            std::vector<Vec3> dxp_for_stats(state_n.x.size());
            for (std::size_t pp = 0; pp < state_n.x.size(); ++pp) {
                const Vec3 vbar_stats = mul3(0.5, add3(state.v[pp], state_n.v[pp]));
                dxp_for_stats[pp] = mul3(cfg.dt, vbar_stats);
            }
            const PathStats step_path_stats = compute_path_stats(state_n.x, dxp_for_stats, g, cfg.spline_order, cfg.split_orbit_at_knots);
            write_diag(n + 1, time, state, pic_iters, cur_err, vel_err, step_path_stats);
            stepcsv << (n + 1) << "," << std::setprecision(17) << time << ","
                    << sd.A_chain_abs_rms << "," << sd.A_chain_rel_rms << ","
                    << sd.A_chain_max_norm << "," << sd.A_chain_mean_norm << ","
                    << sd.A_chain_work_defect << "," << sd.delta_K << "," << sd.delta_W << ","
                    << sd.delta_total_energy << "," << sd.particle_work << "," << sd.grid_work << ","
                    << sd.deposit_gather_work_residual << "," << sd.particle_energy_residual << ","
                    << sd.field_energy_residual << "," << sd.crossing_particles << ","
                    << sd.total_crossings << "," << sd.average_path_segments << ","
                    << sd.max_path_segments << "\n";

            if ((n + 1) % std::max(1, cfg.n_steps / 5) == 0 || n == cfg.n_steps - 1) {
                const Diagnostics d = compute_diagnostics(state, g, cfg.eps0, cfg.mu0, cfg.c, cfg.perturbation_mode);
                std::cout << std::scientific << std::setprecision(3)
                          << "step " << std::setw(5) << (n + 1) << "/" << cfg.n_steps
                          << " Gauss=" << d.gauss_rms
                          << " R_A=" << sd.A_chain_abs_rms
                          << " dE=" << sd.delta_total_energy
                          << " crossings=" << sd.total_crossings
                          << " Picard=" << pic_iters << "\n";
            }
        }

        write_particle_sample(cfg.output_prefix + "_particle_sample.csv", state, 20000);
        const Diagnostics fd = compute_diagnostics(state, g, cfg.eps0, cfg.mu0, cfg.c, cfg.perturbation_mode);
        std::cout << std::scientific << std::setprecision(6)
                  << "Final Gauss RMS = " << fd.gauss_rms << "\n"
                  << "Final gauge RMS = " << fd.gauge_rms << "\n"
                  << "Final total energy = " << fd.total_energy << "\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
