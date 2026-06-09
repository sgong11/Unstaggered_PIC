// -----------------------------------------------------------------------------
// spectral_pic_part3.cpp
// -----------------------------------------------------------------------------
// A simple, serial, self-contained C++17 teaching code for the spectral
// generalized-momentum PIC method used in
//
//   Christlieb, Sands, and White,
//   "A Particle-in-cell Method for Plasmas with A Generalized Momentum
//    Formulation, Part III: A family of Gauge Conserving Methods",
//   Journal of Scientific Computing 104:38 (2025).
//
// The goals of this file are deliberately modest:
//
//   * Keep the implementation readable enough to teach from.
//   * Use no external dependencies: the FFT is implemented below.
//   * Use one serial executable with text input decks.
//   * Demonstrate the two numerical examples from the paper:
//       - the 2D Weibel/filamentation instability,
//       - the 2D drifting cloud of electrons.
//   * Demonstrate the paper's central numerical point: if charge is advanced
//     with a time discretization consistent with the field solve, the fully
//     discrete Lorenz gauge and Gauss-law errors are greatly reduced compared
//     with naive redeposition of charge.
//
// What this teaching code includes
// --------------------------------
//   - Periodic 2D domain.
//   - Relativistic generalized/canonical particle momentum P = m gamma v + q A.
//   - Improved asymmetric Euler method (IAEM) particle push.
//   - Quadratic tensor-product B-spline scatter/gather on a collocated mesh.
//   - Spectral derivatives and spectral Helmholtz/wave solves using an internal
//     radix-2 FFT.
//   - Unstaggered BDF1 and BDF2 field updates.
//   - Two source treatments:
//       charge_update = conserving : update rho from the same BDF continuity law
//                                    used by the fields.
//       charge_update = naive       : redeposit rho directly from particles.
//
// What this teaching code does not include
// ----------------------------------------
//   The paper also studies CDF2 and DIRK2.  Those methods are not included here
//   because this file focuses on the simple unstaggered BDF form requested for a
//   compact GitHub teaching example.  The BDF1/BDF2 implementation is enough to
//   reproduce the key paper-level validation: continuity-consistent rho gives
//   much smaller Lorenz-gauge and Gauss-law residuals than naive rho deposition.
//
// Units
// -----
//   The input decks use normalized units by default: c = eps0 = mu0 = 1,
//   electron charge-to-mass ratio q/m = -1, and omega_pe = 1 for n0 = 1.  The
//   README explains how the normalized Weibel/cloud parameters correspond to the
//   physical parameter tables in the paper.
// -----------------------------------------------------------------------------

#include <algorithm>
#include <array>
#include <cassert>
#include <cctype>
#include <cmath>
#include <complex>
#include <cstdint>
#include <deque>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
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

static constexpr Real PI = 3.141592653589793238462643383279502884;
static const Complex Iunit(0.0, 1.0);

// -----------------------------------------------------------------------------
// Small vector type for particle data.
// -----------------------------------------------------------------------------

struct Vec3 {
    Real x = 0.0;
    Real y = 0.0;
    Real z = 0.0;

    Real& operator[](int d) {
        if (d == 0) return x;
        if (d == 1) return y;
        return z;
    }
    const Real& operator[](int d) const {
        if (d == 0) return x;
        if (d == 1) return y;
        return z;
    }
};

static Vec3 operator+(const Vec3& a, const Vec3& b) {
    return Vec3{a.x + b.x, a.y + b.y, a.z + b.z};
}
static Vec3 operator-(const Vec3& a, const Vec3& b) {
    return Vec3{a.x - b.x, a.y - b.y, a.z - b.z};
}
static Vec3 operator*(Real s, const Vec3& a) {
    return Vec3{s * a.x, s * a.y, s * a.z};
}
[[maybe_unused]] static Vec3 operator*(const Vec3& a, Real s) {
    return s * a;
}
[[maybe_unused]] static Vec3 operator/(const Vec3& a, Real s) {
    return Vec3{a.x / s, a.y / s, a.z / s};
}
[[maybe_unused]] static Vec3& operator+=(Vec3& a, const Vec3& b) {
    a.x += b.x; a.y += b.y; a.z += b.z; return a;
}
[[maybe_unused]] static Vec3& operator-=(Vec3& a, const Vec3& b) {
    a.x -= b.x; a.y -= b.y; a.z -= b.z; return a;
}
static Vec3& operator*=(Vec3& a, Real s) {
    a.x *= s; a.y *= s; a.z *= s; return a;
}
static Real dot(const Vec3& a, const Vec3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
static Real norm2(const Vec3& a) { return dot(a, a); }
static Real norm(const Vec3& a) { return std::sqrt(norm2(a)); }

// -----------------------------------------------------------------------------
// String and input helpers.
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

static bool parse_bool(std::string v) {
    v = lowercase(trim(v));
    return v == "1" || v == "true" || v == "yes" || v == "on";
}

static bool is_power_of_two(int n) {
    return n > 0 && (n & (n - 1)) == 0;
}

static Real wrap_periodic(Real x, Real L) {
    x -= std::floor(x / L) * L;
    // Avoid returning exactly L because the domain is [0,L).
    if (x >= L) x -= L;
    if (x < 0.0) x += L;
    return x;
}

// -----------------------------------------------------------------------------
// Run-time configuration.  The parser below accepts key = value input files.
// Unknown keys are ignored with a warning, which makes it easy to annotate decks.
// -----------------------------------------------------------------------------

struct Config {
    std::string example = "weibel";          // weibel, cloud
    std::string method = "bdf1";             // bdf1, bdf2
    std::string charge_update = "conserving";// conserving, naive

    int nx = 32;
    int ny = 32;
    Real Lx = 2.0 * PI;
    Real Ly = 2.0 * PI;

    int n_steps = 100;
    Real dt = 0.05;

    // Number of mobile electron quiet-start samples per cell for Weibel.
    // The actual Weibel particle count is two beams times this number.
    int particles_per_cell = 4;

    // Particle count for the Gaussian cloud example.  A matching number of
    // stationary ion macro-particles is used for the neutral background cloud.
    int cloud_particles = 4096;

    // Normalized constants.  The compatibility c^2 * mu0 = 1/eps0 should hold.
    Real c = 1.0;
    Real eps0 = 1.0;
    Real mu0 = 1.0;

    // Physical density scale in normalized units.  With q/m=-1, eps0=1,
    // n0=1 gives omega_pe=1.
    Real n0 = 1.0;

    // Electron charge-to-mass ratio is set by q = -macro_weight and
    // m = macro_weight, so q/m = -1.  These options allow easy experimentation.
    Real electron_charge_sign = -1.0;
    Real ion_charge_sign = +1.0;

    // Weibel parameters in normalized units.  The paper table uses v_x=c/2 and
    // v_y in [-c/100,c/100).  We keep those as the defaults.
    Real weibel_vx = 0.5;             // drift magnitude in x: beams have +/- vx
    Real weibel_vy_max = 0.01;        // random y velocity range [-vy_max,vy_max)
    Real weibel_sine_perturb = 0.002; // optional smooth seed added to vy
    int perturbation_mode = 1;

    // Cloud parameters.  The paper uses a domain [-8,8]^2, drift c/100 in both
    // x and y, and a Maxwellian thermal velocity.  The Gaussian sigma is not
    // tabulated in the paper, so it is an explicit teaching-code parameter.
    Real cloud_sigma = 1.0;
    Real cloud_drift = 0.01;
    Real cloud_thermal = 0.005;

    unsigned int seed = 12345;

    // Output controls.
    int diagnostic_stride = 1;
    int particle_sample_count = 2000;
    std::string output_prefix = "run";

    bool run_unit_tests = true;
};

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
        const std::string key = lowercase(trim(line.substr(0, eq)));
        const std::string val = trim(line.substr(eq + 1));
        if (key.empty()) continue;

        std::istringstream ss(val);
        if      (key == "example" || key == "test_case") cfg.example = lowercase(val);
        else if (key == "method") cfg.method = lowercase(val);
        else if (key == "charge_update") cfg.charge_update = lowercase(val);
        else if (key == "nx") ss >> cfg.nx;
        else if (key == "ny") ss >> cfg.ny;
        else if (key == "lx") ss >> cfg.Lx;
        else if (key == "ly") ss >> cfg.Ly;
        else if (key == "n_steps") ss >> cfg.n_steps;
        else if (key == "dt") ss >> cfg.dt;
        else if (key == "particles_per_cell") ss >> cfg.particles_per_cell;
        else if (key == "cloud_particles") ss >> cfg.cloud_particles;
        else if (key == "c") ss >> cfg.c;
        else if (key == "eps0") ss >> cfg.eps0;
        else if (key == "mu0") ss >> cfg.mu0;
        else if (key == "n0") ss >> cfg.n0;
        else if (key == "electron_charge_sign") ss >> cfg.electron_charge_sign;
        else if (key == "ion_charge_sign") ss >> cfg.ion_charge_sign;
        else if (key == "weibel_vx") ss >> cfg.weibel_vx;
        else if (key == "weibel_vy_max") ss >> cfg.weibel_vy_max;
        else if (key == "weibel_sine_perturb") ss >> cfg.weibel_sine_perturb;
        else if (key == "perturbation_mode") ss >> cfg.perturbation_mode;
        else if (key == "cloud_sigma") ss >> cfg.cloud_sigma;
        else if (key == "cloud_drift") ss >> cfg.cloud_drift;
        else if (key == "cloud_thermal") ss >> cfg.cloud_thermal;
        else if (key == "seed") ss >> cfg.seed;
        else if (key == "diagnostic_stride") ss >> cfg.diagnostic_stride;
        else if (key == "particle_sample_count") ss >> cfg.particle_sample_count;
        else if (key == "output_prefix") cfg.output_prefix = val;
        else if (key == "run_unit_tests") cfg.run_unit_tests = parse_bool(val);
        else std::cerr << "warning: unknown input key ignored: " << key << "\n";
    }

    if (!is_power_of_two(cfg.nx) || !is_power_of_two(cfg.ny)) {
        throw std::runtime_error("the internal radix-2 FFT requires nx and ny to be powers of two");
    }
    if (cfg.method != "bdf1" && cfg.method != "bdf2") {
        throw std::runtime_error("method must be bdf1 or bdf2 in this teaching code");
    }
    if (cfg.charge_update != "conserving" && cfg.charge_update != "naive") {
        throw std::runtime_error("charge_update must be conserving or naive");
    }
    if (cfg.example != "weibel" && cfg.example != "cloud" && cfg.example != "drifting_cloud") {
        throw std::runtime_error("example must be weibel or cloud");
    }
    return cfg;
}

// -----------------------------------------------------------------------------
// Periodic 2D mesh and Fourier wave numbers.
// -----------------------------------------------------------------------------

struct Grid {
    int nx = 0;
    int ny = 0;
    int N = 0;
    Real Lx = 0.0;
    Real Ly = 0.0;
    Real dx = 0.0;
    Real dy = 0.0;
    Real dA = 0.0;
    Field kx;
    Field ky;
    Field k2;

    int index(int i, int j) const {
        i = (i % nx + nx) % nx;
        j = (j % ny + ny) % ny;
        return i * ny + j;
    }
};

static int fft_mode_number(int i, int n) {
    // FFT arrays are ordered 0,1,...,n/2,-n/2+1,...,-1 for even n.
    // On a real collocated grid the Nyquist mode is its own negative.  Its
    // first derivative cannot be represented as a real grid function without
    // ambiguity, so this teaching code applies the common real-grid convention
    // of zeroing the Nyquist wave number.  We use the same convention in k2 so
    // that the wave solve, gradients, divergence, gauge diagnostic, and Gauss
    // diagnostic all use the same discrete spectral operator.
    if (n > 1 && i == n / 2) return 0;
    return (i < n / 2) ? i : i - n;
}

static Grid make_grid(const Config& cfg) {
    Grid g;
    g.nx = cfg.nx;
    g.ny = cfg.ny;
    g.N = cfg.nx * cfg.ny;
    g.Lx = cfg.Lx;
    g.Ly = cfg.Ly;
    g.dx = cfg.Lx / static_cast<Real>(cfg.nx);
    g.dy = cfg.Ly / static_cast<Real>(cfg.ny);
    g.dA = g.dx * g.dy;
    g.kx.assign(g.N, 0.0);
    g.ky.assign(g.N, 0.0);
    g.k2.assign(g.N, 0.0);

    for (int i = 0; i < g.nx; ++i) {
        const Real kxi = 2.0 * PI * static_cast<Real>(fft_mode_number(i, g.nx)) / g.Lx;
        for (int j = 0; j < g.ny; ++j) {
            const Real kyj = 2.0 * PI * static_cast<Real>(fft_mode_number(j, g.ny)) / g.Ly;
            const int id = g.index(i, j);
            g.kx[id] = kxi;
            g.ky[id] = kyj;
            g.k2[id] = kxi * kxi + kyj * kyj;
        }
    }
    return g;
}

// -----------------------------------------------------------------------------
// Internal radix-2 FFT.
//
// This is a compact iterative Cooley-Tukey FFT.  It is not as fast as FFTW, but
// it is small and dependency-free, which is exactly what we want for a teaching
// repository.  The transform convention is
//
//   forward:  f_hat[k] = sum_j f[j] exp(-i 2 pi j k / N)
//   inverse:  f[j]     = (1/N) sum_k f_hat[k] exp(+i 2 pi j k / N)
//
// The 2D transform is obtained by applying the 1D transform in y and then x.
// -----------------------------------------------------------------------------

static void fft1d(std::vector<Complex>& a, bool inverse) {
    const int n = static_cast<int>(a.size());
    if (!is_power_of_two(n)) throw std::runtime_error("fft1d length must be a power of two");

    // Bit-reversal permutation.
    for (int i = 1, j = 0; i < n; ++i) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) std::swap(a[i], a[j]);
    }

    // Danielson-Lanczos stages.
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
        for (Complex& z : a) z *= inv_n;
    }
}

static void fft2d(CField& data, const Grid& g, bool inverse) {
    std::vector<Complex> tmp;

    // Transform each row in y.
    tmp.resize(g.ny);
    for (int i = 0; i < g.nx; ++i) {
        for (int j = 0; j < g.ny; ++j) tmp[j] = data[g.index(i, j)];
        fft1d(tmp, inverse);
        for (int j = 0; j < g.ny; ++j) data[g.index(i, j)] = tmp[j];
    }

    // Transform each column in x.
    tmp.resize(g.nx);
    for (int j = 0; j < g.ny; ++j) {
        for (int i = 0; i < g.nx; ++i) tmp[i] = data[g.index(i, j)];
        fft1d(tmp, inverse);
        for (int i = 0; i < g.nx; ++i) data[g.index(i, j)] = tmp[i];
    }
}

static CField fft_real(const Field& f, const Grid& g) {
    CField out(g.N);
    for (int id = 0; id < g.N; ++id) out[id] = Complex(f[id], 0.0);
    fft2d(out, g, false);
    return out;
}

static Field ifft_real(CField fhat, const Grid& g) {
    fft2d(fhat, g, true);
    Field out(g.N);
    for (int id = 0; id < g.N; ++id) out[id] = fhat[id].real();
    return out;
}

static Field spectral_derivative(const Field& f, const Grid& g, int dir) {
    CField h = fft_real(f, g);
    for (int id = 0; id < g.N; ++id) {
        const Real kk = (dir == 0) ? g.kx[id] : g.ky[id];
        h[id] *= Iunit * kk;
    }
    return ifft_real(std::move(h), g);
}

static Field spectral_divergence(const std::array<Field, 3>& V, const Grid& g) {
    CField hx = fft_real(V[0], g);
    CField hy = fft_real(V[1], g);
    CField div(g.N, Complex(0.0, 0.0));
    for (int id = 0; id < g.N; ++id) div[id] = Iunit * (g.kx[id] * hx[id] + g.ky[id] * hy[id]);
    return ifft_real(std::move(div), g);
}

static std::array<Field, 3> spectral_gradient_scalar(const Field& f, const Grid& g) {
    std::array<Field, 3> grad{Field(g.N, 0.0), Field(g.N, 0.0), Field(g.N, 0.0)};
    grad[0] = spectral_derivative(f, g, 0);
    grad[1] = spectral_derivative(f, g, 1);
    return grad;
}

static std::array<std::array<Field, 2>, 3> spectral_gradient_vector(const std::array<Field, 3>& A,
                                                                    const Grid& g) {
    // gradA[component][direction], where direction 0=x and 1=y.
    std::array<std::array<Field, 2>, 3> gradA;
    for (int comp = 0; comp < 3; ++comp) {
        gradA[comp][0] = spectral_derivative(A[comp], g, 0);
        gradA[comp][1] = spectral_derivative(A[comp], g, 1);
    }
    return gradA;
}

static std::array<Field, 3> spectral_curl_A(const std::array<Field, 3>& A, const Grid& g) {
    // 2D curl of A(x,y):
    //   Bx = d_y A_z, By = -d_x A_z, Bz = d_x A_y - d_y A_x.
    std::array<Field, 3> B{Field(g.N, 0.0), Field(g.N, 0.0), Field(g.N, 0.0)};
    const Field dAz_dx = spectral_derivative(A[2], g, 0);
    const Field dAz_dy = spectral_derivative(A[2], g, 1);
    const Field dAy_dx = spectral_derivative(A[1], g, 0);
    const Field dAx_dy = spectral_derivative(A[0], g, 1);
    for (int id = 0; id < g.N; ++id) {
        B[0][id] = dAz_dy[id];
        B[1][id] = -dAz_dx[id];
        B[2][id] = dAy_dx[id] - dAx_dy[id];
    }
    return B;
}

// -----------------------------------------------------------------------------
// Simple field algebra.
// -----------------------------------------------------------------------------

static Field zero_field(const Grid& g) { return Field(g.N, 0.0); }

static Real rms(const Field& f) {
    if (f.empty()) return 0.0;
    long double s = 0.0;
    for (Real v : f) s += static_cast<long double>(v) * static_cast<long double>(v);
    return std::sqrt(static_cast<Real>(s / static_cast<long double>(f.size())));
}

static Real rms_vec(const std::array<Field, 3>& V) {
    if (V[0].empty()) return 0.0;
    long double s = 0.0;
    for (std::size_t i = 0; i < V[0].size(); ++i) {
        s += static_cast<long double>(V[0][i]) * V[0][i]
           + static_cast<long double>(V[1][i]) * V[1][i]
           + static_cast<long double>(V[2][i]) * V[2][i];
    }
    return std::sqrt(static_cast<Real>(s / static_cast<long double>(V[0].size())));
}

static Real integral_square_vec(const std::array<Field, 3>& V, const Grid& g) {
    long double s = 0.0;
    for (int id = 0; id < g.N; ++id) {
        s += static_cast<long double>(V[0][id]) * V[0][id]
           + static_cast<long double>(V[1][id]) * V[1][id]
           + static_cast<long double>(V[2][id]) * V[2][id];
    }
    return static_cast<Real>(s * g.dA);
}

// -----------------------------------------------------------------------------
// Quadratic B-spline scatter/gather.
//
// We use the same 1D weights in x and y and take a tensor product.  The weights
// are centered on the nearest grid node.  They sum to one and are periodic.
// -----------------------------------------------------------------------------

struct Weight1D {
    int i[3]{};
    Real w[3]{};
};

static Weight1D quadratic_weights(Real x, Real dx, int n) {
    // Convert to grid-index units and find the nearest grid node.
    const Real s = x / dx;
    int ic = static_cast<int>(std::floor(s + 0.5));
    const Real r = s - static_cast<Real>(ic); // in approximately [-1/2, 1/2)

    Weight1D out;
    out.i[0] = (ic - 1 + n) % n;
    out.i[1] = (ic + n) % n;
    out.i[2] = (ic + 1 + n) % n;

    out.w[0] = 0.5 * (0.5 - r) * (0.5 - r);
    out.w[1] = 0.75 - r * r;
    out.w[2] = 0.5 * (0.5 + r) * (0.5 + r);
    return out;
}

static void scatter_scalar(Field& f, const Grid& g, Real x, Real y, Real value) {
    const Weight1D wx = quadratic_weights(x, g.dx, g.nx);
    const Weight1D wy = quadratic_weights(y, g.dy, g.ny);
    for (int a = 0; a < 3; ++a) {
        for (int b = 0; b < 3; ++b) {
            f[g.index(wx.i[a], wy.i[b])] += value * wx.w[a] * wy.w[b] / g.dA;
        }
    }
}

static Real gather_scalar(const Field& f, const Grid& g, Real x, Real y) {
    const Weight1D wx = quadratic_weights(x, g.dx, g.nx);
    const Weight1D wy = quadratic_weights(y, g.dy, g.ny);
    Real value = 0.0;
    for (int a = 0; a < 3; ++a) {
        for (int b = 0; b < 3; ++b) {
            value += f[g.index(wx.i[a], wy.i[b])] * wx.w[a] * wy.w[b];
        }
    }
    return value;
}

static Vec3 gather_vector(const std::array<Field, 3>& V, const Grid& g, Real x, Real y) {
    return Vec3{gather_scalar(V[0], g, x, y),
                gather_scalar(V[1], g, x, y),
                gather_scalar(V[2], g, x, y)};
}

// -----------------------------------------------------------------------------
// Particle data and relativistic velocity/momentum conversions.
// -----------------------------------------------------------------------------

struct Particle {
    Real x = 0.0;
    Real y = 0.0;
    Vec3 P;       // generalized/canonical momentum: m gamma v + q A
    Vec3 v;       // current velocity
    Vec3 v_prev;  // previous velocity, used in IAEM v* = 2 v^n - v^{n-1}
    Real q = 0.0; // macro-particle charge
    Real m = 0.0; // macro-particle mass
};

static Real gamma_from_v(const Vec3& v, Real c) {
    const Real beta2 = norm2(v) / (c * c);
    const Real safe = std::max(1.0e-14, 1.0 - beta2);
    return 1.0 / std::sqrt(safe);
}

static Vec3 velocity_from_canonical(const Vec3& P, Real q, Real m, const Vec3& A, Real c) {
    const Vec3 p = P - q * A; // mechanical momentum m gamma v
    const Real denom = std::sqrt(c * c * norm2(p) + (m * c * c) * (m * c * c));
    if (denom == 0.0) return Vec3{};
    return (c * c / denom) * p;
}

static Vec3 canonical_from_velocity(const Vec3& v, Real q, Real m, const Vec3& A, Real c) {
    const Real gam = gamma_from_v(v, c);
    return (m * gam) * v + q * A;
}

// -----------------------------------------------------------------------------
// Initial particle loading.
// -----------------------------------------------------------------------------

struct ParticleSet {
    std::vector<Particle> electrons; // mobile particles
    std::vector<Particle> ions;      // stationary particles used only for cloud rho
    Field fixed_ion_rho;             // uniform ion background for Weibel, cloud ions otherwise
};

static void initialize_weibel_particles(const Config& cfg, const Grid& g, ParticleSet& ps) {
    const int nsub = std::max(1, static_cast<int>(std::round(std::sqrt(static_cast<Real>(cfg.particles_per_cell)))));
    const int samples_per_cell = nsub * nsub;
    const int total_electrons = 2 * g.nx * g.ny * samples_per_cell;
    const Real domain_area = g.Lx * g.Ly;
    const Real macro_weight = cfg.n0 * domain_area / static_cast<Real>(total_electrons);
    const Real q = cfg.electron_charge_sign * macro_weight;
    const Real m = macro_weight;

    std::mt19937 rng(cfg.seed);
    std::uniform_real_distribution<Real> uy(-cfg.weibel_vy_max, cfg.weibel_vy_max);

    ps.electrons.clear();
    ps.electrons.reserve(static_cast<std::size_t>(total_electrons));

    for (int i = 0; i < g.nx; ++i) {
        for (int j = 0; j < g.ny; ++j) {
            for (int a = 0; a < nsub; ++a) {
                for (int b = 0; b < nsub; ++b) {
                    const Real x = (static_cast<Real>(i) + (static_cast<Real>(a) + 0.5) / nsub) * g.dx;
                    const Real y = (static_cast<Real>(j) + (static_cast<Real>(b) + 0.5) / nsub) * g.dy;
                    const Real sine_seed = cfg.weibel_sine_perturb * cfg.c *
                        std::sin(2.0 * PI * cfg.perturbation_mode * x / g.Lx);

                    for (int beam = -1; beam <= 1; beam += 2) {
                        Particle p;
                        p.x = x;
                        p.y = y;
                        p.q = q;
                        p.m = m;
                        p.v = Vec3{static_cast<Real>(beam) * cfg.weibel_vx * cfg.c,
                                   uy(rng) + sine_seed,
                                   0.0};
                        // A is zero at initialization, so P = m gamma v.
                        p.P = canonical_from_velocity(p.v, p.q, p.m, Vec3{}, cfg.c);
                        p.v_prev = p.v; // first step uses v* = v; see README notes.
                        ps.electrons.push_back(p);
                    }
                }
            }
        }
    }

    // Uniform immobile positive ion background neutralizes the electrons.
    ps.fixed_ion_rho.assign(g.N, cfg.n0);
}

static void initialize_cloud_particles(const Config& cfg, const Grid& g, ParticleSet& ps) {
    const int Np = cfg.cloud_particles;
    const Real domain_area = g.Lx * g.Ly;
    const Real macro_weight = cfg.n0 * domain_area / static_cast<Real>(Np);
    const Real qe = cfg.electron_charge_sign * macro_weight;
    const Real qi = cfg.ion_charge_sign * macro_weight;
    const Real me = macro_weight;
    const Real mi = macro_weight; // static; only the charge is used.

    std::mt19937 rng(cfg.seed);
    std::normal_distribution<Real> normal(0.0, 1.0);

    ps.electrons.clear();
    ps.ions.clear();
    ps.electrons.reserve(static_cast<std::size_t>(Np));
    ps.ions.reserve(static_cast<std::size_t>(Np));

    for (int n = 0; n < Np; ++n) {
        const Real xc = 0.5 * g.Lx + cfg.cloud_sigma * normal(rng);
        const Real yc = 0.5 * g.Ly + cfg.cloud_sigma * normal(rng);
        const Real x = wrap_periodic(xc, g.Lx);
        const Real y = wrap_periodic(yc, g.Ly);

        Particle e;
        e.x = x;
        e.y = y;
        e.q = qe;
        e.m = me;
        e.v = Vec3{cfg.cloud_drift * cfg.c + cfg.cloud_thermal * normal(rng),
                   cfg.cloud_drift * cfg.c + cfg.cloud_thermal * normal(rng),
                   0.0};
        // Keep the teaching example safely sub-relativistic even if a rare
        // Gaussian draw is large.
        const Real sp = norm(e.v);
        const Real vmax = 0.2 * cfg.c;
        if (sp > vmax) e.v *= (vmax / sp);
        e.P = canonical_from_velocity(e.v, e.q, e.m, Vec3{}, cfg.c);
        e.v_prev = e.v;
        ps.electrons.push_back(e);

        Particle ion;
        ion.x = x;
        ion.y = y;
        ion.q = qi;
        ion.m = mi;
        ion.v = Vec3{};
        ion.v_prev = Vec3{};
        ion.P = Vec3{};
        ps.ions.push_back(ion);
    }

    // Deposit the stationary ion cloud once.  It is included in total rho but
    // has zero current, so the continuity update naturally leaves it fixed.
    ps.fixed_ion_rho.assign(g.N, 0.0);
    for (const Particle& ion : ps.ions) scatter_scalar(ps.fixed_ion_rho, g, ion.x, ion.y, ion.q);
}

static ParticleSet initialize_particles(const Config& cfg, const Grid& g) {
    ParticleSet ps;
    if (cfg.example == "weibel") initialize_weibel_particles(cfg, g, ps);
    else initialize_cloud_particles(cfg, g, ps);
    return ps;
}

// -----------------------------------------------------------------------------
// Scatter charge and current from particles.
// -----------------------------------------------------------------------------

static Field deposit_total_charge_naive(const std::vector<Particle>& electrons,
                                        const ParticleSet& ps,
                                        const Grid& g) {
    Field rho = ps.fixed_ion_rho;
    for (const Particle& p : electrons) scatter_scalar(rho, g, p.x, p.y, p.q);
    return rho;
}

static std::array<Field, 3> deposit_current(const std::vector<Particle>& electrons, const Grid& g) {
    std::array<Field, 3> J{zero_field(g), zero_field(g), zero_field(g)};
    for (const Particle& p : electrons) {
        scatter_scalar(J[0], g, p.x, p.y, p.q * p.v.x);
        scatter_scalar(J[1], g, p.x, p.y, p.q * p.v.y);
        scatter_scalar(J[2], g, p.x, p.y, p.q * p.v.z);
    }
    return J;
}

// -----------------------------------------------------------------------------
// BDF helpers.
//
// For BDF-k, D_t u^{n+1} = (a0 u^{n+1} + a1 u^n + ... + ak u^{n+1-k})/dt.
// The wave equation uses D_t(D_t u), which gives convolution coefficients b.
// -----------------------------------------------------------------------------

struct BDFCoeffs {
    std::string active_name;
    std::vector<Real> a; // a[0] is the coefficient multiplying the new state.
};

static BDFCoeffs bdf_coefficients_for_step(const Config& cfg, int /*step_number*/) {
    // We initialize the time history with repeated copies of the t=0 fields.
    // That is only first-order accurate as a start-up procedure, but it is
    // algebraically consistent with the selected BDF formula and keeps the
    // gauge/continuity/Gauss validation clean.  A production code would usually
    // initialize BDF2 with a dedicated one-step method or analytic histories.
    if (cfg.method == "bdf2") return BDFCoeffs{"bdf2", {1.5, -2.0, 0.5}};
    return BDFCoeffs{"bdf1", {1.0, -1.0}};
}

static std::vector<Real> convolve_coefficients(const std::vector<Real>& a) {
    std::vector<Real> b(2 * a.size() - 1, 0.0);
    for (std::size_t i = 0; i < a.size(); ++i) {
        for (std::size_t j = 0; j < a.size(); ++j) b[i + j] += a[i] * a[j];
    }
    return b;
}

struct History {
    // state[0] is the current value u^n, state[1] is u^{n-1}, etc.
    std::deque<Field> state;
    int max_size = 5;

    void initialize(const Field& f, int count) {
        max_size = count;
        state.clear();
        for (int i = 0; i < count; ++i) state.push_back(f);
    }

    const Field& current() const { return state.front(); }
    Field& current() { return state.front(); }

    const Field& past(int i) const {
        if (i < 0 || i >= static_cast<int>(state.size())) throw std::runtime_error("history index out of range");
        return state[static_cast<std::size_t>(i)];
    }

    void push_new(Field f) {
        state.push_front(std::move(f));
        while (static_cast<int>(state.size()) > max_size) state.pop_back();
    }
};

struct FieldState {
    History phi;
    std::array<History, 3> A;
    History rho;
    std::array<Field, 3> J; // current source at the current time level
};

static Field solve_initial_phi_from_poisson(const Field& rho, const Config& cfg, const Grid& g) {
    CField rh = fft_real(rho, g);
    CField ph(g.N, Complex(0.0, 0.0));
    for (int id = 0; id < g.N; ++id) {
        if (g.k2[id] > 0.0) ph[id] = rh[id] / (cfg.eps0 * g.k2[id]);
        else ph[id] = Complex(0.0, 0.0); // zero-mean potential gauge choice.
    }
    return ifft_real(std::move(ph), g);
}

static Field solve_bdf_wave_scalar(const Field& source_new,
                                   const History& u_hist,
                                   const BDFCoeffs& bdf,
                                   const Config& cfg,
                                   const Grid& g) {
    const std::vector<Real> b = convolve_coefficients(bdf.a);
    const Real scale = 1.0 / (cfg.c * cfg.c * cfg.dt * cfg.dt);

    CField rhs = fft_real(source_new, g);
    for (std::size_t l = 1; l < b.size(); ++l) {
        CField uh = fft_real(u_hist.past(static_cast<int>(l - 1)), g);
        for (int id = 0; id < g.N; ++id) rhs[id] -= scale * b[l] * uh[id];
    }

    for (int id = 0; id < g.N; ++id) {
        const Real denom = scale * b[0] + g.k2[id];
        rhs[id] /= denom;
    }
    return ifft_real(std::move(rhs), g);
}

static Field bdf_first_derivative(const Field& u_new,
                                  const History& u_hist,
                                  const BDFCoeffs& bdf,
                                  const Grid& g,
                                  Real dt) {
    Field out(g.N, 0.0);
    for (int id = 0; id < g.N; ++id) out[id] = bdf.a[0] * u_new[id];
    for (std::size_t l = 1; l < bdf.a.size(); ++l) {
        const Field& old = u_hist.past(static_cast<int>(l - 1));
        for (int id = 0; id < g.N; ++id) out[id] += bdf.a[l] * old[id];
    }
    for (Real& v : out) v /= dt;
    return out;
}

static Field update_charge_from_continuity(const History& rho_hist,
                                           const std::array<Field, 3>& J_new,
                                           const BDFCoeffs& bdf,
                                           const Config& cfg,
                                           const Grid& g) {
    const Field divJ = spectral_divergence(J_new, g);
    Field rho_new(g.N, 0.0);
    for (int id = 0; id < g.N; ++id) {
        Real history_sum = 0.0;
        for (std::size_t l = 1; l < bdf.a.size(); ++l) {
            history_sum += bdf.a[l] * rho_hist.past(static_cast<int>(l - 1))[id];
        }
        rho_new[id] = -(cfg.dt * divJ[id] + history_sum) / bdf.a[0];
    }
    return rho_new;
}

// -----------------------------------------------------------------------------
// Diagnostics.
// -----------------------------------------------------------------------------

struct Diagnostics {
    Real gauge_rms = 0.0;
    Real gauss_rms = 0.0;
    Real continuity_rms = 0.0;
    Real Bz_l2 = 0.0;
    Real B_l2 = 0.0;
    Real E_l2 = 0.0;
    Real rho_l2 = 0.0;
    Real J_l2 = 0.0;
    Real field_energy = 0.0;
    Real kinetic_energy = 0.0;
    Real total_energy = 0.0;
    Real max_speed = 0.0;
    Real mean_gamma = 0.0;
};

static Diagnostics compute_diagnostics(const std::vector<Particle>& electrons,
                                       const FieldState& state,
                                       const Field& phi_new_for_derivative,
                                       const std::array<Field, 3>& A_new_for_derivative,
                                       const Field& rho_current,
                                       const std::array<Field, 3>& J_current,
                                       const BDFCoeffs& bdf_for_derivatives,
                                       const Config& cfg,
                                       const Grid& g) {
    Diagnostics d;

    // Lorenz gauge residual: (1/c^2) d_t phi + div A.
    const Field phi_t = bdf_first_derivative(phi_new_for_derivative, state.phi, bdf_for_derivatives, g, cfg.dt);
    const Field divA = spectral_divergence(A_new_for_derivative, g);
    Field gauge(g.N, 0.0);
    for (int id = 0; id < g.N; ++id) gauge[id] = phi_t[id] / (cfg.c * cfg.c) + divA[id];
    d.gauge_rms = rms(gauge);

    // Electric field: E = -grad phi - d_t A.
    const auto grad_phi = spectral_gradient_scalar(phi_new_for_derivative, g);
    std::array<Field, 3> A_t{zero_field(g), zero_field(g), zero_field(g)};
    std::array<Field, 3> E{zero_field(g), zero_field(g), zero_field(g)};
    for (int comp = 0; comp < 3; ++comp) {
        A_t[comp] = bdf_first_derivative(A_new_for_derivative[comp], state.A[comp], bdf_for_derivatives, g, cfg.dt);
        for (int id = 0; id < g.N; ++id) E[comp][id] = -grad_phi[comp][id] - A_t[comp][id];
    }

    const Field divE = spectral_divergence(E, g);
    Field gauss(g.N, 0.0);
    for (int id = 0; id < g.N; ++id) gauss[id] = divE[id] - rho_current[id] / cfg.eps0;
    d.gauss_rms = rms(gauss);

    const Field divJ = spectral_divergence(J_current, g);
    const Field rho_t = bdf_first_derivative(rho_current, state.rho, bdf_for_derivatives, g, cfg.dt);
    Field cont(g.N, 0.0);
    for (int id = 0; id < g.N; ++id) cont[id] = rho_t[id] + divJ[id];
    d.continuity_rms = rms(cont);

    const auto B = spectral_curl_A(A_new_for_derivative, g);
    d.Bz_l2 = rms(B[2]);
    d.B_l2 = rms_vec(B);
    d.E_l2 = rms_vec(E);
    d.rho_l2 = rms(rho_current);
    d.J_l2 = rms_vec(J_current);

    d.field_energy = 0.5 * cfg.eps0 * integral_square_vec(E, g)
                   + 0.5 / cfg.mu0 * integral_square_vec(B, g);

    long double kinetic = 0.0;
    long double gamma_sum = 0.0;
    for (const Particle& p : electrons) {
        const Real gam = gamma_from_v(p.v, cfg.c);
        gamma_sum += gam;
        kinetic += (gam - 1.0) * p.m * cfg.c * cfg.c;
        d.max_speed = std::max(d.max_speed, norm(p.v));
    }
    d.kinetic_energy = static_cast<Real>(kinetic);
    d.mean_gamma = electrons.empty() ? 0.0 : static_cast<Real>(gamma_sum / electrons.size());
    d.total_energy = d.field_energy + d.kinetic_energy;
    return d;
}

static void write_config_echo(const Config& cfg, const Grid& g, const ParticleSet& ps) {
    std::ofstream out(cfg.output_prefix + "_config_echo.txt");
    out << std::setprecision(17);
    out << "example = " << cfg.example << "\n";
    out << "method = " << cfg.method << "\n";
    out << "charge_update = " << cfg.charge_update << "\n";
    out << "nx = " << cfg.nx << "\n";
    out << "ny = " << cfg.ny << "\n";
    out << "Lx = " << cfg.Lx << "\n";
    out << "Ly = " << cfg.Ly << "\n";
    out << "dt = " << cfg.dt << "\n";
    out << "n_steps = " << cfg.n_steps << "\n";
    out << "c = " << cfg.c << "\n";
    out << "eps0 = " << cfg.eps0 << "\n";
    out << "mu0 = " << cfg.mu0 << "\n";
    out << "n0 = " << cfg.n0 << "\n";
    out << "mobile_electrons = " << ps.electrons.size() << "\n";
    out << "static_ions = " << ps.ions.size() << "\n";
    out << "cell_area = " << g.dA << "\n";
    out << "output_prefix = " << cfg.output_prefix << "\n";
}

static void write_particle_sample(const Config& cfg, const Grid& g, const std::vector<Particle>& electrons) {
    std::ofstream out(cfg.output_prefix + "_particle_sample.csv");
    out << std::setprecision(17);
    out << "id,x,y,x_centered,y_centered,vx,vy,vz,q,m,gamma\n";
    const int count = std::min<int>(cfg.particle_sample_count, static_cast<int>(electrons.size()));
    for (int i = 0; i < count; ++i) {
        const Particle& p = electrons[static_cast<std::size_t>(i)];
        const Real xc = p.x - 0.5 * g.Lx;
        const Real yc = p.y - 0.5 * g.Ly;
        out << i << ',' << p.x << ',' << p.y << ',' << xc << ',' << yc << ','
            << p.v.x << ',' << p.v.y << ',' << p.v.z << ',' << p.q << ',' << p.m << ','
            << gamma_from_v(p.v, cfg.c) << "\n";
    }
}

// -----------------------------------------------------------------------------
// Unit tests.  These are small deterministic checks for the in-file FFT and the
// spectral derivative.  They are not exhaustive, but they catch the most common
// sign/scaling mistakes.
// -----------------------------------------------------------------------------

static void run_unit_tests(const Config& cfg) {
    Config tcfg = cfg;
    tcfg.nx = 16;
    tcfg.ny = 8;
    tcfg.Lx = 2.0 * PI;
    tcfg.Ly = 2.0 * PI;
    Grid g = make_grid(tcfg);

    Field f(g.N, 0.0);
    for (int i = 0; i < g.nx; ++i) {
        const Real x = i * g.dx;
        for (int j = 0; j < g.ny; ++j) {
            const Real y = j * g.dy;
            f[g.index(i, j)] = std::sin(3.0 * x) + 0.5 * std::cos(2.0 * y);
        }
    }

    Field fx = spectral_derivative(f, g, 0);
    Field fy = spectral_derivative(f, g, 1);
    Real errx = 0.0, erry = 0.0;
    for (int i = 0; i < g.nx; ++i) {
        const Real x = i * g.dx;
        for (int j = 0; j < g.ny; ++j) {
            const Real y = j * g.dy;
            const int id = g.index(i, j);
            const Real exact_x = 3.0 * std::cos(3.0 * x);
            const Real exact_y = -1.0 * std::sin(2.0 * y);
            errx = std::max(errx, std::abs(fx[id] - exact_x));
            erry = std::max(erry, std::abs(fy[id] - exact_y));
        }
    }

    if (errx > 1.0e-10 || erry > 1.0e-10) {
        std::ostringstream msg;
        msg << "FFT derivative unit test failed: errx=" << errx << " erry=" << erry;
        throw std::runtime_error(msg.str());
    }
}

// -----------------------------------------------------------------------------
// Main PIC loop.
// -----------------------------------------------------------------------------

static void advance_particles_positions(std::vector<Particle>& electrons, const Config& cfg, const Grid& g) {
    for (Particle& p : electrons) {
        p.x = wrap_periodic(p.x + cfg.dt * p.v.x, g.Lx);
        p.y = wrap_periodic(p.y + cfg.dt * p.v.y, g.Ly);
    }
}

static void push_particles_momentum_and_velocity(std::vector<Particle>& electrons,
                                                 const std::array<Field, 3>& A_new,
                                                 const std::array<Field, 3>& grad_phi,
                                                 const std::array<std::array<Field, 2>, 3>& grad_A,
                                                 const Config& cfg,
                                                 const Grid& g) {
    for (Particle& p : electrons) {
        const Vec3 A_p = gather_vector(A_new, g, p.x, p.y);
        const Vec3 grad_phi_p = gather_vector(grad_phi, g, p.x, p.y);

        // Gather the 2D gradient of each A component.
        // gradA_dot_v has components indexed by spatial derivative direction:
        //   component x = sum_l v_l * d_x A_l,
        //   component y = sum_l v_l * d_y A_l,
        //   component z = 0 in this 2D x-y code.
        const Vec3 vstar = 2.0 * p.v - p.v_prev;
        Real dAx_dx = gather_scalar(grad_A[0][0], g, p.x, p.y);
        Real dAx_dy = gather_scalar(grad_A[0][1], g, p.x, p.y);
        Real dAy_dx = gather_scalar(grad_A[1][0], g, p.x, p.y);
        Real dAy_dy = gather_scalar(grad_A[1][1], g, p.x, p.y);
        Real dAz_dx = gather_scalar(grad_A[2][0], g, p.x, p.y);
        Real dAz_dy = gather_scalar(grad_A[2][1], g, p.x, p.y);

        const Vec3 gradA_dot_v{dAx_dx * vstar.x + dAy_dx * vstar.y + dAz_dx * vstar.z,
                               dAx_dy * vstar.x + dAy_dy * vstar.y + dAz_dy * vstar.z,
                               0.0};

        const Vec3 force_like = (-1.0) * grad_phi_p + gradA_dot_v;
        const Vec3 P_new = p.P + (p.q * cfg.dt) * force_like;
        const Vec3 v_new = velocity_from_canonical(P_new, p.q, p.m, A_p, cfg.c);

        p.v_prev = p.v;
        p.v = v_new;
        p.P = P_new;
    }
}

static void run_simulation(const Config& cfg) {
    if (cfg.run_unit_tests) run_unit_tests(cfg);

    const Grid g = make_grid(cfg);
    ParticleSet ps = initialize_particles(cfg, g);

    // Initial total charge and current.  Initial phi is obtained from Poisson so
    // that Gauss's law starts small; A starts at zero for a clean teaching setup.
    Field rho0 = deposit_total_charge_naive(ps.electrons, ps, g);
    std::array<Field, 3> J0 = deposit_current(ps.electrons, g);
    Field phi0 = solve_initial_phi_from_poisson(rho0, cfg, g);
    std::array<Field, 3> A0{zero_field(g), zero_field(g), zero_field(g)};

    FieldState state;
    state.phi.initialize(phi0, 5);
    for (int comp = 0; comp < 3; ++comp) state.A[comp].initialize(A0[comp], 5);
    state.rho.initialize(rho0, 5);
    state.J = J0;

    write_config_echo(cfg, g, ps);

    std::ofstream diag(cfg.output_prefix + "_diagnostics.csv");
    diag << std::setprecision(17);
    diag << "step,time,requested_method,active_method,charge_update,"
         << "gauge_rms,gauss_rms,continuity_rms,Bz_l2,B_l2,E_l2,rho_l2,J_l2,"
         << "field_energy,kinetic_energy,total_energy,max_speed,mean_gamma\n";

    // Diagnostics at t=0 use BDF1 with duplicated histories, corresponding to
    // zero initial potential time derivatives.
    BDFCoeffs bdf0{"bdf1", {1.0, -1.0}};
    Diagnostics d0 = compute_diagnostics(ps.electrons, state, state.phi.current(),
                                         {state.A[0].current(), state.A[1].current(), state.A[2].current()},
                                         state.rho.current(), state.J, bdf0, cfg, g);
    diag << 0 << ',' << 0.0 << ',' << cfg.method << ',' << "initial" << ',' << cfg.charge_update << ','
         << d0.gauge_rms << ',' << d0.gauss_rms << ',' << d0.continuity_rms << ','
         << d0.Bz_l2 << ',' << d0.B_l2 << ',' << d0.E_l2 << ',' << d0.rho_l2 << ',' << d0.J_l2 << ','
         << d0.field_energy << ',' << d0.kinetic_energy << ',' << d0.total_energy << ','
         << d0.max_speed << ',' << d0.mean_gamma << "\n";

    for (int step = 1; step <= cfg.n_steps; ++step) {
        const BDFCoeffs bdf = bdf_coefficients_for_step(cfg, step);

        // 1. IAEM position update: x^{n+1} = x^n + v^n dt.
        advance_particles_positions(ps.electrons, cfg, g);

        // 2. Deposit J^{n+1} using x^{n+1} and v^n.  In this code p.v still
        //    contains v^n because the momentum update has not happened yet.
        std::array<Field, 3> J_new = deposit_current(ps.electrons, g);

        // 3. Obtain rho^{n+1}.  The conserving option uses the same BDF
        //    continuity law as the field solve.  The naive option deliberately
        //    redeposits rho from particles and is included for paper-style
        //    comparison plots.
        Field rho_new;
        if (cfg.charge_update == "conserving") {
            rho_new = update_charge_from_continuity(state.rho, J_new, bdf, cfg, g);
        } else {
            rho_new = deposit_total_charge_naive(ps.electrons, ps, g);
        }

        // 4. Spectral BDF wave solves for phi^{n+1} and A^{n+1}.
        Field phi_source(g.N, 0.0);
        for (int id = 0; id < g.N; ++id) phi_source[id] = rho_new[id] / cfg.eps0;
        Field phi_new = solve_bdf_wave_scalar(phi_source, state.phi, bdf, cfg, g);

        std::array<Field, 3> A_new{zero_field(g), zero_field(g), zero_field(g)};
        for (int comp = 0; comp < 3; ++comp) {
            Field A_source(g.N, 0.0);
            for (int id = 0; id < g.N; ++id) A_source[id] = cfg.mu0 * J_new[comp][id];
            A_new[comp] = solve_bdf_wave_scalar(A_source, state.A[comp], bdf, cfg, g);
        }

        // 5. Spectral derivatives on the grid and quadratic gather to particles.
        const auto grad_phi = spectral_gradient_scalar(phi_new, g);
        const auto grad_A = spectral_gradient_vector(A_new, g);

        // 6. IAEM momentum update and relativistic P -> v conversion.
        push_particles_momentum_and_velocity(ps.electrons, A_new, grad_phi, grad_A, cfg, g);

        // 7. Diagnostics before overwriting the histories.  The derivative
        //    routines need access to old states in state.* and the new fields.
        if (step % std::max(1, cfg.diagnostic_stride) == 0 || step == cfg.n_steps) {
            Diagnostics d = compute_diagnostics(ps.electrons, state, phi_new, A_new, rho_new, J_new, bdf, cfg, g);
            diag << step << ',' << step * cfg.dt << ',' << cfg.method << ',' << bdf.active_name << ','
                 << cfg.charge_update << ',' << d.gauge_rms << ',' << d.gauss_rms << ',' << d.continuity_rms << ','
                 << d.Bz_l2 << ',' << d.B_l2 << ',' << d.E_l2 << ',' << d.rho_l2 << ',' << d.J_l2 << ','
                 << d.field_energy << ',' << d.kinetic_energy << ',' << d.total_energy << ','
                 << d.max_speed << ',' << d.mean_gamma << "\n";
        }

        // 8. Shift time histories.
        state.phi.push_new(std::move(phi_new));
        for (int comp = 0; comp < 3; ++comp) state.A[comp].push_new(std::move(A_new[comp]));
        state.rho.push_new(std::move(rho_new));
        state.J = std::move(J_new);
    }

    write_particle_sample(cfg, g, ps.electrons);

    std::cout << "wrote " << cfg.output_prefix << "_diagnostics.csv\n";
    std::cout << "wrote " << cfg.output_prefix << "_particle_sample.csv\n";
    std::cout << "wrote " << cfg.output_prefix << "_config_echo.txt\n";
}

int main(int argc, char** argv) {
    try {
        if (argc != 2) {
            std::cerr << "usage: " << argv[0] << " input_file.txt\n";
            return 2;
        }
        const Config cfg = read_config(argv[1]);
        run_simulation(cfg);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
