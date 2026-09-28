// -----------------------------------------------------------------------------
// ec_pic_nyquist_projected.cpp
// -----------------------------------------------------------------------------
//
// OpenMP C++17 teaching implementation of a relativistic, energy-conserving,
// unstaggered-potential particle-in-cell (PIC) method using a generalized-
// momentum Higuera-Cary (GM-HC) particle push.
//
// This file is intentionally self-contained and uses OpenMP for shared-memory
// parallelism.  There is no MPI and no external FFT dependency.  The priority is
// still readability: every major step in the method is present in this file and
// heavily commented so that the implementation can be used as a reference while
// developing production parallel versions.
//
// To preserve the serial code's numerical path as closely as possible, OpenMP is
// applied to independent particle/mesh transforms and per-particle gathers/pushes.
// Conservation-critical current deposition uses thread-private meshes followed
// by a deterministic mesh reduction, avoiding atomics while retaining the same
// scatter formula. Scalar diagnostic reductions remain in their reference form.
//
// Typical Linux build and run:
//
//   g++ -O3 -std=c++17 -fopenmp ec_pic_nyquist_projected.cpp -o ec_pic_nyquist_projected
//   OMP_NUM_THREADS=8 OMP_PROC_BIND=close OMP_PLACES=cores ./ec_pic_nyquist_projected input.txt
//
// Mathematical variables stored by the code
// ----------------------------------------
// Mesh variables are the Lorenz-gauge potentials and their time derivatives,
//
//     phi  : scalar potential
//     psi  : time derivative of phi
//     A    : vector potential
//     U    : time derivative of A
//
// plus the charge density rho and orbit-centered current J.  Fields are recovered
// only diagnostically from
//
//     E = -grad(phi) - U,        B = curl(A).
//
// Particle variables are position x, velocity v, mechanical momentum p, and
// canonical/generalized momentum
//
//     P = p + q A_h(x),          p = m gamma(v) v.
//
// The force update is written in canonical variables so that the pusher does not
// explicitly form a finite-difference approximation to A_t.  The time dependence
// of A is accounted for by the Crank-Nicolson identity for A and by the orbit
// chain rule below.
//
// Relativistic GM-HC orbit-averaged time step
// -------------------------------------------
// Each nonlinear map evaluation performs the following coupled operations:
//
//   1. Use a guessed work/orbit velocity vbar to trace a straight unwrapped orbit.
//   2. Deposit J^{n+1/2} with the same vbar and the same orbit weights used later
//      for particle work.
//   3. Advance rho from the discrete continuity equation instead of redepositing
//      endpoint charge.
//   4. Solve the first-order Crank-Nicolson potential system for phi, psi, A, U.
//   5. Compute the orbit-discrete-gradient matrix D of the same interpolant A_h
//      used in P = p + q A_h(x).
//   6. Split D^T vbar into a chain-rule piece D vbar and a skew magnetic piece
//      K v, where K = D^T - D.
//   7. Advance the skew part with a Higuera-Cary/Cayley magnetic rotation.
//   8. Update the accepted vbar from the relativistic kinetic-energy secant
//      velocity, which is the velocity that makes particle work equal the change
//      in m kappa^2 (gamma - 1).
//
// The identities monitored by the code are
//
//     A_h^{n+1}(x^{n+1}) - A_h^n(x^n)
//       = dt [ U_bar + D vbar ],
//
//     K(p^{n+1}) - K(p^n) = dt q E_bar dot vbar,
//
// and the mesh work identity obtained by using the same orbit weights in gather
// and current deposition.  Together with the CN field-energy balance, these are
// the discrete ingredients of relativistic total-energy conservation at the
// converged nonlinear fixed point.
//
// Orbit splitting at spline knots
// -------------------------------
// Compactly supported B-splines are piecewise polynomials.  Along a particle
// orbit the active polynomial branch changes when a spline knot is crossed.  A
// single Gauss rule over the whole path is not exact across such a crossing, so
// conservative runs keep
//
//     split_orbit_at_knots = true.
//
// The unsplit mode remains available only as a diagnostic failure-mode test.
//
// Included test cases
// -------------------
//
//     cold_relativistic_two_stream : 3D periodic cold two-stream instability;
//                                    the diagnostic |E_x(k_x)| is compared with
//                                    a cold relativistic two-stream growth rate.
//     cold_relativistic_weibel     : 3D cold filamentation/Weibel instability;
//                                    the diagnostic |B_z(k_y)| is compared with
//                                    a cold-fluid transverse growth-rate model.
//     landau_weak / landau_strong  : retained nonrelativistic-style teaching
//                                    examples embedded in the same 3D code path.
//
// Built-in unit tests exercise static relativistic electric and magnetic fields,
// the vector-potential chain rule, split-versus-unsplit crossings, and one-step
// Gauss/energy-coupling diagnostics.
//
// -----------------------------------------------------------------------------

#include <algorithm>
#include <array>
#include <cassert>
#include <cctype>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <map>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

using Real = double;
using Complex = std::complex<Real>;
using Field = std::vector<Real>;
using CField = std::vector<Complex>;
using Vec3 = std::array<Real, 3>;
using Mat3 = std::array<std::array<Real, 3>, 3>;
using VecField = std::array<Field, 3>;

static constexpr Real PI = 3.141592653589793238462643383279502884;

using WallClock = std::chrono::steady_clock;

// ---------------------------------------------------------------------------
// Wall seconds since
//
// Inputs:
//   start : const WallClock::time_point&; start
//
// Output:
//   Real: wall seconds since.
//
// Dependencies:
//   C++17 standard library (headers listed at the top of this file).
// ---------------------------------------------------------------------------
static Real wall_seconds_since(const WallClock::time_point& start) {
    return std::chrono::duration<Real>(WallClock::now() - start).count();
}

// ---------------------------------------------------------------------------
// Format wall duration
//
// Inputs:
//   seconds : Real; seconds
//
// Output:
//   std::string: format wall duration.
//
// Dependencies:
//   C++17 standard library (headers listed at the top of this file).
// ---------------------------------------------------------------------------
static std::string format_wall_duration(Real seconds) {
    const long long total_milliseconds = static_cast<long long>(
        std::llround(std::max<Real>(0.0, seconds) * 1000.0));
    long long remaining = total_milliseconds;
    const long long days = remaining / 86400000LL;
    remaining %= 86400000LL;
    const long long hours = remaining / 3600000LL;
    remaining %= 3600000LL;
    const long long minutes = remaining / 60000LL;
    remaining %= 60000LL;
    const long long whole_seconds = remaining / 1000LL;
    const long long milliseconds = remaining % 1000LL;
    std::ostringstream out;
    if (days > 0) out << days << "d ";
    out << std::setfill('0') << std::setw(2) << hours << ":"
        << std::setw(2) << minutes << ":"
        << std::setw(2) << whole_seconds << "."
        << std::setw(3) << milliseconds;
    return out.str();
}

// SG added: only nonlinear failures are eligible for adaptive step rejection.
// Other runtime errors still terminate immediately instead of being concealed
// by repeatedly reducing the time step.
class NonlinearSolveFailure : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

enum class LocalParticleFailureReason {
    none,
    nonfinite_map,
    superluminal_map,
    nonfinite_iterate,
    superluminal_iterate,
    maximum_iterations
};

static const char* local_particle_failure_reason_name(
        LocalParticleFailureReason reason) {
    switch (reason) {
    case LocalParticleFailureReason::none: return "none";
    case LocalParticleFailureReason::nonfinite_map: return "nonfinite_map";
    case LocalParticleFailureReason::superluminal_map: return "superluminal_map";
    case LocalParticleFailureReason::nonfinite_iterate: return "nonfinite_iterate";
    case LocalParticleFailureReason::superluminal_iterate: return "superluminal_iterate";
    case LocalParticleFailureReason::maximum_iterations: return "maximum_iterations";
    }
    return "unknown";
}

struct FailedParticleDiagnostic {
    std::size_t particle = 0;
    std::size_t map_evaluation = 0;
    int assigned_substeps = 1;
    int failed_substep = 0;
    Real theta0 = 0.0;
    Real theta1 = 1.0;
    Real h = 0.0;
    int local_iterations = 0;
    LocalParticleFailureReason reason = LocalParticleFailureReason::none;
    Real first_scaled_residual = std::numeric_limits<Real>::infinity();
    Real final_scaled_residual = std::numeric_limits<Real>::infinity();
    Real final_absolute_residual = std::numeric_limits<Real>::infinity();
    Real q = 0.0;
    Real m = 0.0;
    Vec3 x0{0.0, 0.0, 0.0};
    Vec3 P0{0.0, 0.0, 0.0};
    Vec3 initial_guess{0.0, 0.0, 0.0};
    Vec3 last_guess{0.0, 0.0, 0.0};
    Vec3 last_map{0.0, 0.0, 0.0};
};

// A fixed selective layout is part of the outer current map.  If a local
// particle solve fails, report the particle indices to the macro wrapper.  The
// wrapper increases only those levels and restarts outer Anderson, ensuring no
// multisecant history is reused after the map changes.
class SelectiveParticleRefinementNeeded : public NonlinearSolveFailure {
public:
    SelectiveParticleRefinementNeeded(std::string message,
                                      std::vector<FailedParticleDiagnostic> diagnostics)
        : NonlinearSolveFailure(std::move(message)),
          failed_diagnostics(std::move(diagnostics)) {}

    std::vector<FailedParticleDiagnostic> failed_diagnostics;
};

// -----------------------------------------------------------------------------
// Small helpers
// -----------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Trim
//
// Inputs:
//   s : const std::string&; s
//
// Output:
//   std::string: trim.
//
// Dependencies:
//   C++17 standard library (headers listed at the top of this file).
// ---------------------------------------------------------------------------
static std::string trim(const std::string& s) {
    const std::string ws = " \t\r\n";
    const auto b = s.find_first_not_of(ws);
    if (b == std::string::npos) return "";
    const auto e = s.find_last_not_of(ws);
    return s.substr(b, e - b + 1);
}

// ---------------------------------------------------------------------------
// Lowercase
//
// Inputs:
//   s : std::string; s
//
// Output:
//   std::string: lowercase.
//
// Dependencies:
//   C++17 standard library (headers listed at the top of this file).
// ---------------------------------------------------------------------------
static std::string lowercase(std::string s) {
    for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

// ---------------------------------------------------------------------------
// Is power of two
//
// Inputs:
//   n : int; number of entries/nodes
//
// Output:
//   bool: success/predicate result; non-const reference arguments carry any results.
//
// Dependencies:
//   Local scalar/vector types; no external library.
// ---------------------------------------------------------------------------
static bool is_power_of_two(int n) {
    return n > 0 && (n & (n - 1)) == 0;
}

// ---------------------------------------------------------------------------
// Openmp max threads
//
// Inputs:
//   None.
//
// Output:
//   int: openmp max threads.
//
// Dependencies:
//   Local scalar/vector types; no external library.
// ---------------------------------------------------------------------------
static int openmp_max_threads() {
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

// ---------------------------------------------------------------------------
// Openmp thread num
//
// Inputs:
//   None.
//
// Output:
//   int: openmp thread num.
//
// Dependencies:
//   Local scalar/vector types; no external library.
// ---------------------------------------------------------------------------
static int openmp_thread_num() {
#ifdef _OPENMP
    return omp_get_thread_num();
#else
    return 0;
#endif
}

// ---------------------------------------------------------------------------
// Deposition thread count
//
// Inputs:
//   mesh_size : std::size_t; mesh size
//   particle_count : std::size_t; particle count
//
// Output:
//   int: deposition thread count.
//
// Dependencies:
//   openmp_max_threads.
//   C++17 standard library (headers listed at the top of this file).
// ---------------------------------------------------------------------------
static int deposition_thread_count(std::size_t mesh_size, std::size_t particle_count) {
    const int max_threads = openmp_max_threads();
    if (max_threads <= 1 || particle_count < 256) return 1;

    // Bound private-mesh storage so a large grid cannot accidentally allocate
    // one three-component mesh for every hardware thread.  The cap covers the
    // private buffers only; the shared result mesh is separate.
    constexpr std::size_t private_mesh_budget = 512ULL * 1024ULL * 1024ULL;
    const std::size_t bytes_per_thread = 3ULL * mesh_size * sizeof(Real);
    const std::size_t memory_threads = bytes_per_thread
        ? std::max<std::size_t>(1, private_mesh_budget / bytes_per_thread)
        : static_cast<std::size_t>(max_threads);
    const std::size_t useful_threads = (particle_count + 255) / 256;
    return static_cast<int>(std::min<std::size_t>(
        static_cast<std::size_t>(max_threads), std::min(memory_threads, useful_threads)));
}

// ---------------------------------------------------------------------------
// Sqr
//
// NOT NEEDED: uncalled legacy/reference helper; retained for provenance.
//
// Inputs:
//   x : Real; x
//
// Output:
//   Real: sqr.
//
// Dependencies:
//   Local scalar/vector types; no external library.
// ---------------------------------------------------------------------------
[[maybe_unused]] static Real sqr(Real x) { return x * x; }

// ---------------------------------------------------------------------------
// Make vec3
//
// NOT NEEDED: uncalled legacy/reference helper; retained for provenance.
//
// Inputs:
//   x : Real; x
//   y : Real; y
//   z : Real; z
//
// Output:
//   Vec3: make vec3.
//
// Dependencies:
//   Local scalar/vector types; no external library.
// ---------------------------------------------------------------------------
[[maybe_unused]] static Vec3 make_vec3(Real x, Real y, Real z) { return Vec3{x, y, z}; }

// ---------------------------------------------------------------------------
// Add3
//
// Inputs:
//   a : const Vec3&; a
//   b : const Vec3&; b
//
// Output:
//   Vec3: add3.
//
// Dependencies:
//   Local scalar/vector types; no external library.
// ---------------------------------------------------------------------------
static Vec3 add3(const Vec3& a, const Vec3& b) {
    return Vec3{a[0] + b[0], a[1] + b[1], a[2] + b[2]};
}

// ---------------------------------------------------------------------------
// Sub3
//
// Inputs:
//   a : const Vec3&; a
//   b : const Vec3&; b
//
// Output:
//   Vec3: sub3.
//
// Dependencies:
//   Local scalar/vector types; no external library.
// ---------------------------------------------------------------------------
static Vec3 sub3(const Vec3& a, const Vec3& b) {
    return Vec3{a[0] - b[0], a[1] - b[1], a[2] - b[2]};
}

// ---------------------------------------------------------------------------
// Mul3
//
// Inputs:
//   c : Real; light speed
//   a : const Vec3&; a
//
// Output:
//   Vec3: mul3.
//
// Dependencies:
//   Local scalar/vector types; no external library.
// ---------------------------------------------------------------------------
static Vec3 mul3(Real c, const Vec3& a) {
    return Vec3{c * a[0], c * a[1], c * a[2]};
}

// ---------------------------------------------------------------------------
// Dot3
//
// Inputs:
//   a : const Vec3&; a
//   b : const Vec3&; b
//
// Output:
//   Real: dot3.
//
// Dependencies:
//   Local scalar/vector types; no external library.
// ---------------------------------------------------------------------------
static Real dot3(const Vec3& a, const Vec3& b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

// ---------------------------------------------------------------------------
// Norm3
//
// Inputs:
//   a : const Vec3&; a
//
// Output:
//   Real: norm3.
//
// Dependencies:
//   dot3.
//   C++17 standard library (headers listed at the top of this file).
// ---------------------------------------------------------------------------
static Real norm3(const Vec3& a) {
    return std::sqrt(dot3(a, a));
}


// ---------------------------------------------------------------------------
// Cross3
//
// Inputs:
//   a : const Vec3&; a
//   b : const Vec3&; b
//
// Output:
//   Vec3: cross3.
//
// Dependencies:
//   Local scalar/vector types; no external library.
// ---------------------------------------------------------------------------
static Vec3 cross3(const Vec3& a, const Vec3& b) {
    return Vec3{a[1] * b[2] - a[2] * b[1],
                a[2] * b[0] - a[0] * b[2],
                a[0] * b[1] - a[1] * b[0]};
}

// ---------------------------------------------------------------------------
// Matvec3
//
// Inputs:
//   M : const Mat3&; M
//   v : const Vec3&; v
//
// Output:
//   Vec3: matvec3.
//
// Dependencies:
//   Local scalar/vector types; no external library.
// ---------------------------------------------------------------------------
static Vec3 matvec3(const Mat3& M, const Vec3& v) {
    return Vec3{M[0][0] * v[0] + M[0][1] * v[1] + M[0][2] * v[2],
                M[1][0] * v[0] + M[1][1] * v[1] + M[1][2] * v[2],
                M[2][0] * v[0] + M[2][1] * v[1] + M[2][2] * v[2]};
}

// ---------------------------------------------------------------------------
// Transpose matvec3
//
// NOT NEEDED: uncalled legacy/reference helper; retained for provenance.
//
// Inputs:
//   M : const Mat3&; M
//   v : const Vec3&; v
//
// Output:
//   Vec3: transpose matvec3.
//
// Dependencies:
//   Local scalar/vector types; no external library.
// ---------------------------------------------------------------------------
[[maybe_unused]] static Vec3 transpose_matvec3(const Mat3& M, const Vec3& v) {
    return Vec3{M[0][0] * v[0] + M[1][0] * v[1] + M[2][0] * v[2],
                M[0][1] * v[0] + M[1][1] * v[1] + M[2][1] * v[2],
                M[0][2] * v[0] + M[1][2] * v[1] + M[2][2] * v[2]};
}

// ---------------------------------------------------------------------------
// Gamma from p
//
// Inputs:
//   p : const Vec3&; p
//   m : Real; particle mass
//   kappa : Real; nondimensional light speed
//
// Output:
//   Real: gamma from p.
//
// Dependencies:
//   dot3.
//   C++17 standard library (headers listed at the top of this file).
// ---------------------------------------------------------------------------
static Real gamma_from_p(const Vec3& p, Real m, Real kappa) {
    const Real mk = m * kappa;
    if (!(mk > 0.0)) throw std::runtime_error("gamma_from_p requires positive m and kappa");
    return std::sqrt(1.0 + dot3(p, p) / (mk * mk));
}

// ---------------------------------------------------------------------------
// Velocity from p
//
// Inputs:
//   p : const Vec3&; p
//   m : Real; particle mass
//   kappa : Real; nondimensional light speed
//
// Output:
//   Vec3: velocity from p.
//
// Dependencies:
//   gamma_from_p, mul3.
// ---------------------------------------------------------------------------
static Vec3 velocity_from_p(const Vec3& p, Real m, Real kappa) {
    const Real gamma = gamma_from_p(p, m, kappa);
    return mul3(1.0 / (m * gamma), p);
}

// ---------------------------------------------------------------------------
// Gamma from v
//
// Inputs:
//   v : const Vec3&; v
//   kappa : Real; nondimensional light speed
//
// Output:
//   Real: gamma from v.
//
// Dependencies:
//   dot3.
//   C++17 standard library (headers listed at the top of this file).
// ---------------------------------------------------------------------------
static Real gamma_from_v(const Vec3& v, Real kappa) {
    const Real beta2 = dot3(v, v) / (kappa * kappa);
    if (beta2 >= 1.0) {
        throw std::runtime_error("relativistic initial velocity must satisfy |v| < kappa");
    }
    return 1.0 / std::sqrt(std::max<Real>(1.0e-300, 1.0 - beta2));
}

// ---------------------------------------------------------------------------
// Momentum from v
//
// Inputs:
//   v : const Vec3&; v
//   m : Real; particle mass
//   kappa : Real; nondimensional light speed
//
// Output:
//   Vec3: momentum from v.
//
// Dependencies:
//   gamma_from_v, mul3.
// ---------------------------------------------------------------------------
static Vec3 momentum_from_v(const Vec3& v, Real m, Real kappa) {
    return mul3(m * gamma_from_v(v, kappa), v);
}

// ---------------------------------------------------------------------------
// Kinetic energy from p
//
// Inputs:
//   p : const Vec3&; p
//   m : Real; particle mass
//   kappa : Real; nondimensional light speed
//
// Output:
//   Real: kinetic energy from p.
//
// Dependencies:
//   gamma_from_p.
// ---------------------------------------------------------------------------
static Real kinetic_energy_from_p(const Vec3& p, Real m, Real kappa) {
    return m * kappa * kappa * (gamma_from_p(p, m, kappa) - 1.0);
}

// ---------------------------------------------------------------------------
// Compute the relativistic kinetic-energy discrete-gradient velocity
//
// Inputs:
//   a : const Vec3&; a
//   b : const Vec3&; b
//   m : Real; particle mass
//   kappa : Real; nondimensional light speed
//
// Output:
//   Vec3: compute the relativistic kinetic-energy discrete-gradient velocity.
//
// Dependencies:
//   add3, gamma_from_p, mul3.
// ---------------------------------------------------------------------------
static Vec3 kinetic_secant_velocity(const Vec3& a, const Vec3& b, Real m, Real kappa) {
    // Relativistic kinetic-energy discrete-gradient velocity:
    //   K(a)-K(b) = V_dg(a,b) dot (a-b),
    // where K(p)=m kappa^2 (gamma(p)-1).  When a=b this reduces to p/(m gamma).
    const Real ga = gamma_from_p(a, m, kappa);
    const Real gb = gamma_from_p(b, m, kappa);
    return mul3(1.0 / (m * (ga + gb)), add3(a, b));
}

// ---------------------------------------------------------------------------
// Wrap1
//
// Inputs:
//   x : Real; x
//   L : Real; periodic domain length(s)
//
// Output:
//   Real: wrap1.
//
// Dependencies:
//   C++17 standard library (headers listed at the top of this file).
// ---------------------------------------------------------------------------
static Real wrap1(Real x, Real L) {
    Real y = std::fmod(x + 0.5 * L, L);
    if (y < 0.0) y += L;
    return y - 0.5 * L;
}

// ---------------------------------------------------------------------------
// Wrap pos
//
// Inputs:
//   x : const Vec3&; x
//   L : const Vec3&; periodic domain length(s)
//
// Output:
//   Vec3: wrap pos.
//
// Dependencies:
//   wrap1.
// ---------------------------------------------------------------------------
static Vec3 wrap_pos(const Vec3& x, const Vec3& L) {
    return Vec3{wrap1(x[0], L[0]), wrap1(x[1], L[1]), wrap1(x[2], L[2])};
}

// ---------------------------------------------------------------------------
// Rms field
//
// Inputs:
//   f : const Field&; f
//
// Output:
//   Real: rms field.
//
// Dependencies:
//   C++17 standard library (headers listed at the top of this file).
// ---------------------------------------------------------------------------
static Real rms_field(const Field& f) {
    long double s = 0.0L;
    for (Real x : f) s += static_cast<long double>(x) * static_cast<long double>(x);
    return std::sqrt(static_cast<Real>(s / std::max<std::size_t>(1, f.size())));
}

// ---------------------------------------------------------------------------
// Rms vec particles
//
// Inputs:
//   a : const std::vector<Vec3>&; a
//
// Output:
//   Real: rms vec particles.
//
// Dependencies:
//   C++17 standard library (headers listed at the top of this file).
// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
// Mean field
//
// Inputs:
//   f : const Field&; f
//
// Output:
//   Real: mean field.
//
// Dependencies:
//   C++17 standard library (headers listed at the top of this file).
// ---------------------------------------------------------------------------
static Real mean_field(const Field& f) {
    long double s = 0.0L;
    for (Real x : f) s += x;
    return static_cast<Real>(s / std::max<std::size_t>(1, f.size()));
}

// ---------------------------------------------------------------------------
// Subtract mean
//
// Inputs:
//   f : Field&; f
//
// Output:
//   None; updates the non-const reference arguments in place.
//
// Dependencies:
//   mean_field.
// ---------------------------------------------------------------------------
static void subtract_mean(Field& f) {
    const Real m = mean_field(f);
    for (Real& x : f) x -= m;
}

// ---------------------------------------------------------------------------
// Zero field
//
// NOT NEEDED: uncalled legacy/reference helper; retained for provenance.
//
// Inputs:
//   n : std::size_t; number of entries/nodes
//
// Output:
//   Field: zero field.
//
// Dependencies:
//   Local scalar/vector types; no external library.
// ---------------------------------------------------------------------------
[[maybe_unused]] static Field zero_field(std::size_t n) { return Field(n, 0.0); }
// ---------------------------------------------------------------------------
// Zero vecfield
//
// Inputs:
//   n : std::size_t; number of entries/nodes
//
// Output:
//   VecField: zero vecfield.
//
// Dependencies:
//   Local scalar/vector types; no external library.
// ---------------------------------------------------------------------------
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
    // SG added: n_steps*dt remains the target physical time. A rejected
    // nonlinear step is retried from the unchanged state with a smaller dt,
    // and the reduced dt persists for subsequent accepted steps.
    bool adaptive_dt = false;
    Real adaptive_dt_min = 0.0; // <=0 selects dt/16
    Real adaptive_dt_shrink = 0.5;
    Real adaptive_dt_growth = 2.0;
    int adaptive_dt_growth_interval = 4;
    int adaptive_max_attempts = 4;

    // ------------------------------------------------------------------
    // Initial condition selector.
    // ------------------------------------------------------------------
    // Supported values:
    //   cold_relativistic_two_stream - 3D cold two-stream instability.
    //   cold_relativistic_weibel     - 3D cold transverse Weibel/filamentation.
    //   landau_weak / landau_strong  - retained teaching examples.
    // The relativistic examples are the main targets of this version.
    std::string test_case = "cold_relativistic_two_stream";

    // For the relativistic two-stream example this is the number of particle
    // pairs per mesh cell.  For the Weibel example it controls the number of
    // x-direction particles per cell for each of the two beams.  For Landau
    // examples it is the number of quiet-start velocity samples per cell.
    int particles_per_cell_pair = 4;

    // Particle shape degree r (not support width): manuscript production uses
    // quadratic r=2; the auxiliary comparison uses r=1 and r=2.
    // Degrees 3 and 4 are optional experiments, not manuscript runs.
    int spline_order = 2;

    // Number of Gauss-Legendre points used on each split orbit segment.  For a
    // B-spline degree r, the paper notes that 2*nq-1 >= 3*r is sufficient for
    // exact integration on each segment.  nq=16 is intentionally generous.
    int orbit_quad_order = 16;
    bool split_orbit_at_knots = true;
    // Manuscript Nyquist-plane projector; false is the matched off comparison.
    bool nyquist_projection = false;

    // Nonlinear orbit-velocity solve. Anderson uses
    // a damped type-II multisecant/quasi-Newton update and falls back to damped
    // Picard whenever a safeguard is triggered. Adaptive rejection changes dt
    // only after the complete nonlinear attempt has failed.
    std::string nonlinear_solver = "anderson_quasi_newton";
    // SG added: mixed relative/absolute tolerances avoid an unrealistically
    // small convergence threshold for particles whose velocity is near zero.
    Real nonlinear_rtol = 1.0e-12;
    Real nonlinear_atol = 1.0e-13;
    int nonlinear_max_iter = 32;
    int anderson_memory = 4;
    int anderson_start = 2;
    Real anderson_damping = 0.7;
    Real anderson_regularization = 1.0e-12;
    Real anderson_growth_limit = 1.2;

    // NOT NEEDED FOR MANUSCRIPT: uniform subcycling is disabled by the driver.
    // Uniform particle subcycling.  Anderson then acts on the macro-averaged
    // grid current (and therefore on the macro field solve), while each
    // particle independently advances through uniform_substeps local solves.
    // If that local pass fails, all particles are retried at a finer uniform
    // resolution, up to uniform_max_substeps.  The macro dt is unchanged.
    // With uniform_refine_on_failure, the accepted resolution persists for
    // all later macro steps: initial_substeps -> substeps -> max_substeps.
    bool uniform_particle_subcycling = false;
    bool uniform_refine_on_failure = false;
    int uniform_initial_substeps = 1;
    int uniform_substeps = 10;
    int uniform_max_substeps = 20;
    Real uniform_subcycling_start_time = 0.0; // <=0 starts uniform_substeps immediately
    Real uniform_local_rtol = 1.0e-11;
    Real uniform_local_atol = 1.0e-13;
    int uniform_local_max_iter = 24;
    Real uniform_local_damping = 1.0;

    // AUXILIARY SPLINE COMPARISON: direct-first selective particle fallback.  The ordinary coupled macro
    // solver is attempted first.  Only after that complete solve fails do we
    // switch, from the unchanged state, to outer grid-current Anderson with a
    // per-particle substep layout.  A particle level may increase through
    // 1,2,4,...,selective_particle_max_substeps; the macro dt never changes.
    bool selective_particle_fallback = false;
    int selective_particle_max_substeps = 64;
    int selective_particle_dynamic_chunk = 4;
    bool selective_failure_diagnostics = true;
    int selective_failure_diagnostic_limit = 16;

    // Field-reconstruction diagnostics for comparing particle-shape degrees.
    // The line probe and knot-jump files are compact and enabled by default.
    // Full 3-D grid snapshots are optional because their CSV files can be large.
    bool field_diagnostics = true;
    Real field_diagnostics_interval = 10.0; // physical time; <=0 writes initial/final only
    int field_probe_points_per_cell = 32;
    std::string field_probe_axis = "auto"; // auto: y for Weibel, x otherwise
    bool field_grid_snapshots = false;

    // Full-particle output at the first accepted endpoint at/after each time.
    std::vector<Real> particle_snapshot_times;

    bool run_unit_tests = true;

    // Nondimensional field/source coefficients from Section 2 of the paper.
    // The field equations are
    //
    //   (1/κ^2) φ_tt - Δφ = σ1 ρ,
    //   (1/κ^2) A_tt - ΔA = σ2 J,
    //   (1/κ^2) φ_t + div A = 0.
    //
    // Maxwell-compatible nondimensionalization satisfies σ1 = κ^2 σ2.
    Real kappa = 1.0;               // nondimensional light speed κ=cT/L
    Real sigma1 = 1.0;              // scalar-potential source coefficient σ1
    Real sigma2 = 1.0;              // vector-potential source coefficient σ2, with σ1=κ^2 σ2 for Maxwell-compatible scaling

    // Plasma and perturbation parameters.
    Real n0 = 1.0;                 // fixed ion background density
    Real v0 = 0.3;                 // relativistic beam drift speed |v0| < kappa
    Real perturbation = 0.02;      // velocity seed for the legacy two-stream form,
                                   // density perturbation fallback for Landau,
                                   // magnetic-field seed for Weibel if weibel_B0<0
    Real two_stream_density_alpha = -1.0; // total charge-density eigenmode amplitude; negative means legacy velocity seed
    bool two_stream_eigenmode = true;     // initialize the growing cold-fluid eigenmode when alpha>=0
    bool two_stream_finite_grid_theory = true; // include linear-B-spline mode factor in reported growth rate
    Real weibel_B0 = -1.0;        // B_z seed amplitude for Weibel; negative uses perturbation
    bool weibel_eigenmode = true;       // initialize the growing cold-fluid Weibel eigenmode, including U_x, velocities, and stream weights
    bool weibel_finite_grid_theory = true; // include B-spline scatter/gather factor in Weibel theory and eigenmode
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

    // Optional time-convergence investigation; defaults preserve existing runs.
    bool transverse_diagnostics = false;
    Real transverse_spectrum_interval = 1.0; // physical time, plus initial/final states
    Real tsi_transverse_seed = 0.0; // common delta-v_x = amplitude*cos(2*pi*mode*y/Ly)
    int tsi_transverse_mode = 1; // current perturbation, not a growing eigenmode
    unsigned int seed = 12345;     // NOT NEEDED: accepted input placeholder; no initializer uses it
    std::string output_prefix = "orbit_smoke";
};

// SG added: keep an automatic lower bound proportional to the requested dt.
// ---------------------------------------------------------------------------
// Effective adaptive dt min
//
// NOT NEEDED FOR MANUSCRIPT RUNS: retained optional/legacy branch.
//
// Inputs:
//   cfg : const Config&; simulation settings and solver tolerances
//
// Output:
//   Real: effective adaptive dt min.
//
// Dependencies:
//   Local scalar/vector types; no external library.
// ---------------------------------------------------------------------------
static Real effective_adaptive_dt_min(const Config& cfg) {
    return (cfg.adaptive_dt_min > 0.0) ? cfg.adaptive_dt_min : cfg.dt / 16.0;
}

// ---------------------------------------------------------------------------
// Parse bool
//
// Inputs:
//   v : std::string; v
//
// Output:
//   bool: success/predicate result; non-const reference arguments carry any results.
//
// Dependencies:
//   lowercase, trim.
// ---------------------------------------------------------------------------
static bool parse_bool(std::string v) {
    v = lowercase(trim(v));
    return v == "1" || v == "true" || v == "yes" || v == "on";
}


// ---------------------------------------------------------------------------
// Is two stream case
//
// Inputs:
//   cfg : const Config&; simulation settings and solver tolerances
//
// Output:
//   bool: success/predicate result; non-const reference arguments carry any results.
//
// Dependencies:
//   lowercase.
//   C++17 standard library (headers listed at the top of this file).
// ---------------------------------------------------------------------------
static bool is_two_stream_case(const Config& cfg) {
    const std::string tc = lowercase(cfg.test_case);
    return tc == "two_stream" || tc == "two-stream" || tc == "twostream" ||
           tc == "relativistic_two_stream" || tc == "relativistic-two-stream" ||
           tc == "cold_relativistic_two_stream" || tc == "cold-relativistic-two-stream" ||
           tc == "cold_two_stream" || tc == "cold-two-stream";
}

// ---------------------------------------------------------------------------
// Is weibel case
//
// Inputs:
//   cfg : const Config&; simulation settings and solver tolerances
//
// Output:
//   bool: success/predicate result; non-const reference arguments carry any results.
//
// Dependencies:
//   lowercase.
//   C++17 standard library (headers listed at the top of this file).
// ---------------------------------------------------------------------------
static bool is_weibel_case(const Config& cfg) {
    const std::string tc = lowercase(cfg.test_case);
    return tc == "weibel" || tc == "relativistic_weibel" || tc == "relativistic-weibel" ||
           tc == "cold_relativistic_weibel" || tc == "cold-relativistic-weibel" ||
           tc == "filamentation" || tc == "cold_filamentation" || tc == "cold-filamentation";
}

// ---------------------------------------------------------------------------
// Weibel seed b0
//
// Inputs:
//   cfg : const Config&; simulation settings and solver tolerances
//
// Output:
//   Real: weibel seed b0.
//
// Dependencies:
//   Local scalar/vector types; no external library.
// ---------------------------------------------------------------------------
static Real weibel_seed_B0(const Config& cfg) {
    return (cfg.weibel_B0 >= 0.0) ? cfg.weibel_B0 : cfg.perturbation;
}

// ---------------------------------------------------------------------------
// Is landau case
//
// NOT NEEDED FOR MANUSCRIPT RUNS: retained optional/legacy branch.
//
// Inputs:
//   cfg : const Config&; simulation settings and solver tolerances
//
// Output:
//   bool: success/predicate result; non-const reference arguments carry any results.
//
// Dependencies:
//   lowercase.
//   C++17 standard library (headers listed at the top of this file).
// ---------------------------------------------------------------------------
static bool is_landau_case(const Config& cfg) {
    const std::string tc = lowercase(cfg.test_case);
    return tc == "landau" || tc == "landau_weak" || tc == "landau_strong" ||
           tc == "weak_landau" || tc == "strong_landau";
}

// ---------------------------------------------------------------------------
// Landau density amplitude
//
// NOT NEEDED FOR MANUSCRIPT RUNS: retained optional/legacy branch.
//
// Inputs:
//   cfg : const Config&; simulation settings and solver tolerances
//
// Output:
//   Real: landau density amplitude.
//
// Dependencies:
//   Local scalar/vector types; no external library.
// ---------------------------------------------------------------------------
static Real landau_density_amplitude(const Config& cfg) {
    return (cfg.landau_alpha >= 0.0) ? cfg.landau_alpha : cfg.perturbation;
}


// ---------------------------------------------------------------------------
// Sinc unormalized
//
// Inputs:
//   x : Real; x
//
// Output:
//   Real: sinc unormalized.
//
// Dependencies:
//   C++17 standard library (headers listed at the top of this file).
// ---------------------------------------------------------------------------
static Real sinc_unormalized(Real x) {
    return (std::abs(x) < 1.0e-12) ? 1.0 : std::sin(x) / x;
}

// ---------------------------------------------------------------------------
// Compute the centered B-spline Fourier scatter/gather amplitude
//
// Inputs:
//   degree : int; spline polynomial degree r
//   k : Real; k
//   dx : Real; grid spacing
//
// Output:
//   Real: compute the centered B-spline Fourier scatter/gather amplitude.
//
// Dependencies:
//   sinc_unormalized.
//   C++17 standard library (headers listed at the top of this file).
// ---------------------------------------------------------------------------
static Real bspline_mode_factor_1d(int degree, Real k, Real dx) {
    // Fourier amplitude of a centered cardinal B-spline assignment/gather shape.
    // The PIC linear response contains this factor once in scatter and once in
    // gather, so cold-fluid mode tests should use omega_p^2 |S(k)|^2 when
    // finite-grid theory is requested.  For linear splines this is sinc(k dx/2)^2.
    return std::pow(sinc_unormalized(0.5 * k * dx), static_cast<Real>(degree + 1));
}

// ---------------------------------------------------------------------------
// Read and validate a key=value simulation input file
//
// Inputs:
//   path : const std::string&; input/output filename
//
// Output:
//   Config: read and validate a key=value simulation input file.
//
// Dependencies:
//   effective_adaptive_dt_min, is_two_stream_case, lowercase, parse_bool, trim.
//   C++17 standard library (headers listed at the top of this file).
// ---------------------------------------------------------------------------
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
        else if (key == "nonlinear_solver") cfg.nonlinear_solver = lowercase(val);
        // SG added: keep the old nonlinear_tol and picard_tol names as
        // backward-compatible aliases for the new relative tolerance.
        else if (key == "nonlinear_rtol" || key == "nonlinear_tol" || key == "picard_tol") ss >> cfg.nonlinear_rtol;
        else if (key == "nonlinear_atol") ss >> cfg.nonlinear_atol;
        else if (key == "nyquist_projection") cfg.nyquist_projection = parse_bool(val);
        else if (key == "nonlinear_max_iter" || key == "picard_max_iter") ss >> cfg.nonlinear_max_iter;
        else if (key == "anderson_memory") ss >> cfg.anderson_memory;
        else if (key == "anderson_start") ss >> cfg.anderson_start;
        else if (key == "anderson_damping") ss >> cfg.anderson_damping;
        else if (key == "anderson_regularization") ss >> cfg.anderson_regularization;
        else if (key == "anderson_growth_limit") ss >> cfg.anderson_growth_limit;
        else if (key == "uniform_particle_subcycling" || key == "uniform_subcycling")
            cfg.uniform_particle_subcycling = parse_bool(val);
        else if (key == "uniform_refine_on_failure" ||
                 key == "particle_refine_on_failure")
            cfg.uniform_refine_on_failure = parse_bool(val);
        else if (key == "uniform_initial_substeps" || key == "particle_initial_substeps")
            ss >> cfg.uniform_initial_substeps;
        else if (key == "uniform_substeps" || key == "particle_substeps")
            ss >> cfg.uniform_substeps;
        else if (key == "uniform_max_substeps" || key == "particle_max_substeps")
            ss >> cfg.uniform_max_substeps;
        else if (key == "uniform_subcycling_start_time" || key == "particle_subcycling_start_time")
            ss >> cfg.uniform_subcycling_start_time;
        else if (key == "uniform_local_rtol" || key == "particle_local_rtol")
            ss >> cfg.uniform_local_rtol;
        else if (key == "uniform_local_atol" || key == "particle_local_atol")
            ss >> cfg.uniform_local_atol;
        else if (key == "uniform_local_max_iter" || key == "particle_local_max_iter")
            ss >> cfg.uniform_local_max_iter;
        else if (key == "uniform_local_damping" || key == "particle_local_damping")
            ss >> cfg.uniform_local_damping;
        else if (key == "selective_particle_fallback" ||
                 key == "selective_subcycle_fallback")
            cfg.selective_particle_fallback = parse_bool(val);
        else if (key == "selective_particle_max_substeps" ||
                 key == "selective_max_substeps")
            ss >> cfg.selective_particle_max_substeps;
        else if (key == "selective_particle_dynamic_chunk" ||
                 key == "selective_dynamic_chunk")
            ss >> cfg.selective_particle_dynamic_chunk;
        else if (key == "selective_failure_diagnostics" ||
                 key == "failed_particle_diagnostics")
            cfg.selective_failure_diagnostics = parse_bool(val);
        else if (key == "selective_failure_diagnostic_limit" ||
                 key == "failed_particle_diagnostic_limit")
            ss >> cfg.selective_failure_diagnostic_limit;
        else if (key == "field_diagnostics" || key == "write_field_diagnostics")
            cfg.field_diagnostics = parse_bool(val);
        else if (key == "field_diagnostics_interval" ||
                 key == "field_output_interval")
            ss >> cfg.field_diagnostics_interval;
        else if (key == "field_probe_points_per_cell" ||
                 key == "field_samples_per_cell")
            ss >> cfg.field_probe_points_per_cell;
        else if (key == "field_probe_axis")
            cfg.field_probe_axis = lowercase(val);
        else if (key == "field_grid_snapshots" ||
                 key == "write_full_field_snapshots")
            cfg.field_grid_snapshots = parse_bool(val);
        else if (key == "particle_snapshot_times") {
            std::string times = val;
            std::replace(times.begin(), times.end(), ',', ' ');
            std::istringstream values(times);
            cfg.particle_snapshot_times.clear();
            Real t;
            while (values >> t) {
                if (!std::isfinite(t) || t < 0.0)
                    throw std::runtime_error("particle_snapshot_times must be finite and nonnegative");
                cfg.particle_snapshot_times.push_back(t);
            }
            if (!values.eof())
                throw std::runtime_error("invalid particle_snapshot_times list");
            std::sort(cfg.particle_snapshot_times.begin(), cfg.particle_snapshot_times.end());
            cfg.particle_snapshot_times.erase(
                std::unique(cfg.particle_snapshot_times.begin(), cfg.particle_snapshot_times.end()),
                cfg.particle_snapshot_times.end());
        }
        // SG added: nonlinear-failure step rejection controls.
        else if (key == "adaptive_dt" || key == "adaptive_time_step" ||
                 key == "adaptive_timestep" || key == "adaptive_reject_on_unconverged")
            cfg.adaptive_dt = parse_bool(val);
        else if (key == "adaptive_dt_min" || key == "dt_min") ss >> cfg.adaptive_dt_min;
        else if (key == "adaptive_dt_shrink" || key == "dt_shrink") ss >> cfg.adaptive_dt_shrink;
        else if (key == "adaptive_dt_growth" || key == "dt_growth") ss >> cfg.adaptive_dt_growth;
        else if (key == "adaptive_dt_growth_interval" || key == "dt_growth_interval")
            ss >> cfg.adaptive_dt_growth_interval;
        else if (key == "adaptive_max_attempts" || key == "max_step_attempts")
            ss >> cfg.adaptive_max_attempts;
        else if (key == "picard_relaxation" || key == "picard_omega" || key == "relaxed_picard_omega" ||
                 key == "picard_relaxation_min" || key == "picard_omega_min" ||
                 key == "picard_relaxation_shrink" || key == "picard_relaxation_growth" ||
                 key == "picard_adaptive_relaxation" || key == "adaptive_picard_relaxation" ||
                 key == "picard_current_in_convergence" || key == "picard_converge_current" ||
                 key == "final_consistency_sweep" || key == "picard_require_converged" ||
                 key == "reject_unconverged_picard") {
            std::cerr << "warning: obsolete Picard control ignored: " << key << "\n";
        }
        else if (key == "adaptive_dt_max" || key == "dt_max" ||
                 key == "adaptive_max_accepted_steps" || key == "max_accepted_steps" ||
                 key == "adaptive_energy_rtol" || key == "energy_accept_rtol" || key == "adaptive_finish_exactly") {
            std::cerr << "warning: unsupported adaptive time-step control ignored: " << key << "\n";
        }
        else if (key == "run_unit_tests") cfg.run_unit_tests = parse_bool(val);
        else if (key == "kappa") ss >> cfg.kappa;
        else if (key == "c") { ss >> cfg.kappa; std::cerr << "warning: c is deprecated; use kappa in the nondimensional paper notation\n"; }
        else if (key == "sigma1") ss >> cfg.sigma1;
        else if (key == "sigma2") ss >> cfg.sigma2;
        else if (key == "eps0") { Real eps0_deprecated = 1.0; ss >> eps0_deprecated; cfg.sigma1 = 1.0 / eps0_deprecated; std::cerr << "warning: eps0 is deprecated; use sigma1=1/eps0 in the nondimensional paper notation\n"; }
        else if (key == "mu0") { ss >> cfg.sigma2; std::cerr << "warning: mu0 is deprecated; use sigma2 in the nondimensional paper notation\n"; }
        else if (key == "n0" || key == "rho0" || key == "background_density") ss >> cfg.n0;
        else if (key == "v0") ss >> cfg.v0;
        else if (key == "perturbation") ss >> cfg.perturbation;
        else if (key == "two_stream_density_alpha" || key == "density_perturbation" || key == "alpha_x") ss >> cfg.two_stream_density_alpha;
        else if (key == "two_stream_eigenmode" || key == "initialize_two_stream_eigenmode") cfg.two_stream_eigenmode = parse_bool(val);
        else if (key == "two_stream_finite_grid_theory") cfg.two_stream_finite_grid_theory = parse_bool(val);
        else if (key == "weibel_finite_grid_theory") cfg.weibel_finite_grid_theory = parse_bool(val);
        else if (key == "finite_grid_theory") { cfg.two_stream_finite_grid_theory = parse_bool(val); cfg.weibel_finite_grid_theory = parse_bool(val); }
        else if (key == "weibel_B0" || key == "magnetic_perturbation" || key == "initial_B0") ss >> cfg.weibel_B0;
        else if (key == "weibel_eigenmode" || key == "initialize_weibel_eigenmode") cfg.weibel_eigenmode = parse_bool(val);
        else if (key == "landau_alpha" || key == "landau_density_perturbation") ss >> cfg.landau_alpha;
        else if (key == "thermal_velocity" || key == "vth") ss >> cfg.thermal_velocity;
        else if (key == "perturbation_mode") ss >> cfg.perturbation_mode;
        else if (key == "theory_omega" || key == "landau_theory_omega") ss >> cfg.theory_omega;
        else if (key == "theory_gamma" || key == "landau_theory_gamma") ss >> cfg.theory_gamma;
        else if (key == "transverse_diagnostics") cfg.transverse_diagnostics = parse_bool(val);
        else if (key == "transverse_spectrum_interval") ss >> cfg.transverse_spectrum_interval;
        else if (key == "tsi_transverse_seed") ss >> cfg.tsi_transverse_seed;
        else if (key == "tsi_transverse_mode") ss >> cfg.tsi_transverse_mode;
        else if (key == "seed") ss >> cfg.seed;
        else if (key == "output_prefix") cfg.output_prefix = val;
        else std::cerr << "warning: unknown input key ignored: " << key << "\n";
    }
    if (!std::isfinite(cfg.transverse_spectrum_interval) ||
        !(cfg.transverse_spectrum_interval > 0.0) ||
        !std::isfinite(cfg.tsi_transverse_seed) || cfg.tsi_transverse_mode < 1) {
        throw std::runtime_error("invalid transverse diagnostic interval or seed controls");
    }
    if (cfg.tsi_transverse_seed != 0.0 &&
        (!is_two_stream_case(cfg) || cfg.ny < 3 ||
         cfg.tsi_transverse_mode >= cfg.ny / 2)) {
        throw std::runtime_error("TSI transverse seed requires a two-stream case and 0<mode<ny/2");
    }
    if (!(cfg.kappa > 0.0) || !(cfg.sigma1 > 0.0) || !(cfg.sigma2 > 0.0)) {
        throw std::runtime_error("nondimensional coefficients require kappa>0, sigma1>0, sigma2>0");
    }
    if (!(cfg.dt > 0.0) || cfg.n_steps < 0) {
        throw std::runtime_error("time stepping requires dt>0 and n_steps>=0");
    }
    // SG added: validate rejection controls even when adaptive_dt=false so a
    // later input toggle cannot activate invalid parameters.
    const Real adaptive_dt_min = effective_adaptive_dt_min(cfg);
    if (!(adaptive_dt_min > 0.0) || adaptive_dt_min > cfg.dt ||
        !(cfg.adaptive_dt_shrink > 0.0) || !(cfg.adaptive_dt_shrink < 1.0) ||
        !(cfg.adaptive_dt_growth > 1.0) || cfg.adaptive_dt_growth_interval < 1 ||
        cfg.adaptive_max_attempts < 1) {
        throw std::runtime_error(
            "adaptive controls require 0<dt_min<=dt, 0<dt_shrink<1, "
            "dt_growth>1, dt_growth_interval>=1, and adaptive_max_attempts>=1");
    }
    if (cfg.nonlinear_solver != "picard" && cfg.nonlinear_solver != "anderson_quasi_newton") {
        throw std::runtime_error("nonlinear_solver must be picard or anderson_quasi_newton");
    }
    // SG added: both tolerances must be positive because their sum is used as
    // the denominator of the mixed residual norm.
    if (!(cfg.nonlinear_rtol > 0.0) || !(cfg.nonlinear_atol > 0.0) || cfg.nonlinear_max_iter <= 0) {
        throw std::runtime_error("nonlinear controls require nonlinear_rtol>0, "
        "nonlinear_atol>0, and nonlinear_max_iter>0");
    }
    if (cfg.anderson_memory < 1 || cfg.anderson_start < 0) {
        throw std::runtime_error("Anderson controls require anderson_memory>=1 and anderson_start>=0");
    }
    if (!(cfg.anderson_damping > 0.0) || !(cfg.anderson_damping <= 1.0)) {
        throw std::runtime_error("anderson_damping must satisfy 0 < damping <= 1");
    }
    if (!(cfg.anderson_regularization >= 0.0) || !(cfg.anderson_growth_limit > 1.0)) {
        throw std::runtime_error("Anderson controls require regularization>=0 and growth_limit>1");
    }
    if (cfg.uniform_initial_substeps < 1 ||
        cfg.uniform_initial_substeps > cfg.uniform_substeps ||
        cfg.uniform_substeps < 1 || cfg.uniform_max_substeps < cfg.uniform_substeps ||
        !(cfg.uniform_subcycling_start_time >= 0.0) ||
        !(cfg.uniform_local_rtol > 0.0) ||
        !(cfg.uniform_local_atol > 0.0) || cfg.uniform_local_max_iter < 1 ||
        !(cfg.uniform_local_damping > 0.0) || !(cfg.uniform_local_damping <= 1.0)) {
        throw std::runtime_error(
            "uniform particle subcycling requires 1<=initial_substeps<=substeps<=max_substeps, "
            "start_time>=0, positive local "
            "rtol/atol/max_iter, and 0<local_damping<=1");
    }
    if (cfg.selective_particle_max_substeps < 1 ||
        cfg.selective_particle_max_substeps > 64 ||
        (cfg.selective_particle_max_substeps &
         (cfg.selective_particle_max_substeps - 1)) != 0 ||
        cfg.selective_particle_dynamic_chunk < 1 ||
        cfg.selective_failure_diagnostic_limit < 1) {
        throw std::runtime_error(
            "selective particle fallback requires a power-of-two "
            "max_substeps in [1,64], dynamic_chunk>=1, and "
            "failure_diagnostic_limit>=1");
    }
    if (cfg.selective_particle_fallback && cfg.uniform_particle_subcycling) {
        throw std::runtime_error(
            "selective_particle_fallback and uniform_particle_subcycling "
            "cannot both be enabled");
    }
    if (cfg.selective_particle_fallback && cfg.adaptive_dt) {
        throw std::runtime_error(
            "selective_particle_fallback requires adaptive_dt=false so the "
            "macro dt remains fixed when the dt/max_substeps particle floor "
            "is exhausted");
    }
    if (!(cfg.field_diagnostics_interval >= 0.0) ||
        cfg.field_probe_points_per_cell < 2 ||
        (cfg.field_probe_axis != "auto" && cfg.field_probe_axis != "x" &&
         cfg.field_probe_axis != "y" && cfg.field_probe_axis != "z")) {
        throw std::runtime_error(
            "field diagnostics require interval>=0, points_per_cell>=2, "
            "and field_probe_axis=auto, x, y, or z");
    }
    const Real compat = cfg.kappa * cfg.kappa * cfg.sigma2;
    const Real rel = std::abs(cfg.sigma1 - compat) / std::max<Real>(1.0, std::abs(cfg.sigma1));
    if (rel > 1.0e-10) {
        std::ostringstream msg;
        msg << "Maxwell-compatible nondimensionalization requires sigma1 = kappa^2*sigma2; got sigma1="
            << cfg.sigma1 << ", kappa^2*sigma2=" << compat;
        throw std::runtime_error(msg.str());
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
    bool nyquist_projection = false;

    int index(int i, int j, int k) const {
        return (i * ny + j) * nz + k;
    }
};

// ---------------------------------------------------------------------------
// Is even nyquist index
//
// Inputs:
//   i : int; i
//   n : int; number of entries/nodes
//
// Output:
//   bool: success/predicate result; non-const reference arguments carry any results.
//
// Dependencies:
//   Local scalar/vector types; no external library.
// ---------------------------------------------------------------------------
static bool is_even_nyquist_index(int i, int n) {
    return (n > 1 && (n % 2 == 0) && i == n / 2);
}

// ---------------------------------------------------------------------------
// Fft mode number
//
// Inputs:
//   i : int; i
//   n : int; number of entries/nodes
//
// Output:
//   int: fft mode number.
//
// Dependencies:
//   is_even_nyquist_index.
// ---------------------------------------------------------------------------
static int fft_mode_number(int i, int n) {
    // Map an FFT array index to its integer Fourier mode for the real-valued
    // pseudo-spectral derivative used by this code.  On an even grid the
    // Nyquist coefficient is self-conjugate.  Multiplication by i*k would make
    // that self-conjugate coefficient purely imaginary, and the subsequent
    // real inverse transform would drop it.  Therefore the represented first
    // derivative has zero symbol at the even-grid Nyquist index.
    //
    // The Laplacian used by Poisson and by the CN wave solve is built from the
    // same symbols below.  This makes
    //
    //      Delta_h = div_h grad_h
    //
    // on the represented real Fourier grid, including modes that live on a
    // Nyquist plane.  Using a nonzero Nyquist wavenumber in k2 while grad/div
    // effectively use zero is enough to inject long-time Gauss-law error.
    if (n == 1) return 0;
    if (is_even_nyquist_index(i, n)) return 0;
    return (i <= n / 2) ? i : i - n;
}

// ---------------------------------------------------------------------------
// Make grid
//
// Inputs:
//   cfg : const Config&; simulation settings and solver tolerances
//
// Output:
//   Grid: make grid.
//
// Dependencies:
//   fft_mode_number, is_power_of_two.
//   C++17 standard library (headers listed at the top of this file).
//   Optional OpenMP; serial loop fallback when OpenMP is disabled.
// ---------------------------------------------------------------------------
static Grid make_grid(const Config& cfg) {
    if (!is_power_of_two(cfg.nx) || !is_power_of_two(cfg.ny) || !is_power_of_two(cfg.nz)) {
        throw std::runtime_error("the in-file radix-2 FFT requires nx, ny, nz to be powers of two");
    }
    Grid g;
    g.nx = cfg.nx; g.ny = cfg.ny; g.nz = cfg.nz;
    g.nyquist_projection = cfg.nyquist_projection;
    g.N = g.nx * g.ny * g.nz;
    g.L = Vec3{cfg.Lx, cfg.Ly, cfg.Lz};
    g.dx = Vec3{cfg.Lx / cfg.nx, cfg.Ly / cfg.ny, cfg.Lz / cfg.nz};
    g.dV = g.dx[0] * g.dx[1] * g.dx[2];
    g.kx.assign(g.N, 0.0); g.ky.assign(g.N, 0.0); g.kz.assign(g.N, 0.0); g.k2.assign(g.N, 0.0);
    #pragma omp parallel for schedule(static)
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

// ---------------------------------------------------------------------------
// Apply the in-place radix-two complex FFT or its normalized inverse
//
// Inputs:
//   a : std::vector<Complex>&; a
//   inverse : bool; true for normalized inverse transform
//
// Output:
//   None; updates the non-const reference arguments in place.
//
// Dependencies:
//   is_power_of_two.
//   C++17 standard library (headers listed at the top of this file).
// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
// Apply the tensor-product three-dimensional FFT in place
//
// Inputs:
//   data : CField&; data
//   g : const Grid&; periodic grid geometry and Fourier symbols
//   inverse : bool; true for normalized inverse transform
//
// Output:
//   None; updates the non-const reference arguments in place.
//
// Dependencies:
//   fft1d.
//   C++17 standard library (headers listed at the top of this file).
//   Optional OpenMP; serial loop fallback when OpenMP is disabled.
// ---------------------------------------------------------------------------
static void fft3d(CField& data, const Grid& g, bool inverse) {
    // Keep one scratch line per worker instead of allocating a vector for every
    // transformed line.  On particle-sized 3-D grids those allocator calls can
    // otherwise cost as much as the butterflies themselves.
    #pragma omp parallel
    {
        std::vector<Complex> tmp(g.nz);
        #pragma omp for collapse(2) schedule(static)
        for (int i = 0; i < g.nx; ++i) {
            for (int j = 0; j < g.ny; ++j) {
                for (int k = 0; k < g.nz; ++k) tmp[k] = data[g.index(i, j, k)];
                fft1d(tmp, inverse);
                for (int k = 0; k < g.nz; ++k) data[g.index(i, j, k)] = tmp[k];
            }
        }
    }

    #pragma omp parallel
    {
        std::vector<Complex> tmp(g.ny);
        #pragma omp for collapse(2) schedule(static)
        for (int i = 0; i < g.nx; ++i) {
            for (int k = 0; k < g.nz; ++k) {
                for (int j = 0; j < g.ny; ++j) tmp[j] = data[g.index(i, j, k)];
                fft1d(tmp, inverse);
                for (int j = 0; j < g.ny; ++j) data[g.index(i, j, k)] = tmp[j];
            }
        }
    }

    #pragma omp parallel
    {
        std::vector<Complex> tmp(g.nx);
        #pragma omp for collapse(2) schedule(static)
        for (int j = 0; j < g.ny; ++j) {
            for (int k = 0; k < g.nz; ++k) {
                for (int i = 0; i < g.nx; ++i) tmp[i] = data[g.index(i, j, k)];
                fft1d(tmp, inverse);
                for (int i = 0; i < g.nx; ++i) data[g.index(i, j, k)] = tmp[i];
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Fft real
//
// Inputs:
//   f : const Field&; f
//   g : const Grid&; periodic grid geometry and Fourier symbols
//
// Output:
//   CField: fft real.
//
// Dependencies:
//   fft3d.
//   Optional OpenMP; serial loop fallback when OpenMP is disabled.
// ---------------------------------------------------------------------------
static CField fft_real(const Field& f, const Grid& g) {
    CField out(g.N);
    #pragma omp parallel for schedule(static)
    for (int id = 0; id < g.N; ++id) out[id] = Complex(f[id], 0.0);
    fft3d(out, g, false);
    return out;
}

// ---------------------------------------------------------------------------
// Ifft real
//
// Inputs:
//   fhat : CField; fhat
//   g : const Grid&; periodic grid geometry and Fourier symbols
//
// Output:
//   Field: ifft real.
//
// Dependencies:
//   fft3d.
//   Optional OpenMP; serial loop fallback when OpenMP is disabled.
// ---------------------------------------------------------------------------
static Field ifft_real(CField fhat, const Grid& g) {
    fft3d(fhat, g, true);
    Field f(g.N);
    #pragma omp parallel for schedule(static)
    for (int id = 0; id < g.N; ++id) f[id] = fhat[id].real();
    return f;
}

// -----------------------------------------------------------------------------
// Spectral operators
// -----------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Spectral gradient scalar
//
// Inputs:
//   f : const Field&; f
//   g : const Grid&; periodic grid geometry and Fourier symbols
//
// Output:
//   VecField: spectral gradient scalar.
//
// Dependencies:
//   fft_real, ifft_real, zero_vecfield.
//   C++17 standard library (headers listed at the top of this file).
//   Optional OpenMP; serial loop fallback when OpenMP is disabled.
// ---------------------------------------------------------------------------
static VecField spectral_gradient_scalar(const Field& f, const Grid& g) {
    const Complex I(0.0, 1.0);
    const CField fhat = fft_real(f, g);
    VecField out = zero_vecfield(g.N);
    for (int d = 0; d < 3; ++d) {
        CField dhat(g.N);
        #pragma omp parallel for schedule(static)
        for (int id = 0; id < g.N; ++id) {
            const Real kk = (d == 0) ? g.kx[id] : (d == 1) ? g.ky[id] : g.kz[id];
            dhat[id] = I * kk * fhat[id];
        }
        out[d] = ifft_real(std::move(dhat), g);
    }
    return out;
}

// ---------------------------------------------------------------------------
// Spectral divergence vector
//
// Inputs:
//   V : const VecField&; V
//   g : const Grid&; periodic grid geometry and Fourier symbols
//
// Output:
//   Field: spectral divergence vector.
//
// Dependencies:
//   fft_real, ifft_real.
//   C++17 standard library (headers listed at the top of this file).
//   Optional OpenMP; serial loop fallback when OpenMP is disabled.
// ---------------------------------------------------------------------------
static Field spectral_divergence_vector(const VecField& V, const Grid& g) {
    const Complex I(0.0, 1.0);
    const CField hx = fft_real(V[0], g);
    const CField hy = fft_real(V[1], g);
    const CField hz = fft_real(V[2], g);
    CField divhat(g.N);
    #pragma omp parallel for schedule(static)
    for (int id = 0; id < g.N; ++id) {
        divhat[id] = I * (g.kx[id] * hx[id] + g.ky[id] * hy[id] + g.kz[id] * hz[id]);
    }
    return ifft_real(std::move(divhat), g);
}

// ---------------------------------------------------------------------------
// Spectral curl
//
// Inputs:
//   A : const VecField&; nodal vector potential
//   g : const Grid&; periodic grid geometry and Fourier symbols
//
// Output:
//   VecField: spectral curl.
//
// Dependencies:
//   fft_real, ifft_real, zero_vecfield.
//   C++17 standard library (headers listed at the top of this file).
//   Optional OpenMP; serial loop fallback when OpenMP is disabled.
// ---------------------------------------------------------------------------
static VecField spectral_curl(const VecField& A, const Grid& g) {
    const Complex I(0.0, 1.0);
    const CField ax = fft_real(A[0], g);
    const CField ay = fft_real(A[1], g);
    const CField az = fft_real(A[2], g);
    std::array<CField, 3> ch{CField(g.N), CField(g.N), CField(g.N)};
    #pragma omp parallel for schedule(static)
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

// ---------------------------------------------------------------------------
// Cardinal bspline
//
// NOT NEEDED: uncalled legacy/reference helper; retained for provenance.
//
// Inputs:
//   order : int; spline degree r, except cardinal_bspline where it is degree+1
//   t : Real; t
//
// Output:
//   Real: cardinal bspline.
//
// Dependencies:
//   C++17 standard library (headers listed at the top of this file).
// ---------------------------------------------------------------------------
[[maybe_unused]] static Real cardinal_bspline(int order, Real t) {
    if (order < 1) throw std::runtime_error("cardinal_bspline order must be positive");
    if (order == 1) return (t >= 0.0 && t < 1.0) ? 1.0 : 0.0;
    return (t / static_cast<Real>(order - 1)) * cardinal_bspline(order - 1, t)
         + ((static_cast<Real>(order) - t) / static_cast<Real>(order - 1)) * cardinal_bspline(order - 1, t - 1.0);
}

// ---------------------------------------------------------------------------
// Cardinal bspline derivative
//
// NOT NEEDED: uncalled legacy/reference helper; retained for provenance.
//
// Inputs:
//   order : int; spline degree r, except cardinal_bspline where it is degree+1
//   t : Real; t
//
// Output:
//   Real: cardinal bspline derivative.
//
// Dependencies:
//   cardinal_bspline.
// ---------------------------------------------------------------------------
[[maybe_unused]] static Real cardinal_bspline_derivative(int order, Real t) {
    if (order <= 1) return 0.0;
    return cardinal_bspline(order - 1, t) - cardinal_bspline(order - 1, t - 1.0);
}

// ---------------------------------------------------------------------------
// Evaluate a centered cardinal B-spline of degree 1 through 4
//
// Inputs:
//   degree : int; spline polynomial degree r
//   r : Real; r
//
// Output:
//   Real: evaluate a centered cardinal B-spline of degree 1 through 4.
//
// Dependencies:
//   C++17 standard library (headers listed at the top of this file).
// ---------------------------------------------------------------------------
static Real centered_bspline(int degree, Real r) {
    if (degree < 1 || degree > 4) throw std::runtime_error("spline_order must be 1, 2, 3, or 4");
    // Closed polynomial forms avoid the recursive Cox-de Boor evaluation in
    // the innermost particle/quad-point loop.
    if (degree == 1) {
        if (r >= -1.0 && r < 0.0) return 1.0 + r;
        if (r >= 0.0 && r < 1.0) return 1.0 - r;
        return 0.0;
    }
    const Real a = std::abs(r);
    if (degree == 2) {
        if (a < 0.5) return 0.75 - a * a;
        if (a < 1.5) {
            const Real z = 1.5 - a;
            return 0.5 * z * z;
        }
        return 0.0;
    }
    if (degree == 3) {
        if (a < 1.0) return (2.0 / 3.0) - a * a + 0.5 * a * a * a;
        if (a < 2.0) {
            const Real z = 2.0 - a;
            return z * z * z / 6.0;
        }
        return 0.0;
    }

    if (a < 0.5) {
        const Real a2 = a * a;
        return 115.0 / 192.0 - (5.0 / 8.0) * a2 + 0.25 * a2 * a2;
    }
    if (a < 1.5) {
        return 55.0 / 96.0 + (5.0 / 24.0) * a - 1.25 * a * a
             + (5.0 / 6.0) * a * a * a - (1.0 / 6.0) * a * a * a * a;
    }
    if (a < 2.5) {
        const Real z = 2.5 - a;
        const Real z2 = z * z;
        return z2 * z2 / 24.0;
    }
    return 0.0;
}

// ---------------------------------------------------------------------------
// Evaluate the piecewise derivative of a centered B-spline
//
// Inputs:
//   degree : int; spline polynomial degree r
//   r : Real; r
//
// Output:
//   Real: evaluate the piecewise derivative of a centered B-spline.
//
// Dependencies:
//   C++17 standard library (headers listed at the top of this file).
// ---------------------------------------------------------------------------
static Real centered_bspline_derivative(int degree, Real r) {
    if (degree < 1 || degree > 4) throw std::runtime_error("spline_order must be 1, 2, 3, or 4");
    if (degree == 1) {
        if (r >= -1.0 && r < 0.0) return 1.0;
        if (r >= 0.0 && r < 1.0) return -1.0;
        return 0.0;
    }
    const Real a = std::abs(r);
    const Real sign = (r < 0.0) ? -1.0 : 1.0;
    if (degree == 2) {
        if (a < 0.5) return -2.0 * r;
        if (a < 1.5) return -sign * (1.5 - a);
        return 0.0;
    }
    if (degree == 3) {
        if (a < 1.0) return sign * (-2.0 * a + 1.5 * a * a);
        if (a < 2.0) {
            const Real z = 2.0 - a;
            return -0.5 * sign * z * z;
        }
        return 0.0;
    }

    if (a < 0.5) {
        return sign * (-(5.0 / 4.0) * a + a * a * a);
    }
    if (a < 1.5) {
        return sign * (5.0 / 24.0 - 2.5 * a + 2.5 * a * a
                       - (2.0 / 3.0) * a * a * a);
    }
    if (a < 2.5) {
        const Real z = 2.5 - a;
        return -sign * z * z * z / 6.0;
    }
    return 0.0;
}

struct AxisStencil {
    std::array<int, 5> idx{};
    std::array<Real, 5> w{};
    std::array<Real, 5> dw_dx{};
};

// ---------------------------------------------------------------------------
// Build periodic one-dimensional spline weights and physical derivatives
//
// Inputs:
//   x : Real; x
//   n : int; number of entries/nodes
//   L : Real; periodic domain length(s)
//   order : int; spline degree r, except cardinal_bspline where it is degree+1
//
// Output:
//   AxisStencil: build periodic one-dimensional spline weights and physical derivatives.
//
// Dependencies:
//   centered_bspline, centered_bspline_derivative.
//   C++17 standard library (headers listed at the top of this file).
// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
// Deposit scalar add
//
// Inputs:
//   mesh : Field&; scalar/vector nodal field
//   pos : const Vec3&; particle/probe position(s)
//   value : Real; value
//   g : const Grid&; periodic grid geometry and Fourier symbols
//   order : int; spline degree r, except cardinal_bspline where it is degree+1
//   divide_by_dV : bool; divide by dV
//
// Output:
//   None; updates the non-const reference arguments in place.
//
// Dependencies:
//   axis_stencil, wrap_pos.
// ---------------------------------------------------------------------------
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


// Orthogonal Fourier projection onto modes off all even-grid Nyquist planes.
// Subtracting each line's alternating mean is exactly the Fourier plane mask,
// but costs O(N), with no extra FFT. The three axis projections commute.
// This is a field-space restriction, NOT particle smoothing or post-step damping.
// ---------------------------------------------------------------------------
// Remove all even-grid Nyquist planes using commuting axis projections
//
// Inputs:
//   f : Field&; f
//   g : const Grid&; periodic grid geometry and Fourier symbols
//
// Output:
//   None; updates the non-const reference arguments in place.
//
// Dependencies:
//   Optional OpenMP; serial loop fallback when OpenMP is disabled.
//   Local scalar/vector types; no external library.
// ---------------------------------------------------------------------------
static void project_nyquist(Field& f, const Grid& g) {
    if (!g.nyquist_projection) return;
    const int dims[3] = {g.nx, g.ny, g.nz};
    const int strides[3] = {g.ny*g.nz, g.nz, 1};
    for (int axis=0; axis<3; ++axis) {
        const int n=dims[axis], stride=strides[axis];
        if (n<=1 || n%2) continue;
        const int lines=g.N/n;
        #pragma omp parallel for schedule(static)
        for (int line=0; line<lines; ++line) {
            const int base=(line/stride)*(n*stride)+line%stride;
            long double sum=0;
            for (int j=0; j<n; ++j) sum+=(j%2 ? -1.0L:1.0L)*f[base+j*stride];
            const Real mean=static_cast<Real>(sum/n);
            for (int j=0; j<n; ++j) f[base+j*stride]-=(j%2 ? -mean:mean);
        }
    }
}

// ---------------------------------------------------------------------------
// Remove all even-grid Nyquist planes using commuting axis projections
//
// Inputs:
//   f : VecField&; f
//   g : const Grid&; periodic grid geometry and Fourier symbols
//
// Output:
//   None; updates the non-const reference arguments in place.
//
// Dependencies:
//   Local scalar/vector types; no external library.
// ---------------------------------------------------------------------------
static void project_nyquist(VecField& f, const Grid& g) {
    for (auto& component:f) project_nyquist(component,g);
}

// ---------------------------------------------------------------------------
// Excluded nyquist mode
//
// Inputs:
//   id : int; id
//   g : const Grid&; periodic grid geometry and Fourier symbols
//
// Output:
//   bool: success/predicate result; non-const reference arguments carry any results.
//
// Dependencies:
//   is_even_nyquist_index.
// ---------------------------------------------------------------------------
static bool excluded_nyquist_mode(int id, const Grid& g) {
    if (!g.nyquist_projection) return false;
    const int i=id/(g.ny*g.nz), j=(id/g.nz)%g.ny, k=id%g.nz;
    return is_even_nyquist_index(i,g.nx) || is_even_nyquist_index(j,g.ny)
        || is_even_nyquist_index(k,g.nz);
}

// ---------------------------------------------------------------------------
// Deposit scalar to mesh
//
// Inputs:
//   pos : const std::vector<Vec3>&; particle/probe position(s)
//   values : const std::vector<Real>&; values
//   g : const Grid&; periodic grid geometry and Fourier symbols
//   order : int; spline degree r, except cardinal_bspline where it is degree+1
//   divide_by_dV : bool; divide by dV
//
// Output:
//   Field: deposit scalar to mesh.
//
// Dependencies:
//   deposit_scalar_add, project_nyquist.
//   C++17 standard library (headers listed at the top of this file).
// ---------------------------------------------------------------------------
static Field deposit_scalar_to_mesh(const std::vector<Vec3>& pos, const std::vector<Real>& values,
                                    const Grid& g, int order, bool divide_by_dV) {
    if (pos.size() != values.size()) throw std::runtime_error("deposit_scalar size mismatch");
    Field mesh(g.N, 0.0);
    // Deliberately serial: this preserves the serial floating-point accumulation
    // order exactly.
    for (std::size_t p = 0; p < pos.size(); ++p) deposit_scalar_add(mesh, pos[p], values[p], g, order, divide_by_dV);
    project_nyquist(mesh,g);
    return mesh;
}

// ---------------------------------------------------------------------------
// Deposit current add
//
// Inputs:
//   J : VecField&; mesh current accumulated in place
//   pos : const Vec3&; particle/probe position(s)
//   vel : const Vec3&; vel
//   charge : Real; particle charge
//   g : const Grid&; periodic grid geometry and Fourier symbols
//   order : int; spline degree r, except cardinal_bspline where it is degree+1
//   prefactor : Real; prefactor
//
// Output:
//   None; updates the non-const reference arguments in place.
//
// Dependencies:
//   axis_stencil, wrap_pos.
// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
// Deposit current to mesh
//
// Inputs:
//   pos : const std::vector<Vec3>&; particle/probe position(s)
//   vel : const std::vector<Vec3>&; vel
//   q : const std::vector<Real>&; particle charge(s)
//   g : const Grid&; periodic grid geometry and Fourier symbols
//   order : int; spline degree r, except cardinal_bspline where it is degree+1
//
// Output:
//   VecField: deposit current to mesh.
//
// Dependencies:
//   deposit_current_add, deposition_thread_count, openmp_thread_num, project_nyquist,
//   zero_vecfield.
//   C++17 standard library (headers listed at the top of this file).
//   Optional OpenMP; serial loop fallback when OpenMP is disabled.
// ---------------------------------------------------------------------------
static VecField deposit_current_to_mesh(const std::vector<Vec3>& pos, const std::vector<Vec3>& vel,
                                        const std::vector<Real>& q, const Grid& g, int order) {
    VecField J = zero_vecfield(g.N);
    const int nt = deposition_thread_count(static_cast<std::size_t>(g.N), pos.size());
    if (nt == 1 || pos.size() < 256) {
        for (std::size_t p = 0; p < pos.size(); ++p) {
            deposit_current_add(J, pos[p], vel[p], q[p], g, order, 1.0);
        }
        project_nyquist(J,g);
        return J;
    }

    // Each worker scatters into a private mesh, removing atomics from the hot
    // loop.  The fixed-order mesh reduction is deterministic for a fixed OpenMP
    // thread count and changes only floating-point association, not the PIC
    // deposition formula.
    std::vector<VecField> local(static_cast<std::size_t>(nt));
    for (auto& mesh : local) mesh = zero_vecfield(g.N);
    #pragma omp parallel num_threads(nt)
    {
        const int tid = openmp_thread_num();
        #pragma omp for schedule(static)
        for (std::size_t p = 0; p < pos.size(); ++p) {
            deposit_current_add(local[tid], pos[p], vel[p], q[p], g, order, 1.0);
        }
        #pragma omp for schedule(static)
        for (int id = 0; id < g.N; ++id) {
            for (int d = 0; d < 3; ++d) {
                Real sum = 0.0;
                for (int t = 0; t < nt; ++t) sum += local[t][d][id];
                J[d][id] = sum;
            }
        }
    }
    project_nyquist(J,g);
    return J;
}

// ---------------------------------------------------------------------------
// Gather scalar at
//
// Inputs:
//   mesh : const Field&; scalar/vector nodal field
//   pos : const Vec3&; particle/probe position(s)
//   g : const Grid&; periodic grid geometry and Fourier symbols
//   order : int; spline degree r, except cardinal_bspline where it is degree+1
//
// Output:
//   Real: gather scalar at.
//
// Dependencies:
//   axis_stencil, wrap_pos.
// ---------------------------------------------------------------------------
[[maybe_unused]] static Real gather_scalar_at(const Field& mesh, const Vec3& pos, const Grid& g, int order) {
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

// ---------------------------------------------------------------------------
// Gather vector at
//
// Inputs:
//   mesh : const VecField&; scalar/vector nodal field
//   pos : const Vec3&; particle/probe position(s)
//   g : const Grid&; periodic grid geometry and Fourier symbols
//   order : int; spline degree r, except cardinal_bspline where it is degree+1
//
// Output:
//   Vec3: gather vector at.
//
// Dependencies:
//   axis_stencil, wrap_pos.
// ---------------------------------------------------------------------------
static Vec3 gather_vector_at(const VecField& mesh, const Vec3& pos, const Grid& g, int order) {
    const Vec3 x = wrap_pos(pos, g.L);
    const AxisStencil sx = axis_stencil(x[0], g.nx, g.L[0], order);
    const AxisStencil sy = axis_stencil(x[1], g.ny, g.L[1], order);
    const AxisStencil sz = axis_stencil(x[2], g.nz, g.L[2], order);
    Vec3 out{0.0, 0.0, 0.0};
    for (int a = 0; a <= order; ++a) {
        for (int b = 0; b <= order; ++b) {
            const Real wxy = sx.w[a] * sy.w[b];
            for (int c = 0; c <= order; ++c) {
                const int id = g.index(sx.idx[a], sy.idx[b], sz.idx[c]);
                const Real w = wxy * sz.w[c];
                out[0] += mesh[0][id] * w;
                out[1] += mesh[1][id] * w;
                out[2] += mesh[2][id] * w;
            }
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// Gather shape gradient scalar at
//
// Inputs:
//   mesh : const Field&; scalar/vector nodal field
//   pos : const Vec3&; particle/probe position(s)
//   g : const Grid&; periodic grid geometry and Fourier symbols
//   order : int; spline degree r, except cardinal_bspline where it is degree+1
//
// Output:
//   Vec3: gather shape gradient scalar at.
//
// Dependencies:
//   axis_stencil, wrap_pos.
// ---------------------------------------------------------------------------
static Vec3 gather_shape_gradient_scalar_at(const Field& mesh, const Vec3& pos,
                                            const Grid& g, int order) {
    const Vec3 x = wrap_pos(pos, g.L);
    const AxisStencil sx = axis_stencil(x[0], g.nx, g.L[0], order);
    const AxisStencil sy = axis_stencil(x[1], g.ny, g.L[1], order);
    const AxisStencil sz = axis_stencil(x[2], g.nz, g.L[2], order);
    Vec3 grad{0.0, 0.0, 0.0};
    for (int a = 0; a <= order; ++a) {
        for (int b = 0; b <= order; ++b) {
            for (int c = 0; c <= order; ++c) {
                const int id = g.index(sx.idx[a], sy.idx[b], sz.idx[c]);
                const Real value = mesh[id];
                grad[0] += value * sx.dw_dx[a] * sy.w[b] * sz.w[c];
                grad[1] += value * sx.w[a] * sy.dw_dx[b] * sz.w[c];
                grad[2] += value * sx.w[a] * sy.w[b] * sz.dw_dx[c];
            }
        }
    }
    return grad;
}

// ---------------------------------------------------------------------------
// Gather vector all
//
// Inputs:
//   pos : const std::vector<Vec3>&; particle/probe position(s)
//   mesh : const VecField&; scalar/vector nodal field
//   g : const Grid&; periodic grid geometry and Fourier symbols
//   order : int; spline degree r, except cardinal_bspline where it is degree+1
//
// Output:
//   std::vector<Vec3>: gather vector all.
//
// Dependencies:
//   gather_vector_at.
//   C++17 standard library (headers listed at the top of this file).
//   Optional OpenMP; serial loop fallback when OpenMP is disabled.
// ---------------------------------------------------------------------------
static std::vector<Vec3> gather_vector_all(const std::vector<Vec3>& pos, const VecField& mesh, const Grid& g, int order) {
    std::vector<Vec3> out(pos.size());
    #pragma omp parallel for schedule(static)
    for (std::size_t p = 0; p < pos.size(); ++p) out[p] = gather_vector_at(mesh, pos[p], g, order);
    return out;
}

// ---------------------------------------------------------------------------
// Zero mat3
//
// Inputs:
//   None.
//
// Output:
//   Mat3: zero mat3.
//
// Dependencies:
//   Local scalar/vector types; no external library.
// ---------------------------------------------------------------------------
static Mat3 zero_mat3() {
    Mat3 M{};
    for (int a = 0; a < 3; ++a) for (int b = 0; b < 3; ++b) M[a][b] = 0.0;
    return M;
}

// ---------------------------------------------------------------------------
// Gather shape gradient vector interp at
//
// Inputs:
//   A0 : const VecField&; vector potential at the beginning of the step
//   A1 : const VecField&; vector potential at the end of the step
//   s_interp : Real; s interp
//   pos : const Vec3&; particle/probe position(s)
//   g : const Grid&; periodic grid geometry and Fourier symbols
//   order : int; spline degree r, except cardinal_bspline where it is degree+1
//
// Output:
//   Mat3: gather shape gradient vector interp at.
//
// Dependencies:
//   axis_stencil, wrap_pos, zero_mat3.
// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
// Gather shape gradient vector at
//
// NOT NEEDED: uncalled legacy/reference helper; retained for provenance.
//
// Inputs:
//   A : const VecField&; nodal vector potential
//   pos : const Vec3&; particle/probe position(s)
//   g : const Grid&; periodic grid geometry and Fourier symbols
//   order : int; spline degree r, except cardinal_bspline where it is degree+1
//
// Output:
//   Mat3: gather shape gradient vector at.
//
// Dependencies:
//   gather_shape_gradient_vector_interp_at.
// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
// Legendre value and derivative
//
// Inputs:
//   n : int; number of entries/nodes
//   x : Real; x
//
// Output:
//   std::pair<Real, Real>: legendre value and derivative.
//
// Dependencies:
//   Local scalar/vector types; no external library.
// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
// Construct Gauss-Legendre quadrature nodes and weights on [0,1]
//
// Inputs:
//   n : int; number of entries/nodes
//
// Output:
//   Quadrature: construct Gauss-Legendre quadrature nodes and weights on [0,1].
//
// Dependencies:
//   legendre_value_and_derivative.
//   C++17 standard library (headers listed at the top of this file).
// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
// Path position
//
// Inputs:
//   x0 : const Vec3&; starting particle position(s)
//   dx_path : const Vec3&; unwrapped orbit displacement(s)
//   s : Real; s
//   g : const Grid&; periodic grid geometry and Fourier symbols
//
// Output:
//   Vec3: path position.
//
// Dependencies:
//   add3, mul3.
// ---------------------------------------------------------------------------
static Vec3 path_position(const Vec3& x0, const Vec3& dx_path, Real s, const Grid& g) {
    // The path is advanced in unwrapped coordinates, then wrapped only for
    // evaluating the periodic mesh shape functions.
    (void)g;
    return add3(x0, mul3(s, dx_path));
}

// ---------------------------------------------------------------------------
// Split an unwrapped straight orbit at every spline knot crossing
//
// Inputs:
//   x0 : const Vec3&; starting particle position(s)
//   dx_path : const Vec3&; unwrapped orbit displacement(s)
//   g : const Grid&; periodic grid geometry and Fourier symbols
//   order : int; spline degree r, except cardinal_bspline where it is degree+1
//   split_at_knots : bool; whether to integrate each polynomial interval separately
//
// Output:
//   std::vector<Real>: split an unwrapped straight orbit at every spline knot crossing.
//
// Dependencies:
//   axis_stencil, wrap1.
//   C++17 standard library (headers listed at the top of this file).
// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
// Compute path stats
//
// Inputs:
//   x0 : const std::vector<Vec3>&; starting particle position(s)
//   dx_path : const std::vector<Vec3>&; unwrapped orbit displacement(s)
//   g : const Grid&; periodic grid geometry and Fourier symbols
//   order : int; spline degree r, except cardinal_bspline where it is degree+1
//   split_at_knots : bool; whether to integrate each polynomial interval separately
//
// Output:
//   PathStats: compute path stats.
//
// Dependencies:
//   path_breakpoints_for_particle.
//   C++17 standard library (headers listed at the top of this file).
//   Optional OpenMP; serial loop fallback when OpenMP is disabled.
// ---------------------------------------------------------------------------
static PathStats compute_path_stats(const std::vector<Vec3>& x0, const std::vector<Vec3>& dx_path,
                                    const Grid& g, int order, bool split_at_knots) {
    PathStats stats;
    stats.n_particles = x0.size();
    stats.max_segments = 1;
    std::size_t crossing_particles = 0;
    std::size_t total_segments = 0;
    int max_segments = 1;
    #pragma omp parallel for reduction(+:crossing_particles,total_segments) reduction(max:max_segments) schedule(static)
    for (std::size_t p = 0; p < x0.size(); ++p) {
        const auto br = path_breakpoints_for_particle(x0[p], dx_path[p], g, order, split_at_knots);
        const int nseg = std::max<int>(1, static_cast<int>(br.size()) - 1);
        total_segments += static_cast<std::size_t>(nseg);
        max_segments = std::max(max_segments, nseg);
        if (nseg > 1) ++crossing_particles;
    }
    stats.crossing_particles = crossing_particles;
    stats.total_segments = total_segments;
    stats.max_segments = max_segments;
    return stats;
}

// ---------------------------------------------------------------------------
// Update path stats
//
// Inputs:
//   stats : PathStats&; stats
//   br : const std::vector<Real>&; br
//
// Output:
//   None; updates the non-const reference arguments in place.
//
// Dependencies:
//   C++17 standard library (headers listed at the top of this file).
// ---------------------------------------------------------------------------
static void update_path_stats(PathStats& stats, const std::vector<Real>& br) {
    const int nseg = std::max<int>(1, static_cast<int>(br.size()) - 1);
    ++stats.n_particles;
    stats.total_segments += static_cast<std::size_t>(nseg);
    stats.max_segments = std::max(stats.max_segments, nseg);
    if (nseg > 1) ++stats.crossing_particles;
}

// ---------------------------------------------------------------------------
// Path average gather vector
//
// Inputs:
//   x0 : const std::vector<Vec3>&; starting particle position(s)
//   dx_path : const std::vector<Vec3>&; unwrapped orbit displacement(s)
//   mesh : const VecField&; scalar/vector nodal field
//   g : const Grid&; periodic grid geometry and Fourier symbols
//   order : int; spline degree r, except cardinal_bspline where it is degree+1
//   quad : const Quadrature&; Gauss nodes and weights on [0,1]
//   split_at_knots : bool; whether to integrate each polynomial interval separately
//
// Output:
//   std::vector<Vec3>: path average gather vector.
//
// Dependencies:
//   gather_vector_at, path_breakpoints_for_particle, path_position.
//   C++17 standard library (headers listed at the top of this file).
//   Optional OpenMP; serial loop fallback when OpenMP is disabled.
// ---------------------------------------------------------------------------
static std::vector<Vec3> path_average_gather_vector(const std::vector<Vec3>& x0,
                                                    const std::vector<Vec3>& dx_path,
                                                    const VecField& mesh, const Grid& g,
                                                    int order, const Quadrature& quad,
                                                    bool split_at_knots = true) {
    std::vector<Vec3> out(x0.size(), Vec3{0.0, 0.0, 0.0});
    #pragma omp parallel for schedule(static)
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

// Scalar-particle versions are used inside the local uniform substep solve.
// They deliberately do not create nested OpenMP regions: the outer particle
// loop supplies the parallelism and the ten substeps of one particle remain
// sequential.
// ---------------------------------------------------------------------------
// Path average gather vector one
//
// Inputs:
//   x0 : const Vec3&; starting particle position(s)
//   dx_path : const Vec3&; unwrapped orbit displacement(s)
//   mesh : const VecField&; scalar/vector nodal field
//   g : const Grid&; periodic grid geometry and Fourier symbols
//   order : int; spline degree r, except cardinal_bspline where it is degree+1
//   quad : const Quadrature&; Gauss nodes and weights on [0,1]
//   split_at_knots : bool; whether to integrate each polynomial interval separately
//
// Output:
//   Vec3: path average gather vector one.
//
// Dependencies:
//   gather_vector_at, path_breakpoints_for_particle, path_position.
//   C++17 standard library (headers listed at the top of this file).
// ---------------------------------------------------------------------------
static Vec3 path_average_gather_vector_one(const Vec3& x0, const Vec3& dx_path,
                                           const VecField& mesh, const Grid& g,
                                           int order, const Quadrature& quad,
                                           bool split_at_knots) {
    Vec3 out{0.0, 0.0, 0.0};
    const auto br = path_breakpoints_for_particle(x0, dx_path, g, order, split_at_knots);
    for (std::size_t seg = 0; seg + 1 < br.size(); ++seg) {
        const Real a = br[seg];
        const Real b = br[seg + 1];
        const Real len = b - a;
        if (len <= 0.0) continue;
        for (std::size_t iq = 0; iq < quad.s.size(); ++iq) {
            const Real s = a + len * quad.s[iq];
            const Real w = len * quad.w[iq];
            const Vec3 xs = path_position(x0, dx_path, s, g);
            const Vec3 val = gather_vector_at(mesh, xs, g, order);
            for (int d = 0; d < 3; ++d) out[d] += w * val[d];
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// Path average deposit current
//
// Inputs:
//   x0 : const std::vector<Vec3>&; starting particle position(s)
//   dx_path : const std::vector<Vec3>&; unwrapped orbit displacement(s)
//   vbar : const std::vector<Vec3>&; orbit/work velocity(s)
//   q : const std::vector<Real>&; particle charge(s)
//   g : const Grid&; periodic grid geometry and Fourier symbols
//   order : int; spline degree r, except cardinal_bspline where it is degree+1
//   quad : const Quadrature&; Gauss nodes and weights on [0,1]
//   split_at_knots : bool; whether to integrate each polynomial interval separately
//   stats : PathStats*; stats
//
// Output:
//   VecField: path average deposit current.
//
// Dependencies:
//   deposit_current_add, deposition_thread_count, openmp_thread_num,
//   path_breakpoints_for_particle, path_position, project_nyquist, update_path_stats,
//   zero_vecfield.
//   C++17 standard library (headers listed at the top of this file).
//   Optional OpenMP; serial loop fallback when OpenMP is disabled.
// ---------------------------------------------------------------------------
static VecField path_average_deposit_current(const std::vector<Vec3>& x0, const std::vector<Vec3>& dx_path,
                                             const std::vector<Vec3>& vbar, const std::vector<Real>& q,
                                             const Grid& g, int order, const Quadrature& quad,
                                             bool split_at_knots = true, PathStats* stats = nullptr) {
    VecField J = zero_vecfield(g.N);
    if (stats) *stats = PathStats{};
    const int nt = deposition_thread_count(static_cast<std::size_t>(g.N), x0.size());
    if (nt == 1 || x0.size() < 256) {
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
        project_nyquist(J,g);
        return J;
    }

    std::vector<VecField> local(static_cast<std::size_t>(nt));
    for (auto& mesh : local) mesh = zero_vecfield(g.N);
    std::size_t crossing_particles = 0;
    std::size_t total_segments = 0;
    int max_segments = 1;
    #pragma omp parallel num_threads(nt) reduction(+:crossing_particles,total_segments) reduction(max:max_segments)
    {
        const int tid = openmp_thread_num();
        #pragma omp for schedule(static)
        for (std::size_t p = 0; p < x0.size(); ++p) {
            const auto br = path_breakpoints_for_particle(x0[p], dx_path[p], g, order, split_at_knots);
            const int nseg = std::max<int>(1, static_cast<int>(br.size()) - 1);
            total_segments += static_cast<std::size_t>(nseg);
            max_segments = std::max(max_segments, nseg);
            if (nseg > 1) ++crossing_particles;
            for (std::size_t seg = 0; seg + 1 < br.size(); ++seg) {
                const Real a = br[seg];
                const Real b = br[seg + 1];
                const Real len = b - a;
                if (len <= 0.0) continue;
                for (std::size_t iq = 0; iq < quad.s.size(); ++iq) {
                    const Real s = a + len * quad.s[iq];
                    const Real w = len * quad.w[iq];
                    const Vec3 xs = path_position(x0[p], dx_path[p], s, g);
                    deposit_current_add(local[tid], xs, vbar[p], q[p], g, order, w);
                }
            }
        }
        #pragma omp for schedule(static)
        for (int id = 0; id < g.N; ++id) {
            for (int d = 0; d < 3; ++d) {
                Real sum = 0.0;
                for (int t = 0; t < nt; ++t) sum += local[t][d][id];
                J[d][id] = sum;
            }
        }
    }
    if (stats) {
        stats->n_particles = x0.size();
        stats->crossing_particles = crossing_particles;
        stats->total_segments = total_segments;
        stats->max_segments = max_segments;
    }
    project_nyquist(J,g);
    return J;
}

// ---------------------------------------------------------------------------
// Orbit discrete gradient a
//
// Inputs:
//   x0 : const std::vector<Vec3>&; starting particle position(s)
//   dx_path : const std::vector<Vec3>&; unwrapped orbit displacement(s)
//   A_n : const VecField&; old vector potential
//   A_np1 : const VecField&; new vector potential
//   g : const Grid&; periodic grid geometry and Fourier symbols
//   order : int; spline degree r, except cardinal_bspline where it is degree+1
//   quad : const Quadrature&; Gauss nodes and weights on [0,1]
//   split_at_knots : bool; whether to integrate each polynomial interval separately
//
// Output:
//   std::vector<Mat3>: orbit discrete gradient a.
//
// Dependencies:
//   gather_shape_gradient_vector_interp_at, path_breakpoints_for_particle, path_position,
//   zero_mat3.
//   C++17 standard library (headers listed at the top of this file).
//   Optional OpenMP; serial loop fallback when OpenMP is disabled.
// ---------------------------------------------------------------------------
static std::vector<Mat3> orbit_discrete_gradient_A(const std::vector<Vec3>& x0, const std::vector<Vec3>& dx_path,
                                                   const VecField& A_n, const VecField& A_np1,
                                                   const Grid& g, int order, const Quadrature& quad,
                                                   bool split_at_knots = true) {
    std::vector<Mat3> D(x0.size(), zero_mat3());
    #pragma omp parallel for schedule(static)
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

// ---------------------------------------------------------------------------
// Gather vector time interp at
//
// Inputs:
//   A_n : const VecField&; old vector potential
//   A_np1 : const VecField&; new vector potential
//   theta : Real; theta
//   pos : const Vec3&; particle/probe position(s)
//   g : const Grid&; periodic grid geometry and Fourier symbols
//   order : int; spline degree r, except cardinal_bspline where it is degree+1
//
// Output:
//   Vec3: gather vector time interp at.
//
// Dependencies:
//   add3, gather_vector_at, mul3.
// ---------------------------------------------------------------------------
static Vec3 gather_vector_time_interp_at(const VecField& A_n, const VecField& A_np1,
                                         Real theta, const Vec3& pos,
                                         const Grid& g, int order) {
    const Vec3 a0 = gather_vector_at(A_n, pos, g, order);
    const Vec3 a1 = gather_vector_at(A_np1, pos, g, order);
    return add3(mul3(1.0 - theta, a0), mul3(theta, a1));
}

// ---------------------------------------------------------------------------
// Orbit discrete gradient a interval
//
// Inputs:
//   x0 : const Vec3&; starting particle position(s)
//   dx_path : const Vec3&; unwrapped orbit displacement(s)
//   A_n : const VecField&; old vector potential
//   A_np1 : const VecField&; new vector potential
//   theta0 : Real; theta0
//   theta1 : Real; theta1
//   g : const Grid&; periodic grid geometry and Fourier symbols
//   order : int; spline degree r, except cardinal_bspline where it is degree+1
//   quad : const Quadrature&; Gauss nodes and weights on [0,1]
//   split_at_knots : bool; whether to integrate each polynomial interval separately
//
// Output:
//   Mat3: orbit discrete gradient a interval.
//
// Dependencies:
//   gather_shape_gradient_vector_interp_at, path_breakpoints_for_particle, path_position,
//   zero_mat3.
//   C++17 standard library (headers listed at the top of this file).
// ---------------------------------------------------------------------------
static Mat3 orbit_discrete_gradient_A_interval(const Vec3& x0, const Vec3& dx_path,
                                               const VecField& A_n, const VecField& A_np1,
                                               Real theta0, Real theta1,
                                               const Grid& g, int order,
                                               const Quadrature& quad,
                                               bool split_at_knots) {
    Mat3 D = zero_mat3();
    const auto br = path_breakpoints_for_particle(x0, dx_path, g, order, split_at_knots);
    for (std::size_t seg = 0; seg + 1 < br.size(); ++seg) {
        const Real a = br[seg];
        const Real b = br[seg + 1];
        const Real len = b - a;
        if (len <= 0.0) continue;
        for (std::size_t iq = 0; iq < quad.s.size(); ++iq) {
            const Real s = a + len * quad.s[iq];
            const Real w = len * quad.w[iq];
            const Real theta = theta0 + (theta1 - theta0) * s;
            const Vec3 xs = path_position(x0, dx_path, s, g);
            const Mat3 G = gather_shape_gradient_vector_interp_at(
                A_n, A_np1, theta, xs, g, order);
            for (int a0 = 0; a0 < 3; ++a0) {
                for (int b0 = 0; b0 < 3; ++b0) D[a0][b0] += w * G[a0][b0];
            }
        }
    }
    return D;
}

struct OrbitGatherGradient {
    std::vector<Vec3> gathered;
    std::vector<Mat3> gradient;
};

// The scalar-potential gather and vector-potential shape gradient use the same
// orbit points and the same tensor-product spline stencil.  Computing them in
// one traversal removes a full set of breakpoint, wrapping, and spline work.
// ---------------------------------------------------------------------------
// Gather the electric-gradient contribution and vector-potential orbit gradient together
//
// Inputs:
//   x0 : const std::vector<Vec3>&; starting particle position(s)
//   dx_path : const std::vector<Vec3>&; unwrapped orbit displacement(s)
//   mesh : const VecField&; scalar/vector nodal field
//   A_n : const VecField&; old vector potential
//   A_np1 : const VecField&; new vector potential
//   g : const Grid&; periodic grid geometry and Fourier symbols
//   order : int; spline degree r, except cardinal_bspline where it is degree+1
//   quad : const Quadrature&; Gauss nodes and weights on [0,1]
//   split_at_knots : bool; whether to integrate each polynomial interval separately
//
// Output:
//   OrbitGatherGradient: gather the electric-gradient contribution and vector-potential orbit gradient together.
//
// Dependencies:
//   axis_stencil, path_breakpoints_for_particle, path_position, wrap_pos, zero_mat3.
//   C++17 standard library (headers listed at the top of this file).
//   Optional OpenMP; serial loop fallback when OpenMP is disabled.
// ---------------------------------------------------------------------------
static OrbitGatherGradient path_average_gather_and_gradient(
        const std::vector<Vec3>& x0, const std::vector<Vec3>& dx_path,
        const VecField& mesh, const VecField& A_n, const VecField& A_np1,
        const Grid& g, int order, const Quadrature& quad, bool split_at_knots = true) {
    OrbitGatherGradient out;
    out.gathered.assign(x0.size(), Vec3{0.0, 0.0, 0.0});
    out.gradient.assign(x0.size(), zero_mat3());

    #pragma omp parallel for schedule(static)
    for (std::size_t p = 0; p < x0.size(); ++p) {
        const auto br = path_breakpoints_for_particle(x0[p], dx_path[p], g, order, split_at_knots);
        for (std::size_t seg = 0; seg + 1 < br.size(); ++seg) {
            const Real a0 = br[seg];
            const Real b0 = br[seg + 1];
            const Real len = b0 - a0;
            if (len <= 0.0) continue;
            for (std::size_t iq = 0; iq < quad.s.size(); ++iq) {
                const Real s = a0 + len * quad.s[iq];
                const Real qweight = len * quad.w[iq];
                const Vec3 xs = wrap_pos(path_position(x0[p], dx_path[p], s, g), g.L);
                const AxisStencil sx = axis_stencil(xs[0], g.nx, g.L[0], order);
                const AxisStencil sy = axis_stencil(xs[1], g.ny, g.L[1], order);
                const AxisStencil sz = axis_stencil(xs[2], g.nz, g.L[2], order);

                for (int ia = 0; ia <= order; ++ia) {
                    for (int ib = 0; ib <= order; ++ib) {
                        const Real wx = sx.w[ia];
                        const Real wy = sy.w[ib];
                        const Real wxy = wx * wy;
                        for (int ic = 0; ic <= order; ++ic) {
                            const int id = g.index(sx.idx[ia], sy.idx[ib], sz.idx[ic]);
                            const Real wz = sz.w[ic];
                            const Real shape = wxy * wz;
                            for (int comp = 0; comp < 3; ++comp) {
                                out.gathered[p][comp] += qweight * mesh[comp][id] * shape;
                            }

                            const Real dSdx[3] = {
                                sx.dw_dx[ia] * wy * wz,
                                wx * sy.dw_dx[ib] * wz,
                                wxy * sz.dw_dx[ic]
                            };
                            for (int comp = 0; comp < 3; ++comp) {
                                const Real Aval = (1.0 - s) * A_n[comp][id] + s * A_np1[comp][id];
                                for (int d = 0; d < 3; ++d) {
                                    out.gradient[p][comp][d] += qweight * Aval * dSdx[d];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    return out;
}

// -----------------------------------------------------------------------------
// Field solve and diagnostics
// -----------------------------------------------------------------------------

struct WaveSpectralState {
    CField phi, psi;
    std::array<CField, 3> A, U;
};

// ---------------------------------------------------------------------------
// Make wave spectral state
//
// Inputs:
//   phi : const Field&; scalar potential
//   psi : const Field&; scalar-potential time derivative
//   A : const VecField&; nodal vector potential
//   U : const VecField&; vector-potential time derivative
//   g : const Grid&; periodic grid geometry and Fourier symbols
//
// Output:
//   WaveSpectralState: make wave spectral state.
//
// Dependencies:
//   fft_real.
// ---------------------------------------------------------------------------
static WaveSpectralState make_wave_spectral_state(const Field& phi, const Field& psi,
                                                  const VecField& A, const VecField& U,
                                                  const Grid& g) {
    WaveSpectralState cache;
    cache.phi = fft_real(phi, g);
    cache.psi = fft_real(psi, g);
    for (int d = 0; d < 3; ++d) {
        cache.A[d] = fft_real(A[d], g);
        cache.U[d] = fft_real(U[d], g);
    }
    return cache;
}

// ---------------------------------------------------------------------------
// Advance one forced wave equation by Crank-Nicolson in Fourier space
//
// Inputs:
//   uhat : const CField&; uhat
//   vhat : const CField&; vhat
//   S_mid : const Field&; S mid
//   dt : Real; field time step
//   g : const Grid&; periodic grid geometry and Fourier symbols
//   kappa : Real; nondimensional light speed
//   source_scale : Real; source scale
//
// Output:
//   std::pair<Field, Field>: advance one forced wave equation by Crank-Nicolson in Fourier space.
//
// Dependencies:
//   excluded_nyquist_mode, fft_real, ifft_real.
//   C++17 standard library (headers listed at the top of this file).
//   Optional OpenMP; serial loop fallback when OpenMP is disabled.
// ---------------------------------------------------------------------------
static std::pair<Field, Field> cn_fft_wave_step_from_spectra(const CField& uhat, const CField& vhat,
                                                             const Field& S_mid, Real dt,
                                                             const Grid& g, Real kappa,
                                                             Real source_scale) {
    const CField Shat = fft_real(S_mid, g);
    CField unew_hat(g.N), vnew_hat(g.N);
    const Real c2 = kappa * kappa;
    const Real source_coeff = c2 * source_scale;
    #pragma omp parallel for schedule(static)
    for (int id = 0; id < g.N; ++id) {
        if (excluded_nyquist_mode(id,g)) {
            unew_hat[id]=vnew_hat[id]=Complex(0.0,0.0);
            continue;
        }
        const Real theta = 0.25 * c2 * dt * dt * g.k2[id];
        const Real denom = 1.0 + theta;
        unew_hat[id] = ((1.0 - theta) * uhat[id] + dt * vhat[id] + 0.5 * source_coeff * dt * dt * Shat[id]) / denom;
        vnew_hat[id] = ((1.0 - theta) * vhat[id] - c2 * dt * g.k2[id] * uhat[id] + source_coeff * dt * Shat[id]) / denom;
    }
    Field unew = ifft_real(std::move(unew_hat), g);
    Field vnew = ifft_real(std::move(vnew_hat), g);
    return {std::move(unew), std::move(vnew)};
}

// ---------------------------------------------------------------------------
// Cn fft wave step
//
// Inputs:
//   u : const Field&; u
//   ut : const Field&; ut
//   S_mid : const Field&; S mid
//   dt : Real; field time step
//   g : const Grid&; periodic grid geometry and Fourier symbols
//   kappa : Real; nondimensional light speed
//   source_scale : Real; source scale
//
// Output:
//   std::pair<Field, Field>: cn fft wave step.
//
// Dependencies:
//   cn_fft_wave_step_from_spectra, fft_real.
// ---------------------------------------------------------------------------
static std::pair<Field, Field> cn_fft_wave_step(const Field& u, const Field& ut, const Field& S_mid,
                                                Real dt, const Grid& g, Real kappa, Real source_scale) {
    const CField uhat = fft_real(u, g);
    const CField vhat = fft_real(ut, g);
    return cn_fft_wave_step_from_spectra(uhat, vhat, S_mid, dt, g, kappa, source_scale);
}

// ---------------------------------------------------------------------------
// Solve periodic poisson for phi
//
// Inputs:
//   rho : const Field&; mesh charge density
//   g : const Grid&; periodic grid geometry and Fourier symbols
//   sigma1 : Real; scalar-potential source coefficient
//
// Output:
//   Field: solve periodic poisson for phi.
//
// Dependencies:
//   excluded_nyquist_mode, fft_real, ifft_real, subtract_mean.
//   C++17 standard library (headers listed at the top of this file).
//   Optional OpenMP; serial loop fallback when OpenMP is disabled.
// ---------------------------------------------------------------------------
static Field solve_periodic_poisson_for_phi(const Field& rho, const Grid& g, Real sigma1) {
    CField rhohat = fft_real(rho, g);
    CField phihat(g.N, Complex(0.0, 0.0));
    #pragma omp parallel for schedule(static)
    for (int id = 0; id < g.N; ++id) {
        if (!excluded_nyquist_mode(id,g) && g.k2[id] > 0.0) phihat[id] = sigma1 * rhohat[id] / g.k2[id];
    }
    Field phi = ifft_real(std::move(phihat), g);
    subtract_mean(phi);
    return phi;
}

struct State {
    std::vector<Vec3> x, v, v_prev, vbar_last, P;
    std::vector<Real> q, m;
    Field phi, psi, rho;
    VecField A, U, J;
    Real dt_last = 0.0; // length of the previous accepted macro step, for variable-dt predictors
    // Persistent uniform-particle resolution. Zero means use the input-deck
    // initial level. It is carried only by accepted states, so retries always
    // start from the unchanged physical state.
    int uniform_substeps_active = 0;
    // Accepted piecewise paths for uniform particle subcycling diagnostics.
    // Empty offsets mean that the ordinary one-orbit macro solver was used.
    std::vector<std::size_t> subcycle_offset_last;
    std::vector<Vec3> subcycle_x0_last, subcycle_vbar_last;
};

// ---------------------------------------------------------------------------
// Advance Lorenz-gauge potentials by the spectral Crank-Nicolson solve
//
// Inputs:
//   st : const State&; accepted particle and mesh state
//   rho_mid : const Field&; midpoint charge density
//   J_mid : const VecField&; midpoint mesh current
//   dt : Real; field time step
//   g : const Grid&; periodic grid geometry and Fourier symbols
//   kappa : Real; nondimensional light speed
//   sigma1 : Real; scalar-potential source coefficient
//   sigma2 : Real; vector-potential source coefficient
//   phi_new : Field&; phi new
//   psi_new : Field&; psi new
//   A_new : VecField&; A new
//   U_new : VecField&; U new
//   cache : const WaveSpectralState*; cache
//
// Output:
//   None; updates the non-const reference arguments in place.
//
// Dependencies:
//   cn_fft_wave_step, cn_fft_wave_step_from_spectra, subtract_mean, zero_vecfield.
//   C++17 standard library (headers listed at the top of this file).
// ---------------------------------------------------------------------------
static void cn_potential_update(const State& st, const Field& rho_mid, const VecField& J_mid,
                                Real dt, const Grid& g, Real kappa, Real sigma1, Real sigma2,
                                Field& phi_new, Field& psi_new, VecField& A_new, VecField& U_new,
                                const WaveSpectralState* cache = nullptr) {
    auto pp = cache
        ? cn_fft_wave_step_from_spectra(cache->phi, cache->psi, rho_mid, dt, g, kappa, sigma1)
        : cn_fft_wave_step(st.phi, st.psi, rho_mid, dt, g, kappa, sigma1);
    phi_new = std::move(pp.first);
    psi_new = std::move(pp.second);
    subtract_mean(phi_new);
    subtract_mean(psi_new);

    A_new = zero_vecfield(g.N);
    U_new = zero_vecfield(g.N);
    for (int d = 0; d < 3; ++d) {
        auto au = cache
            ? cn_fft_wave_step_from_spectra(cache->A[d], cache->U[d], J_mid[d], dt, g, kappa, sigma2)
            : cn_fft_wave_step(st.A[d], st.U[d], J_mid[d], dt, g, kappa, sigma2);
        A_new[d] = std::move(au.first);
        U_new[d] = std::move(au.second);
    }
}

// ---------------------------------------------------------------------------
// Advance mesh charge by the compatible discrete continuity equation
//
// Inputs:
//   rho_n : const Field&; old mesh charge density
//   J_mid : const VecField&; midpoint mesh current
//   dt : Real; field time step
//   g : const Grid&; periodic grid geometry and Fourier symbols
//
// Output:
//   Field: advance mesh charge by the compatible discrete continuity equation.
//
// Dependencies:
//   project_nyquist, spectral_divergence_vector, subtract_mean.
//   Optional OpenMP; serial loop fallback when OpenMP is disabled.
// ---------------------------------------------------------------------------
static Field continuity_charge_update(const Field& rho_n, const VecField& J_mid, Real dt, const Grid& g) {
    Field divJ = spectral_divergence_vector(J_mid, g);
    Field rho(g.N);
    #pragma omp parallel for schedule(static)
    for (int id = 0; id < g.N; ++id) rho[id] = rho_n[id] - dt * divJ[id];
    subtract_mean(rho);
    project_nyquist(rho,g);
    return rho;
}

// ---------------------------------------------------------------------------
// Add vecfield
//
// Inputs:
//   a : const VecField&; a
//   b : const VecField&; b
//   ca : Real; ca
//   cb : Real; cb
//
// Output:
//   VecField: add vecfield.
//
// Dependencies:
//   zero_vecfield.
//   C++17 standard library (headers listed at the top of this file).
//   Optional OpenMP; serial loop fallback when OpenMP is disabled.
// ---------------------------------------------------------------------------
static VecField add_vecfield(const VecField& a, const VecField& b, Real ca, Real cb) {
    VecField out = zero_vecfield(a[0].size());
    for (int d = 0; d < 3; ++d) {
        #pragma omp parallel for schedule(static)
        for (std::size_t i = 0; i < a[d].size(); ++i) out[d][i] = ca * a[d][i] + cb * b[d][i];
    }
    return out;
}

// ---------------------------------------------------------------------------
// Compute e
//
// Inputs:
//   st : const State&; accepted particle and mesh state
//   g : const Grid&; periodic grid geometry and Fourier symbols
//
// Output:
//   VecField: compute e.
//
// Dependencies:
//   spectral_gradient_scalar, zero_vecfield.
//   Optional OpenMP; serial loop fallback when OpenMP is disabled.
// ---------------------------------------------------------------------------
static VecField compute_E(const State& st, const Grid& g) {
    VecField grad_phi = spectral_gradient_scalar(st.phi, g);
    VecField E = zero_vecfield(g.N);
    for (int d = 0; d < 3; ++d) {
        #pragma omp parallel for schedule(static)
        for (int id = 0; id < g.N; ++id) E[d][id] = -grad_phi[d][id] - st.U[d][id];
    }
    return E;
}

// ---------------------------------------------------------------------------
// Integrate the nondimensional electric and magnetic mesh energies
//
// Inputs:
//   st : const State&; accepted particle and mesh state
//   g : const Grid&; periodic grid geometry and Fourier symbols
//   sigma1 : Real; scalar-potential source coefficient
//   sigma2 : Real; vector-potential source coefficient
//
// Output:
//   Real: integrate the nondimensional electric and magnetic mesh energies.
//
// Dependencies:
//   compute_E, spectral_curl.
// ---------------------------------------------------------------------------
static Real compute_field_energy(const State& st, const Grid& g, Real sigma1, Real sigma2) {
    const VecField E = compute_E(st, g);
    const VecField B = spectral_curl(st.A, g);
    long double se = 0.0L, sb = 0.0L;
    for (int d = 0; d < 3; ++d) {
        for (int id = 0; id < g.N; ++id) {
            se += static_cast<long double>(E[d][id]) * static_cast<long double>(E[d][id]);
            sb += static_cast<long double>(B[d][id]) * static_cast<long double>(B[d][id]);
        }
    }
    return static_cast<Real>((0.5 / sigma1 * se + 0.5 / sigma2 * sb) * g.dV);
}




// Apply only when constructing an initial state. Preserve specified mechanical
// velocities while rebuilding canonical momentum with the projected potential.
// During evolution the field solve and projected orbit-current map stay in this
// subspace; never call this as a post-step momentum/energy correction.
// ---------------------------------------------------------------------------
// Project initial mesh variables and reconstruct canonical momenta
//
// Inputs:
//   st : State&; accepted particle and mesh state
//   g : const Grid&; periodic grid geometry and Fourier symbols
//   cfg : const Config&; simulation settings and solver tolerances
//
// Output:
//   None; updates the non-const reference arguments in place.
//
// Dependencies:
//   add3, gather_vector_at, momentum_from_v, mul3, project_nyquist.
//   C++17 standard library (headers listed at the top of this file).
// ---------------------------------------------------------------------------
static void project_initial_state(State& st, const Grid& g, const Config& cfg) {
    if (!g.nyquist_projection) return;
    project_nyquist(st.rho,g); project_nyquist(st.phi,g); project_nyquist(st.psi,g);
    project_nyquist(st.A,g); project_nyquist(st.U,g); project_nyquist(st.J,g);
    for (std::size_t p=0; p<st.x.size(); ++p)
        st.P[p]=add3(momentum_from_v(st.v[p],st.m[p],cfg.kappa),
                     mul3(st.q[p],gather_vector_at(st.A,st.x[p],g,cfg.spline_order)));
}

// ---------------------------------------------------------------------------
// Initialize relativistic history velocity
//
// Inputs:
//   st : State&; accepted particle and mesh state
//   g : const Grid&; periodic grid geometry and Fourier symbols
//   cfg : const Config&; simulation settings and solver tolerances
//
// Output:
//   None; updates the non-const reference arguments in place.
//
// Dependencies:
//   add3, compute_E, cross3, gather_vector_all, momentum_from_v, mul3, spectral_curl, sub3,
//   velocity_from_p.
//   C++17 standard library (headers listed at the top of this file).
// ---------------------------------------------------------------------------
static void initialize_relativistic_history_velocity(State& st, const Grid& g, const Config& cfg) {
    // Predictor history for the first Picard iteration.  This is not part of the
    // accepted conservative update; it only supplies vbar^(0).  We initialize it
    // by one backward Euler/Taylor step in mechanical momentum using the static
    // Lorentz force from the initial fields.
    const VecField E0 = compute_E(st, g);
    const VecField B0 = spectral_curl(st.A, g);
    const auto E0p = gather_vector_all(st.x, E0, g, cfg.spline_order);
    const auto B0p = gather_vector_all(st.x, B0, g, cfg.spline_order);
    st.vbar_last = st.v;
    st.dt_last = cfg.dt;
    for (std::size_t p = 0; p < st.x.size(); ++p) {
        const Vec3 p_now = momentum_from_v(st.v[p], st.m[p], cfg.kappa);
        const Vec3 vxB = cross3(st.v[p], B0p[p]);
        const Vec3 lorentz = mul3(st.q[p], add3(E0p[p], vxB));
        const Vec3 p_prev = sub3(p_now, mul3(cfg.dt, lorentz));
        st.v_prev[p] = velocity_from_p(p_prev, st.m[p], cfg.kappa);
    }
}

// ---------------------------------------------------------------------------
// Sum relativistic particle kinetic energies
//
// Inputs:
//   st : const State&; accepted particle and mesh state
//   c : Real; light speed
//
// Output:
//   Real: sum relativistic particle kinetic energies.
//
// Dependencies:
//   gamma_from_v.
//   C++17 standard library (headers listed at the top of this file).
// ---------------------------------------------------------------------------
static Real compute_kinetic_energy(const State& st, Real c) {
    long double s = 0.0L;
    for (std::size_t p = 0; p < st.v.size(); ++p) {
        // Diagnostics use the accepted endpoint velocity.  The pusher recovers
        // v from p=P-qA, so this is equivalent to summing m kappa^2 (gamma(p)-1)
        // up to roundoff in the velocity recovery.
        const Real gamma = gamma_from_v(st.v[p], c);
        s += static_cast<long double>(st.m[p]) * c * c * (gamma - 1.0);
    }
    return static_cast<Real>(s);
}

struct Diagnostics {
    Real gauss_rms = 0.0;
    Real gauss_max = 0.0;
    Real divE_rms = 0.0;
    Real sigma1_rho_rms = 0.0;
    Real gauss_normalization_rms = 0.0;
    Real gauss_relative = 0.0;
    Real gauge_rms = 0.0;
    Real gauge_max = 0.0;
    Real psi_over_kappa2_rms = 0.0;
    Real divA_rms = 0.0;
    Real gauge_normalization_rms = 0.0;
    Real gauge_relative = 0.0;
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
    Real Bz_y_mode_re = 0.0;
    Real Bz_y_mode_im = 0.0;
    Real Bz_y_mode_abs = 0.0;
    Real mean_rho = 0.0;
};

// ---------------------------------------------------------------------------
// Compute endpoint energies, constraints, and selected Fourier-mode amplitudes
//
// Inputs:
//   st : const State&; accepted particle and mesh state
//   g : const Grid&; periodic grid geometry and Fourier symbols
//   sigma1 : Real; scalar-potential source coefficient
//   sigma2 : Real; vector-potential source coefficient
//   kappa : Real; nondimensional light speed
//   mode : int; mode
//
// Output:
//   Diagnostics: compute endpoint energies, constraints, and selected Fourier-mode amplitudes.
//
// Dependencies:
//   compute_E, compute_field_energy, compute_kinetic_energy, fft_real, mean_field,
//   rms_field, spectral_curl, spectral_divergence_vector.
//   C++17 standard library (headers listed at the top of this file).
//   Optional OpenMP; serial loop fallback when OpenMP is disabled.
// ---------------------------------------------------------------------------
static Diagnostics compute_diagnostics(const State& st, const Grid& g, Real sigma1, Real sigma2, Real kappa, int mode) {
    Diagnostics d;
    const VecField E = compute_E(st, g);
    Field divE = spectral_divergence_vector(E, g);
    Field gauss(g.N);
    #pragma omp parallel for schedule(static)
    for (int id = 0; id < g.N; ++id) gauss[id] = divE[id] - sigma1 * st.rho[id];
    d.gauss_rms = rms_field(gauss);
    d.gauss_max = 0.0;
    for (Real x : gauss) d.gauss_max = std::max(d.gauss_max, std::abs(x));
    d.divE_rms = rms_field(divE);
    d.sigma1_rho_rms = std::abs(sigma1) * rms_field(st.rho);
    d.gauss_normalization_rms = d.divE_rms + d.sigma1_rho_rms;
    d.gauss_relative = d.gauss_normalization_rms > 0.0
        ? d.gauss_rms / d.gauss_normalization_rms
        : (d.gauss_rms == 0.0 ? 0.0 : std::numeric_limits<Real>::infinity());

    Field divA = spectral_divergence_vector(st.A, g);
    Field gauge(g.N);
    #pragma omp parallel for schedule(static)
    for (int id = 0; id < g.N; ++id) gauge[id] = st.psi[id] / (kappa * kappa) + divA[id];
    d.gauge_rms = rms_field(gauge);
    d.gauge_max = 0.0;
    for (Real x : gauge) d.gauge_max = std::max(d.gauge_max, std::abs(x));
    d.psi_over_kappa2_rms = rms_field(st.psi) / (kappa * kappa);
    d.divA_rms = rms_field(divA);
    d.gauge_normalization_rms = d.psi_over_kappa2_rms + d.divA_rms;
    d.gauge_relative = d.gauge_normalization_rms > 0.0
        ? d.gauge_rms / d.gauge_normalization_rms
        : (d.gauge_rms == 0.0 ? 0.0 : std::numeric_limits<Real>::infinity());

    d.field_energy = compute_field_energy(st, g, sigma1, sigma2);
    d.kinetic_energy = compute_kinetic_energy(st, kappa);
    d.total_energy = d.field_energy + d.kinetic_energy;

    CField rhohat = fft_real(st.rho, g);
    CField Exhat = fft_real(E[0], g);
    const VecField B = spectral_curl(st.A, g);
    CField Bzhat = fft_real(B[2], g);
    const int imode = ((mode % g.nx) + g.nx) % g.nx;
    const int jmode = ((mode % g.ny) + g.ny) % g.ny;
    const int id_x = g.index(imode, 0, 0);
    const int id_y = g.index(0, jmode, 0);
    const Real invN = 1.0 / static_cast<Real>(g.N);
    d.rho_mode = std::abs(rhohat[id_x]) * invN;
    d.E_mode_re = Exhat[id_x].real() * invN;
    d.E_mode_im = Exhat[id_x].imag() * invN;
    d.E_mode_abs = std::abs(Exhat[id_x]) * invN;
    d.Bz_y_mode_re = Bzhat[id_y].real() * invN;
    d.Bz_y_mode_im = Bzhat[id_y].imag() * invN;
    d.Bz_y_mode_abs = std::abs(Bzhat[id_y]) * invN;
    d.mean_rho = mean_field(st.rho);
    return d;
}

// Forward declaration: the two-stream initializer uses the same cold-fluid
// reference rate that is later written to the diagnostics.
static Real cold_two_stream_growth_rate(const Config& cfg, const Grid& g);
static Real cold_weibel_growth_rate(const Config& cfg, const Grid& g);

// -----------------------------------------------------------------------------
// Initial condition: 3D embedding of linearly spaced 1D two-stream setup
// -----------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Cell centered grid
//
// Inputs:
//   n : int; number of entries/nodes
//   L : Real; periodic domain length(s)
//   centered : bool; centered
//
// Output:
//   std::vector<Real>: cell centered grid.
//
// Dependencies:
//   C++17 standard library (headers listed at the top of this file).
// ---------------------------------------------------------------------------
static std::vector<Real> cell_centered_grid(int n, Real L, bool centered) {
    std::vector<Real> x(n);
    const Real dx = L / static_cast<Real>(n);
    for (int i = 0; i < n; ++i) {
        x[i] = (static_cast<Real>(i) + 0.5) * dx;
        if (centered) x[i] -= 0.5 * L;
    }
    return x;
}

// ---------------------------------------------------------------------------
// Normal quantile
//
// NOT NEEDED FOR MANUSCRIPT RUNS: retained optional/legacy branch.
//
// Inputs:
//   p : Real; p
//
// Output:
//   Real: normal quantile.
//
// Dependencies:
//   C++17 standard library (headers listed at the top of this file).
// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
// Initialize the cold relativistic longitudinal two-stream eigenmode
//
// Inputs:
//   cfg : const Config&; simulation settings and solver tolerances
//   g : const Grid&; periodic grid geometry and Fourier symbols
//
// Output:
//   State: new particle/field state; input state remains unchanged.
//
// Dependencies:
//   bspline_mode_factor_1d, cell_centered_grid, cold_two_stream_growth_rate,
//   deposit_current_to_mesh, deposit_scalar_to_mesh, gamma_from_v,
//   initialize_relativistic_history_velocity, momentum_from_v, norm3, project_initial_state,
//   solve_periodic_poisson_for_phi, subtract_mean, wrap_pos, zero_vecfield.
//   C++17 standard library (headers listed at the top of this file).
//   Optional OpenMP; serial loop fallback when OpenMP is disabled.
// ---------------------------------------------------------------------------
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
    st.vbar_last.assign(np, Vec3{0.0, 0.0, 0.0});
    st.P.assign(np, Vec3{0.0, 0.0, 0.0});
    st.q.assign(np, 0.0);
    st.m.assign(np, 0.0);

    const Real kx = 2.0 * PI * static_cast<Real>(cfg.perturbation_mode) / g.L[0];
    if (!(kx > 0.0)) throw std::runtime_error("two-stream initialization needs perturbation_mode>0");
    const Real gamma0 = gamma_from_v(Vec3{cfg.v0, 0.0, 0.0}, cfg.kappa);
    const Real gamma03 = gamma0 * gamma0 * gamma0;
    const Real alpha_x = (cfg.two_stream_density_alpha >= 0.0) ? cfg.two_stream_density_alpha : 0.0;
    if (std::abs(alpha_x) >= 0.25) {
        throw std::runtime_error("two_stream_density_alpha is a linear-mode amplitude and should be < 0.25");
    }

    const bool use_growing_eigenmode = cfg.two_stream_eigenmode && cfg.two_stream_density_alpha >= 0.0;
    const Real shape_S = cfg.two_stream_finite_grid_theory ? bspline_mode_factor_1d(cfg.spline_order, kx, g.dx[0]) : 1.0;
    const Real gamma_lin = cold_two_stream_growth_rate(cfg, g);

    // In the eigenmode initializer, alpha_x is the grid charge-density amplitude:
    //      rho_g(x,0) = alpha_x * n0 * cos(kx x).
    // Therefore the grid electric field is E_g(x,0)=E0 sin(kx x), with
    //      E0 = σ1 alpha_x n0/kx.
    // The particles feel the gathered field S(k) E_g, so the cold-fluid
    // eigenvector below uses E_particle_amp = S(k) E0.  When finite-grid theory
    // is disabled, S(k)=1 and these reduce to the continuum formulas.
    const Real E_grid_amp = (use_growing_eigenmode && alpha_x > 0.0) ? (alpha_x * cfg.sigma1 * cfg.n0 / kx) : 0.0;
    const Real E_particle_amp = shape_S * E_grid_amp;
    const Real denom = kx * kx * cfg.v0 * cfg.v0 + gamma_lin * gamma_lin;
    const Real vel_cos_amp = (denom > 0.0) ? (E_particle_amp * kx * cfg.v0 / (gamma03 * denom)) : 0.0;
    const Real vel_sin_amp = (denom > 0.0) ? (E_particle_amp * gamma_lin / (gamma03 * denom)) : 0.0;
    const Real density_cos_amp = (denom > 0.0) ?
        (-kx * E_particle_amp * (kx * kx * cfg.v0 * cfg.v0 - gamma_lin * gamma_lin) / (gamma03 * denom * denom)) : 0.0;
    const Real density_sin_amp = (denom > 0.0) ?
        (2.0 * kx * kx * cfg.v0 * gamma_lin * E_particle_amp / (gamma03 * denom * denom)) : 0.0;

    #pragma omp parallel for schedule(static)
    for (std::size_t p = 0; p < nbase; ++p) {
        const Real phase = kx * base[p][0];
        const Real cph = std::cos(phase);
        const Real sph = std::sin(phase);
        st.x[p] = wrap_pos(base[p], g.L);
        st.x[nbase + p] = wrap_pos(base[p], g.L);

        if (use_growing_eigenmode) {
            // Growing cold relativistic two-stream eigenvector for omega=i*Gamma.
            // Plus/minus beams have opposite cos(kx) velocity perturbations and
            // the same sin(kx) perturbation.  Their density perturbations include
            // a quadrature-phase stream-dependent term.  This is the key fix for
            // the earlier example: a pure density seed is not an eigenmode and can
            // show an initial decay even when the unstable branch exists.
            const Real dv_plus  = +vel_cos_amp * cph - vel_sin_amp * sph;
            const Real dv_minus = -vel_cos_amp * cph - vel_sin_amp * sph;
            st.v[p] = Vec3{+cfg.v0 + dv_plus, 0.0, 0.0};
            st.v[nbase + p] = Vec3{-cfg.v0 + dv_minus, 0.0, 0.0};
        } else {
            // Legacy teaching seed: a velocity perturbation shared by the two
            // beams.  It is useful for nonlinear roll-up demonstrations, but it
            // is not a pure growing eigenmode and should not be used to validate
            // the linear-theory growth rate from a single early-time envelope.
            const Real seed_v = cfg.perturbation * std::sin(kx * base[p][0]);
            st.v[p] = Vec3{+cfg.v0 + seed_v, 0.0, 0.0};
            st.v[nbase + p] = Vec3{-cfg.v0 + seed_v, 0.0, 0.0};
        }
    }

    // A common stream velocity perturbation produces a transverse-wavevector
    // J_x seed without changing the initial charge or Gauss constraint. It is
    // deliberately not advertised as a self-consistent instability eigenmode.
    if (cfg.tsi_transverse_seed != 0.0) {
        for (std::size_t p = 0; p < np; ++p) {
            st.v[p][0] += cfg.tsi_transverse_seed *
                std::cos(2.0 * PI * cfg.tsi_transverse_mode * st.x[p][1] / g.L[1]);
        }
    }
    for (const auto& vv : st.v) {
        if (norm3(vv) >= cfg.kappa) throw std::runtime_error("two-stream initialization generated |v| >= kappa; reduce v0 or seed amplitude");
    }

    const Real volume = g.L[0] * g.L[1] * g.L[2];
    const Real q_macro = -cfg.n0 * volume / static_cast<Real>(np);
    for (std::size_t p = 0; p < nbase; ++p) {
        Real w_plus = 1.0;
        Real w_minus = 1.0;
        if (use_growing_eigenmode) {
            const Real phase = kx * base[p][0];
            const Real cph = std::cos(phase);
            const Real sph = std::sin(phase);
            w_plus  = 1.0 + density_cos_amp * cph + density_sin_amp * sph;
            w_minus = 1.0 + density_cos_amp * cph - density_sin_amp * sph;
        } else if (cfg.two_stream_density_alpha >= 0.0) {
            // Backward-compatible density-only seed.  This branch intentionally
            // preserves the old convention; use two_stream_eigenmode=true for a
            // linearly clean theory comparison.
            const Real phase = kx * base[p][0];
            const Real density_weight = 1.0 + alpha_x * std::cos(phase);
            w_plus = density_weight;
            w_minus = density_weight;
        }
        if (!(w_plus > 0.0) || !(w_minus > 0.0)) {
            throw std::runtime_error("two-stream density/eigenmode seed produced a nonpositive macro weight; reduce alpha");
        }
        st.q[p] = q_macro * w_plus;
        st.q[nbase + p] = q_macro * w_minus;
        st.m[p] = std::abs(st.q[p]);
        st.m[nbase + p] = std::abs(st.q[nbase + p]);
    }

    Field rho_e = deposit_scalar_to_mesh(st.x, st.q, g, cfg.spline_order, true);
    st.rho.assign(g.N, 0.0);
    for (int id = 0; id < g.N; ++id) st.rho[id] = rho_e[id] + cfg.n0;
    subtract_mean(st.rho);

    st.J = deposit_current_to_mesh(st.x, st.v, st.q, g, cfg.spline_order);
    st.phi = solve_periodic_poisson_for_phi(st.rho, g, cfg.sigma1);
    st.psi.assign(g.N, 0.0);
    st.A = zero_vecfield(g.N);
    st.U = zero_vecfield(g.N);

    #pragma omp parallel for schedule(static)
    for (std::size_t p = 0; p < np; ++p) {
        const Vec3 p_mech = momentum_from_v(st.v[p], st.m[p], cfg.kappa);
        st.P[p] = p_mech;  // A=0 for the two-stream initializer.
    }

    project_initial_state(st,g,cfg);
    initialize_relativistic_history_velocity(st, g, cfg);
    return st;
}

// ---------------------------------------------------------------------------
// Initialize landau damping 1d3d
//
// NOT NEEDED FOR MANUSCRIPT RUNS: retained optional/legacy branch.
//
// Inputs:
//   cfg : const Config&; simulation settings and solver tolerances
//   g : const Grid&; periodic grid geometry and Fourier symbols
//
// Output:
//   State: new particle/field state; input state remains unchanged.
//
// Dependencies:
//   cell_centered_grid, deposit_current_to_mesh, deposit_scalar_to_mesh,
//   initialize_relativistic_history_velocity, landau_density_amplitude, momentum_from_v,
//   normal_quantile, project_initial_state, solve_periodic_poisson_for_phi, subtract_mean,
//   wrap_pos, zero_vecfield.
//   C++17 standard library (headers listed at the top of this file).
// ---------------------------------------------------------------------------
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
    st.vbar_last.assign(np, Vec3{0.0, 0.0, 0.0});
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
    st.phi = solve_periodic_poisson_for_phi(st.rho, g, cfg.sigma1);
    st.psi.assign(g.N, 0.0);
    st.A = zero_vecfield(g.N);
    st.U = zero_vecfield(g.N);

    for (std::size_t p = 0; p < np; ++p) {
        const Vec3 p_mech = momentum_from_v(st.v[p], st.m[p], cfg.kappa);
        st.P[p] = p_mech;
    }

    project_initial_state(st,g,cfg);
    initialize_relativistic_history_velocity(st, g, cfg);
    return st;
}


// ---------------------------------------------------------------------------
// Initialize the cold relativistic transverse Weibel eigenmode
//
// Inputs:
//   cfg : const Config&; simulation settings and solver tolerances
//   g : const Grid&; periodic grid geometry and Fourier symbols
//
// Output:
//   State: new particle/field state; input state remains unchanged.
//
// Dependencies:
//   add3, bspline_mode_factor_1d, cell_centered_grid, cold_weibel_growth_rate,
//   deposit_current_to_mesh, deposit_scalar_to_mesh, gamma_from_v, gather_vector_at,
//   initialize_relativistic_history_velocity, momentum_from_v, mul3, norm3,
//   project_initial_state, solve_periodic_poisson_for_phi, subtract_mean, weibel_seed_B0,
//   wrap_pos, zero_vecfield.
//   C++17 standard library (headers listed at the top of this file).
//   Optional OpenMP; serial loop fallback when OpenMP is disabled.
// ---------------------------------------------------------------------------
static State initialize_weibel_filamentation_3d(const Config& cfg, const Grid& g) {
    // ------------------------------------------------------------------
    // Cold relativistic Weibel / filamentation instability.
    // ------------------------------------------------------------------
    // Two cold electron streams occupy the same quiet-start particle positions
    // and move along +/-x.  The perturbation wavevector is transverse, k=k_y yhat.
    //
    // The clean validation setup is not a magnetic field alone.  It is the
    // growing cold-fluid eigenmode used in the paper's Section 5: B_z, U_x
    // (therefore E_x=-U_x), the transverse velocity perturbation, and the
    // stream-opposite density/weight perturbation are all initialized with the
    // same finite-grid B-spline factor.  This removes the large startup transient
    // caused by seeding B_z without the matching particle/current response.
    //
    // Mesh fields for the growing mode:
    //
    //     A_x(y,0) = (B0/k_y) cos(k_y y),
    //     B_z(y,0) = -d_y A_x = B0 sin(k_y y),
    //     U_x(y,0) = Gamma A_x(y,0),
    //     E_x(y,0) = -U_x(y,0).
    //
    // Particle perturbations use gathered mesh amplitudes, S(k) B0, while the
    // current deposited back to the grid contributes another S(k).  Therefore the
    // reference growth rate uses omega_p^2 |S(k)|^2.
    State st;
    const int n_x_per_cell = std::max(1, cfg.particles_per_cell_pair);
    const std::vector<Real> xsub = cell_centered_grid(g.nx * n_x_per_cell, g.L[0], true);
    const std::vector<Real> yline = cell_centered_grid(g.ny, g.L[1], true);
    const std::vector<Real> zline = cell_centered_grid(g.nz, g.L[2], true);

    if (std::abs(cfg.v0) >= cfg.kappa) {
        throw std::runtime_error("Weibel initialization requires |v0| < kappa");
    }

    const int mode = std::max(1, cfg.perturbation_mode);
    const Real ky = 2.0 * PI * static_cast<Real>(mode) / g.L[1];
    if (!(ky > 0.0)) throw std::runtime_error("Weibel initialization needs perturbation_mode>0");
    const Real B0 = weibel_seed_B0(cfg);
    const Real Gamma = cold_weibel_growth_rate(cfg, g);
    if (cfg.weibel_eigenmode && !(Gamma > 0.0)) {
        throw std::runtime_error("Weibel eigenmode requested, but the selected mode is not unstable");
    }
    const Real gamma0 = gamma_from_v(Vec3{cfg.v0, 0.0, 0.0}, cfg.kappa);
    const Real S = cfg.weibel_finite_grid_theory ? bspline_mode_factor_1d(cfg.spline_order, ky, g.dx[1]) : 1.0;

    std::vector<Vec3> base;
    base.reserve(static_cast<std::size_t>(g.nx) * g.ny * g.nz * n_x_per_cell);
    for (Real y : yline) {
        for (Real z : zline) {
            for (Real x : xsub) base.push_back(Vec3{x, y, z});
        }
    }

    const std::size_t nbase = base.size();
    const std::size_t np = 2 * nbase;
    st.x.assign(np, Vec3{0.0, 0.0, 0.0});
    st.v.assign(np, Vec3{0.0, 0.0, 0.0});
    st.v_prev.assign(np, Vec3{0.0, 0.0, 0.0});
    st.vbar_last.assign(np, Vec3{0.0, 0.0, 0.0});
    st.P.assign(np, Vec3{0.0, 0.0, 0.0});
    st.q.assign(np, 0.0);
    st.m.assign(np, 0.0);

    // Eigenmode amplitudes relative to the mesh B_z amplitude B0.  With q/m=-1
    // for electron macro-particles, these are the real growing-mode relations:
    //   dv_x       =  S B0 cos(k y)/(k gamma0^3), same for both streams,
    //   dv_y,+/-   = +/- S v0 B0 sin(k y)/(gamma0 Gamma),
    //   delta n_+  = - (n0/2) S v0 k B0 cos(k y)/(gamma0 Gamma^2),
    //   delta n_-  = + (n0/2) S v0 k B0 cos(k y)/(gamma0 Gamma^2).
    const Real dvx_amp = (cfg.weibel_eigenmode && Gamma > 0.0) ? (S * B0 / (ky * gamma0 * gamma0 * gamma0)) : 0.0;
    const Real dvy_amp = (cfg.weibel_eigenmode && Gamma > 0.0) ? (S * cfg.v0 * B0 / (gamma0 * Gamma)) : 0.0;
    const Real density_amp = (cfg.weibel_eigenmode && Gamma > 0.0) ? (S * cfg.v0 * ky * B0 / (gamma0 * Gamma * Gamma)) : 0.0;
    if (std::abs(density_amp) >= 0.25) {
        throw std::runtime_error("Weibel eigenmode density perturbation is too large; reduce weibel_B0");
    }

    #pragma omp parallel for schedule(static)
    for (std::size_t pidx = 0; pidx < nbase; ++pidx) {
        const Real phase = ky * base[pidx][1];
        const Real cph = std::cos(phase);
        const Real sph = std::sin(phase);
        const Real dvx = dvx_amp * cph;
        const Real dvy_plus = +dvy_amp * sph;
        const Real dvy_minus = -dvy_amp * sph;

        st.x[pidx] = wrap_pos(base[pidx], g.L);
        st.x[nbase + pidx] = wrap_pos(base[pidx], g.L);
        st.v[pidx] = Vec3{+cfg.v0 + dvx, dvy_plus, 0.0};
        st.v[nbase + pidx] = Vec3{-cfg.v0 + dvx, dvy_minus, 0.0};
    }

    for (const auto& vv : st.v) {
        if (norm3(vv) >= cfg.kappa) throw std::runtime_error("Weibel initialization generated |v| >= kappa; reduce v0 or weibel_B0");
    }

    const Real volume = g.L[0] * g.L[1] * g.L[2];
    const Real q_macro = -cfg.n0 * volume / static_cast<Real>(np);
    for (std::size_t pidx = 0; pidx < nbase; ++pidx) {
        const Real phase = ky * base[pidx][1];
        const Real cph = std::cos(phase);
        const Real w_plus = 1.0 - density_amp * cph;
        const Real w_minus = 1.0 + density_amp * cph;
        if (!(w_plus > 0.0) || !(w_minus > 0.0)) {
            throw std::runtime_error("Weibel eigenmode produced a nonpositive stream weight; reduce weibel_B0");
        }
        st.q[pidx] = q_macro * w_plus;
        st.q[nbase + pidx] = q_macro * w_minus;
        st.m[pidx] = std::abs(st.q[pidx]);
        st.m[nbase + pidx] = std::abs(st.q[nbase + pidx]);
    }

    // The stream-opposite density perturbations cancel in total charge, so rho is
    // zero up to particle-mesh roundoff/quiet-start error and phi is therefore
    // zero up to the same tolerance.
    Field rho_e = deposit_scalar_to_mesh(st.x, st.q, g, cfg.spline_order, true);
    st.rho.assign(g.N, 0.0);
    for (int id = 0; id < g.N; ++id) st.rho[id] = rho_e[id] + cfg.n0;
    subtract_mean(st.rho);

    st.phi = solve_periodic_poisson_for_phi(st.rho, g, cfg.sigma1);
    st.psi.assign(g.N, 0.0);
    st.A = zero_vecfield(g.N);
    st.U = zero_vecfield(g.N);

    for (int i = 0; i < g.nx; ++i) {
        for (int j = 0; j < g.ny; ++j) {
            const Real y = -0.5 * g.L[1] + static_cast<Real>(j) * g.dx[1];
            const Real Ax = (B0 / ky) * std::cos(ky * y);
            for (int k = 0; k < g.nz; ++k) {
                const int id = g.index(i, j, k);
                st.A[0][id] = Ax;
                st.U[0][id] = cfg.weibel_eigenmode ? (Gamma * Ax) : 0.0;
            }
        }
    }

    st.J = deposit_current_to_mesh(st.x, st.v, st.q, g, cfg.spline_order);
    #pragma omp parallel for schedule(static)
    for (std::size_t pidx = 0; pidx < np; ++pidx) {
        const Vec3 p_mech = momentum_from_v(st.v[pidx], st.m[pidx], cfg.kappa);
        const Vec3 A0p = gather_vector_at(st.A, st.x[pidx], g, cfg.spline_order);
        st.P[pidx] = add3(p_mech, mul3(st.q[pidx], A0p));
    }

    project_initial_state(st,g,cfg);
    initialize_relativistic_history_velocity(st, g, cfg);
    return st;
}

// ---------------------------------------------------------------------------
// Initialize state
//
// Inputs:
//   cfg : const Config&; simulation settings and solver tolerances
//   g : const Grid&; periodic grid geometry and Fourier symbols
//
// Output:
//   State: new particle/field state; input state remains unchanged.
//
// Dependencies:
//   initialize_landau_damping_1d3d, initialize_two_stream_3d_linear_x,
//   initialize_weibel_filamentation_3d, is_landau_case, is_two_stream_case, is_weibel_case.
//   C++17 standard library (headers listed at the top of this file).
// ---------------------------------------------------------------------------
static State initialize_state(const Config& cfg, const Grid& g) {
    // Centralized switch for teaching examples.  To add a new initial condition,
    // implement another initialize_* function and add one branch here.  The rest
    // of the program intentionally does not know which IC was used.
    if (is_two_stream_case(cfg)) {
        return initialize_two_stream_3d_linear_x(cfg, g);
    }
    if (is_weibel_case(cfg)) {
        return initialize_weibel_filamentation_3d(cfg, g);
    }
    if (is_landau_case(cfg)) {
        return initialize_landau_damping_1d3d(cfg, g);
    }
    throw std::runtime_error("unknown test_case/initial_condition: " + cfg.test_case);
}


// -----------------------------------------------------------------------------
// Linear cold relativistic instability reference rates
// -----------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Cold two stream growth rate
//
// Inputs:
//   cfg : const Config&; simulation settings and solver tolerances
//   g : const Grid&; periodic grid geometry and Fourier symbols
//
// Output:
//   Real: cold two stream growth rate.
//
// Dependencies:
//   bspline_mode_factor_1d, gamma_from_v.
//   C++17 standard library (headers listed at the top of this file).
// ---------------------------------------------------------------------------
static Real cold_two_stream_growth_rate(const Config& cfg, const Grid& g) {
    // Symmetric cold relativistic electron streams, each of density n0/2,
    // streaming at +/-v0 along x with fixed neutralizing ions.  The continuum
    // longitudinal cold-fluid dispersion relation is
    //
    //   1 - (omega_p^2/(2 gamma0^3))/(omega-kv0)^2
    //     - (omega_p^2/(2 gamma0^3))/(omega+kv0)^2 = 0.
    //
    // The PIC code deposits charge/current and gathers fields with the same
    // B-spline.  Linearizing that semi-discrete particle-mesh coupling replaces
    // omega_p^2 by omega_p^2 |S(k)|^2.  This matters on deliberately coarse
    // smoke grids: exact energy conservation can still hold while the physical
    // growth rate is the finite-grid one, not the continuum value.
    const Real k = 2.0 * PI * static_cast<Real>(std::max(1, cfg.perturbation_mode)) / g.L[0];
    const Real gamma0 = gamma_from_v(Vec3{cfg.v0, 0.0, 0.0}, cfg.kappa);
    const Real S = cfg.two_stream_finite_grid_theory ? bspline_mode_factor_1d(cfg.spline_order, k, g.dx[0]) : 1.0;
    const Real omega_p2 = (cfg.sigma1 * cfg.n0) * S * S;
    const Real a = omega_p2 / (2.0 * gamma0 * gamma0 * gamma0);
    const Real b = k * k * cfg.v0 * cfg.v0;
    const Real y_minus = b + a - std::sqrt(a * a + 4.0 * a * b);
    return (y_minus < 0.0) ? std::sqrt(-y_minus) : 0.0;
}

// ---------------------------------------------------------------------------
// Cold weibel growth rate
//
// Inputs:
//   cfg : const Config&; simulation settings and solver tolerances
//   g : const Grid&; periodic grid geometry and Fourier symbols
//
// Output:
//   Real: cold weibel growth rate.
//
// Dependencies:
//   bspline_mode_factor_1d, gamma_from_v.
//   C++17 standard library (headers listed at the top of this file).
// ---------------------------------------------------------------------------
static Real cold_weibel_growth_rate(const Config& cfg, const Grid& g) {
    // Cold symmetric filamentation/Weibel reference for wave vector k_y
    // perpendicular to the beams.  The transverse cold-fluid dispersion used is
    //   omega^4 - (kappa^2 k^2 + omega_p,h^2/gamma0^3) omega^2
    //            - k^2 omega_p,h^2 v0^2/gamma0 = 0.
    // For omega=i Gamma, this gives the positive growth rate below.
    const Real k = 2.0 * PI * static_cast<Real>(std::max(1, cfg.perturbation_mode)) / g.L[1];
    const Real gamma0 = gamma_from_v(Vec3{cfg.v0, 0.0, 0.0}, cfg.kappa);
    const Real S = cfg.weibel_finite_grid_theory ? bspline_mode_factor_1d(cfg.spline_order, k, g.dx[1]) : 1.0;
    const Real omega_p2 = (cfg.sigma1 * cfg.n0) * S * S;
    const Real A = cfg.kappa * cfg.kappa * k * k + omega_p2 / (gamma0 * gamma0 * gamma0);
    const Real C = k * k * omega_p2 * cfg.v0 * cfg.v0 / gamma0;
    const Real gamma2 = 0.5 * (std::sqrt(A * A + 4.0 * C) - A);
    return (gamma2 > 0.0) ? std::sqrt(gamma2) : 0.0;
}

// ---------------------------------------------------------------------------
// Selected theory growth rate
//
// Inputs:
//   cfg : const Config&; simulation settings and solver tolerances
//   g : const Grid&; periodic grid geometry and Fourier symbols
//
// Output:
//   Real: selected theory growth rate.
//
// Dependencies:
//   cold_two_stream_growth_rate, cold_weibel_growth_rate, is_two_stream_case,
//   is_weibel_case.
//   C++17 standard library (headers listed at the top of this file).
// ---------------------------------------------------------------------------
static Real selected_theory_growth_rate(const Config& cfg, const Grid& g) {
    if (std::abs(cfg.theory_gamma) > 0.0) return cfg.theory_gamma;
    if (is_two_stream_case(cfg)) return cold_two_stream_growth_rate(cfg, g);
    if (is_weibel_case(cfg)) return cold_weibel_growth_rate(cfg, g);
    return cfg.theory_gamma;
}

// -----------------------------------------------------------------------------
// R_A,p residual and step diagnostics
// -----------------------------------------------------------------------------

struct RAResult {
    std::vector<Vec3> R;
    std::vector<Vec3> A_delta;
    std::vector<Vec3> rhs;
};

// ---------------------------------------------------------------------------
// Evaluate the vector-potential orbit chain-rule defect
//
// Inputs:
//   x0 : const std::vector<Vec3>&; starting particle position(s)
//   vbar : const std::vector<Vec3>&; orbit/work velocity(s)
//   A_n : const VecField&; old vector potential
//   A_np1 : const VecField&; new vector potential
//   U_mid : const VecField&; U mid
//   dt : Real; field time step
//   g : const Grid&; periodic grid geometry and Fourier symbols
//   order : int; spline degree r, except cardinal_bspline where it is degree+1
//   quad : const Quadrature&; Gauss nodes and weights on [0,1]
//   split_at_knots : bool; whether to integrate each polynomial interval separately
//
// Output:
//   RAResult: evaluate the vector-potential orbit chain-rule defect.
//
// Dependencies:
//   add3, gather_vector_at, mul3, orbit_discrete_gradient_A, path_average_gather_vector,
//   sub3, wrap_pos.
//   C++17 standard library (headers listed at the top of this file).
//   Optional OpenMP; serial loop fallback when OpenMP is disabled.
// ---------------------------------------------------------------------------
static RAResult compute_RA_orbit_residual(const std::vector<Vec3>& x0, const std::vector<Vec3>& vbar,
                                          const VecField& A_n, const VecField& A_np1,
                                          const VecField& U_mid, Real dt, const Grid& g,
                                          int order, const Quadrature& quad,
                                          bool split_at_knots = true) {
    const std::size_t np = x0.size();
    std::vector<Vec3> dx_path(np), x1(np);
    #pragma omp parallel for schedule(static)
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
    #pragma omp parallel for schedule(static)
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

// ---------------------------------------------------------------------------
// Rms from vec3s
//
// Inputs:
//   a : const std::vector<Vec3>&; a
//
// Output:
//   Real: rms from vec3s.
//
// Dependencies:
//   rms_vec_particles.
// ---------------------------------------------------------------------------
static Real rms_from_vec3s(const std::vector<Vec3>& a) { return rms_vec_particles(a); }

// ---------------------------------------------------------------------------
// Evaluate energy/work identities on stored piecewise particle paths
//
// Inputs:
//   s0 : const State&; state at the beginning of the step
//   s1 : const State&; state at the end of the step
//   g : const Grid&; periodic grid geometry and Fourier symbols
//   dt : Real; field time step
//   order : int; spline degree r, except cardinal_bspline where it is degree+1
//   quad : const Quadrature&; Gauss nodes and weights on [0,1]
//   sigma1 : Real; scalar-potential source coefficient
//   sigma2 : Real; vector-potential source coefficient
//   kappa : Real; nondimensional light speed
//   split_at_knots : bool; whether to integrate each polynomial interval separately
//
// Output:
//   StepDiagnostics: chain-rule and particle/grid energy-work defects.
//
// Dependencies:
//   add3, add_vecfield, compute_field_energy, compute_kinetic_energy, dot3,
//   gather_vector_time_interp_at, matvec3, mul3, norm3, orbit_discrete_gradient_A_interval,
//   path_average_gather_vector_one, path_breakpoints_for_particle, spectral_gradient_scalar,
//   sub3, wrap_pos, zero_vecfield.
//   C++17 standard library (headers listed at the top of this file).
//   Optional OpenMP; serial loop fallback when OpenMP is disabled.
// ---------------------------------------------------------------------------
static StepDiagnostics compute_subcycled_step_diagnostics(
        const State& s0, const State& s1, const Grid& g,
        Real dt, int order, const Quadrature& quad,
        Real sigma1, Real sigma2, Real kappa, bool split_at_knots) {
    StepDiagnostics d;
    const std::size_t np = s0.x.size();
    const VecField U_mid = add_vecfield(s0.U, s1.U, 0.5, 0.5);
    const VecField grad_phi0 = spectral_gradient_scalar(s0.phi, g);
    const VecField grad_phi1 = spectral_gradient_scalar(s1.phi, g);
    VecField E_mid = zero_vecfield(g.N);
    for (int m = 0; m < 3; ++m) {
        for (int id = 0; id < g.N; ++id) {
            E_mid[m][id] = -0.5 * (grad_phi0[m][id] + grad_phi1[m][id]) - U_mid[m][id];
        }
    }

    long double r2 = 0.0L, a2 = 0.0L, rhs2 = 0.0L;
    long double norm_sum = 0.0L, work_defect = 0.0L, pw = 0.0L;
    std::size_t total_segments = 0, mesh_crossings = 0, crossing_particle_count = 0;
    Real max_chain_norm = 0.0;
    int max_path_segments = 1;

    #pragma omp parallel for schedule(static) \
        reduction(+:r2,a2,rhs2,norm_sum,work_defect,pw,total_segments,mesh_crossings,crossing_particle_count) \
        reduction(max:max_chain_norm,max_path_segments)
    for (std::size_t p = 0; p < np; ++p) {
        const std::size_t begin = s1.subcycle_offset_last[p];
        const std::size_t end = s1.subcycle_offset_last[p + 1];
        const std::size_t nsub = end - begin;
        const Real h = dt / static_cast<Real>(nsub);
        std::size_t particle_segments = 0;
        bool crossed = false;
        for (std::size_t s = 0; s < nsub; ++s) {
            const std::size_t j = begin + s;
            const Vec3& x0 = s1.subcycle_x0_last[j];
            const Vec3& vbar = s1.subcycle_vbar_last[j];
            const Vec3 dx_path = mul3(h, vbar);
            const Real theta0 = static_cast<Real>(s) / static_cast<Real>(nsub);
            const Real theta1 = static_cast<Real>(s + 1) / static_cast<Real>(nsub);
            const Vec3 x1 = wrap_pos(add3(x0, dx_path), g.L);
            const Vec3 A0p = gather_vector_time_interp_at(
                s0.A, s1.A, theta0, x0, g, order);
            const Vec3 A1p = gather_vector_time_interp_at(
                s0.A, s1.A, theta1, x1, g, order);
            const Vec3 A_delta = sub3(A1p, A0p);
            const Mat3 DA = orbit_discrete_gradient_A_interval(
                x0, dx_path, s0.A, s1.A, theta0, theta1,
                g, order, quad, split_at_knots);
            const Vec3 Ubar = path_average_gather_vector_one(
                x0, dx_path, U_mid, g, order, quad, split_at_knots);
            const Vec3 rhs = mul3(h, add3(Ubar, matvec3(DA, vbar)));
            const Vec3 R = sub3(A_delta, rhs);
            const Real weight = 1.0 / static_cast<Real>(nsub);
            r2 += static_cast<long double>(weight) * dot3(R, R);
            a2 += static_cast<long double>(weight) * dot3(A_delta, A_delta);
            rhs2 += static_cast<long double>(weight) * dot3(rhs, rhs);
            const Real nr = norm3(R);
            max_chain_norm = std::max(max_chain_norm, nr);
            norm_sum += static_cast<long double>(weight) * nr;
            work_defect += -static_cast<long double>(s0.q[p]) * dot3(vbar, R);

            const Vec3 Ebar = path_average_gather_vector_one(
                x0, dx_path, E_mid, g, order, quad, split_at_knots);
            pw += static_cast<long double>(h * s0.q[p]) * dot3(vbar, Ebar);

            const auto br = path_breakpoints_for_particle(
                x0, dx_path, g, order, split_at_knots);
            const std::size_t nseg = std::max<std::size_t>(1, br.size() - 1);
            particle_segments += nseg;
            mesh_crossings += nseg - 1;
            crossed = crossed || nseg > 1;
        }
        total_segments += particle_segments;
        if (crossed) ++crossing_particle_count;
        max_path_segments = std::max(max_path_segments, static_cast<int>(particle_segments));
    }

    const Real denom_count = static_cast<Real>(std::max<std::size_t>(1, 3 * np));
    d.A_chain_abs_rms = std::sqrt(static_cast<Real>(r2) / denom_count);
    const Real arms = std::sqrt(static_cast<Real>(a2) / denom_count);
    const Real rhsrms = std::sqrt(static_cast<Real>(rhs2) / denom_count);
    d.A_chain_rel_rms = d.A_chain_abs_rms /
        (0.5 * (arms + rhsrms) + 1.0e-300);
    d.A_chain_mean_norm = static_cast<Real>(norm_sum /
        std::max<std::size_t>(1, np));
    d.A_chain_max_norm = max_chain_norm;
    d.A_chain_work_defect = static_cast<Real>(work_defect);
    d.crossing_particles = crossing_particle_count;
    d.max_path_segments = max_path_segments;
    d.total_crossings = mesh_crossings;
    d.average_path_segments = np
        ? static_cast<Real>(total_segments) / static_cast<Real>(np) : 1.0;

    long double gw = 0.0L;
    #pragma omp parallel for collapse(2) reduction(+:gw) schedule(static)
    for (int m = 0; m < 3; ++m) {
        for (int id = 0; id < g.N; ++id) {
            gw += static_cast<long double>(s1.J[m][id]) * E_mid[m][id];
        }
    }
    d.particle_work = static_cast<Real>(pw);
    d.grid_work = static_cast<Real>(dt * gw * g.dV);
    d.delta_K = compute_kinetic_energy(s1, kappa) - compute_kinetic_energy(s0, kappa);
    d.delta_W = compute_field_energy(s1, g, sigma1, sigma2) -
                compute_field_energy(s0, g, sigma1, sigma2);
    d.delta_total_energy = d.delta_K + d.delta_W;
    d.deposit_gather_work_residual = d.particle_work - d.grid_work;
    d.particle_energy_residual = d.delta_K - d.particle_work;
    d.field_energy_residual = d.delta_W + d.grid_work;
    return d;
}

// ---------------------------------------------------------------------------
// Evaluate accepted-step chain-rule, work, and energy residuals
//
// Inputs:
//   s0 : const State&; state at the beginning of the step
//   s1 : const State&; state at the end of the step
//   g : const Grid&; periodic grid geometry and Fourier symbols
//   dt : Real; field time step
//   order : int; spline degree r, except cardinal_bspline where it is degree+1
//   quad : const Quadrature&; Gauss nodes and weights on [0,1]
//   sigma1 : Real; scalar-potential source coefficient
//   sigma2 : Real; vector-potential source coefficient
//   kappa : Real; nondimensional light speed
//   split_at_knots : bool; whether to integrate each polynomial interval separately
//
// Output:
//   StepDiagnostics: chain-rule and particle/grid energy-work defects.
//
// Dependencies:
//   add3, add_vecfield, compute_RA_orbit_residual, compute_field_energy,
//   compute_kinetic_energy, compute_path_stats, compute_subcycled_step_diagnostics, dot3,
//   mul3, norm3, path_average_deposit_current, path_average_gather_vector, rms_from_vec3s,
//   spectral_gradient_scalar, zero_vecfield.
//   C++17 standard library (headers listed at the top of this file).
//   Optional OpenMP; serial loop fallback when OpenMP is disabled.
// ---------------------------------------------------------------------------
static StepDiagnostics compute_step_diagnostics(const State& s0, const State& s1, const Grid& g,
                                                Real dt, int order, const Quadrature& quad,
                                                Real sigma1, Real sigma2, Real kappa, bool split_at_knots = true) {
    const std::size_t np = s0.x.size();
    const bool have_subcycle_path = s1.subcycle_offset_last.size() == np + 1 &&
        s1.subcycle_x0_last.size() == s1.subcycle_vbar_last.size() &&
        !s1.subcycle_x0_last.empty() &&
        s1.subcycle_offset_last.back() == s1.subcycle_x0_last.size();
    if (have_subcycle_path) {
        return compute_subcycled_step_diagnostics(
            s0, s1, g, dt, order, quad, sigma1, sigma2, kappa,
            split_at_knots);
    }
    StepDiagnostics d;
    std::vector<Vec3> vbar(np), dx_path(np);
    #pragma omp parallel for schedule(static)
    for (std::size_t p = 0; p < np; ++p) {
        // Relativistic GM-HC uses a kinetic-energy secant velocity for the
        // orbit, current, and particle work.  It is not generally the midpoint
        // of endpoint velocities.  The accepted value is stored in s1.vbar_last.
        vbar[p] = (s1.vbar_last.size() == np) ? s1.vbar_last[p] : mul3(0.5, add3(s1.v[p], s0.v[p]));
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
        #pragma omp parallel for schedule(static)
        for (int id = 0; id < g.N; ++id) E_mid[m][id] = -0.5 * (grad_phi0[m][id] + grad_phi1[m][id]) - U_mid[m][id];
    }
    const auto Ebar_p = path_average_gather_vector(s0.x, dx_path, E_mid, g, order, quad, split_at_knots);
    const VecField Jbar = path_average_deposit_current(s0.x, dx_path, vbar, s0.q, g, order, quad, split_at_knots);

    long double pw = 0.0L, gw = 0.0L;
    for (std::size_t p = 0; p < np; ++p) pw += static_cast<long double>(s0.q[p]) * dot3(vbar[p], Ebar_p[p]);
    for (int m = 0; m < 3; ++m) for (int id = 0; id < g.N; ++id) gw += static_cast<long double>(Jbar[m][id]) * E_mid[m][id];
    d.particle_work = static_cast<Real>(dt * pw);
    d.grid_work = static_cast<Real>(dt * gw * g.dV);

    d.delta_K = compute_kinetic_energy(s1, kappa) - compute_kinetic_energy(s0, kappa);
    d.delta_W = compute_field_energy(s1, g, sigma1, sigma2) - compute_field_energy(s0, g, sigma1, sigma2);
    d.delta_total_energy = d.delta_K + d.delta_W;
    d.deposit_gather_work_residual = d.particle_work - d.grid_work;
    d.particle_energy_residual = d.delta_K - d.particle_work;
    d.field_energy_residual = d.delta_W + d.grid_work;
    return d;
}


// ---------------------------------------------------------------------------
// Skew part from d
//
// Inputs:
//   D : const Mat3&; D
//
// Output:
//   Mat3: skew part from d.
//
// Dependencies:
//   zero_mat3.
// ---------------------------------------------------------------------------
static Mat3 skew_part_from_D(const Mat3& D) {
    Mat3 K = zero_mat3();
    for (int a = 0; a < 3; ++a) {
        for (int b = 0; b < 3; ++b) K[a][b] = D[b][a] - D[a][b];
    }
    return K;
}

// ---------------------------------------------------------------------------
// Beff from skew k
//
// Inputs:
//   K : const Mat3&; K
//
// Output:
//   Vec3: beff from skew k.
//
// Dependencies:
//   test_skew_K_to_Beff_sign.
// ---------------------------------------------------------------------------
static Vec3 beff_from_skew_K(const Mat3& K) {
    // Convention: K v = v x B_eff.  With A=(0,Bx,0),
    //
    //     D = grad A = [[0,0,0],[B,0,0],[0,0,0]],
    //     K = D^T-D = [[0,B,0],[-B,0,0],[0,0,0]],
    //
    // so K(vx,vy,vz)=(B vy,-B vx,0)=v x (0,0,B).  Therefore
    // B_eff=(K_23,-K_13,K_12) in one-based notation, i.e. the
    // 0-based expression below.  The unit test test_skew_K_to_Beff_sign()
    // exists specifically to prevent this sign convention from regressing.
    return Vec3{K[1][2], -K[0][2], K[0][1]};
}

// ---------------------------------------------------------------------------
// Apply the relativistic Higuera-Cary rotation generated by the orbit skew matrix
//
// Inputs:
//   p_minus : const Vec3&; p minus
//   K : const Mat3&; K
//   q : Real; particle charge(s)
//   m : Real; particle mass
//   kappa : Real; nondimensional light speed
//   dt : Real; field time step
//
// Output:
//   Vec3: apply the relativistic Higuera-Cary rotation generated by the orbit skew matrix.
//
// Dependencies:
//   add3, beff_from_skew_K, cross3, dot3, mul3.
//   C++17 standard library (headers listed at the top of this file).
// ---------------------------------------------------------------------------
static Vec3 higuera_cary_rotate_with_K(const Vec3& p_minus, const Mat3& K,
                                       Real q, Real m, Real kappa, Real dt) {
    // Higuera-Cary magnetic rotation for dp/dt = q K v, where K v = v x B_eff.
    // The rotation is applied to normalized mechanical momentum u=p/(m kappa).
    // It preserves |u| (and therefore relativistic kinetic energy) for frozen K.
    const Vec3 B = beff_from_skew_K(K);
    Vec3 u = mul3(1.0 / (m * kappa), p_minus);
    const Real gamma_minus = std::sqrt(1.0 + dot3(u, u));
    const Vec3 tau = mul3(q * dt / (2.0 * m), B);
    const Real tau2 = dot3(tau, tau);
    const Real udottau = dot3(u, tau);
    const Real sigma = gamma_minus * gamma_minus - tau2;
    const Real gamma_hc2 = 0.5 * (sigma + std::sqrt(sigma * sigma + 4.0 * (tau2 + udottau * udottau)));
    const Real gamma_hc = std::sqrt(std::max<Real>(gamma_hc2, 1.0));
    const Vec3 t = mul3(1.0 / gamma_hc, tau);
    const Vec3 svec = mul3(2.0 / (1.0 + dot3(t, t)), t);
    const Vec3 uprime = add3(u, cross3(u, t));
    const Vec3 uplus = add3(u, cross3(uprime, svec));
    return mul3(m * kappa, uplus);
}

// -----------------------------------------------------------------------------
// Picard map and time step
// -----------------------------------------------------------------------------

// NOT NEEDED: data container used only by the legacy evaluate_picard_map_orbit.
struct PicardTrial {
    std::vector<Vec3> vbar, vbar_new, dx_path, x_end;
    VecField Jbar;
    Field rho_end, rho_mid;
    Field phi_end, psi_end;
    VecField A_end, U_end;
    std::vector<Vec3> P_end, p_end, v_next;
    PathStats path_stats;
};

// ---------------------------------------------------------------------------
// Legacy simultaneous orbit-velocity Picard map (not used by the nested solver)
//
// NOT NEEDED: uncalled legacy/reference helper; retained for provenance.
//
// Inputs:
//   vbar_guess : const std::vector<Vec3>&; trial orbit/work velocities
//   st : const State&; accepted particle and mesh state
//   g : const Grid&; periodic grid geometry and Fourier symbols
//   cfg : const Config&; simulation settings and solver tolerances
//   grad_phi_n : const VecField&; old spectral scalar-potential gradient
//   A0_at_particles : const std::vector<Vec3>&; A0 at particles
//   wave_cache : const WaveSpectralState&; cached old-state Fourier transforms
//   quad : const Quadrature&; Gauss nodes and weights on [0,1]
//
// Output:
//   PicardTrial: legacy simultaneous orbit-velocity Picard map (not used by the nested solver).
//
// Dependencies:
//   add3, cn_potential_update, continuity_charge_update, gather_vector_at,
//   higuera_cary_rotate_with_K, kinetic_secant_velocity, matvec3, mul3,
//   path_average_deposit_current, path_average_gather_and_gradient, skew_part_from_D,
//   spectral_gradient_scalar, sub3, velocity_from_p, wrap_pos, zero_vecfield.
//   C++17 standard library (headers listed at the top of this file).
//   Optional OpenMP; serial loop fallback when OpenMP is disabled.
// ---------------------------------------------------------------------------
[[maybe_unused]] static PicardTrial evaluate_picard_map_orbit(const std::vector<Vec3>& vbar_guess,
                                             const State& st, const Grid& g, const Config& cfg,
                                             const VecField& grad_phi_n,
                                             const std::vector<Vec3>& A0_at_particles,
                                             const WaveSpectralState& wave_cache,
                                             const Quadrature& quad) {
    const std::size_t np = st.x.size();
    PicardTrial tr;
    tr.vbar.assign(np, Vec3{0.0, 0.0, 0.0});
    tr.vbar_new.assign(np, Vec3{0.0, 0.0, 0.0});
    tr.dx_path.assign(np, Vec3{0.0, 0.0, 0.0});
    tr.x_end.assign(np, Vec3{0.0, 0.0, 0.0});
    tr.P_end.assign(np, Vec3{0.0, 0.0, 0.0});
    tr.p_end.assign(np, Vec3{0.0, 0.0, 0.0});
    tr.v_next.assign(np, Vec3{0.0, 0.0, 0.0});

    #pragma omp parallel for schedule(static)
    for (std::size_t p = 0; p < np; ++p) {
        tr.vbar[p] = vbar_guess[p];
        tr.dx_path[p] = mul3(cfg.dt, tr.vbar[p]);
        tr.x_end[p] = wrap_pos(add3(st.x[p], tr.dx_path[p]), g.L);
    }

    // The same relativistic work velocity vbar is used in the orbit, in the
    // current deposition, and in the particle energy identity.
    tr.Jbar = path_average_deposit_current(st.x, tr.dx_path, tr.vbar, st.q, g, cfg.spline_order, quad, cfg.split_orbit_at_knots, &tr.path_stats);
    tr.rho_end = continuity_charge_update(st.rho, tr.Jbar, cfg.dt, g);
    tr.rho_mid.assign(g.N, 0.0);
    #pragma omp parallel for schedule(static)
    for (int id = 0; id < g.N; ++id) tr.rho_mid[id] = 0.5 * (tr.rho_end[id] + st.rho[id]);

    cn_potential_update(st, tr.rho_mid, tr.Jbar, cfg.dt, g, cfg.kappa, cfg.sigma1, cfg.sigma2,
                        tr.phi_end, tr.psi_end, tr.A_end, tr.U_end, &wave_cache);

    VecField grad_phi_end = spectral_gradient_scalar(tr.phi_end, g);
    VecField grad_phi_mid = zero_vecfield(g.N);
    for (int d = 0; d < 3; ++d) {
        #pragma omp parallel for schedule(static)
        for (int id = 0; id < g.N; ++id) grad_phi_mid[d][id] = 0.5 * (grad_phi_n[d][id] + grad_phi_end[d][id]);
    }

    OrbitGatherGradient orbit_fields = path_average_gather_and_gradient(
        st.x, tr.dx_path, grad_phi_mid, st.A, tr.A_end, g, cfg.spline_order,
        quad, cfg.split_orbit_at_knots);
    const auto& grad_phi_bar_p = orbit_fields.gathered;
    const auto& DA = orbit_fields.gradient;

    #pragma omp parallel for schedule(static)
    for (std::size_t p = 0; p < np; ++p) {
        const Vec3& A0p = A0_at_particles[p];
        const Vec3 A1p = gather_vector_at(tr.A_end, tr.x_end[p], g, cfg.spline_order);
        const Vec3 Aep = mul3(0.5, add3(A0p, A1p));
        const Vec3 p_n = sub3(st.P[p], mul3(st.q[p], A0p));

        const Vec3 Dvbar = matvec3(DA[p], tr.vbar[p]);
        const Mat3 K = skew_part_from_D(DA[p]);

        // Canonical no-explicit-A_t form of the relativistic GM-HC step.
        // The chain-rule piece D vbar handles the endpoint/gauge part of the
        // vector-potential change.  The skew part K=D^T-D is advanced by an HC
        // magnetic rotation.  No finite-difference A_t is formed in the force.
        Vec3 p_minus = st.P[p];
        p_minus = sub3(p_minus, mul3(st.q[p], Aep));
        p_minus = sub3(p_minus, mul3(0.5 * st.q[p] * cfg.dt, grad_phi_bar_p[p]));
        p_minus = add3(p_minus, mul3(0.5 * st.q[p] * cfg.dt, Dvbar));

        const Vec3 p_plus = higuera_cary_rotate_with_K(p_minus, K, st.q[p], st.m[p], cfg.kappa, cfg.dt);

        Vec3 P_end = p_plus;
        P_end = add3(P_end, mul3(st.q[p], Aep));
        P_end = sub3(P_end, mul3(0.5 * st.q[p] * cfg.dt, grad_phi_bar_p[p]));
        P_end = add3(P_end, mul3(0.5 * st.q[p] * cfg.dt, Dvbar));
        tr.P_end[p] = P_end;
        tr.p_end[p] = sub3(P_end, mul3(st.q[p], A1p));
        tr.v_next[p] = velocity_from_p(tr.p_end[p], st.m[p], cfg.kappa);

        const Vec3 vEminus = kinetic_secant_velocity(p_minus, p_n, st.m[p], cfg.kappa);
        const Vec3 vEplus  = kinetic_secant_velocity(tr.p_end[p], p_plus, st.m[p], cfg.kappa);
        tr.vbar_new[p] = mul3(0.5, add3(vEminus, vEplus));
    }
    return tr;
}

struct UniformSubcycleTrial {
    std::vector<Vec3> current_guess, current_map;
    std::vector<Vec3> sub_x0, sub_dx, sub_vbar;
    std::vector<std::size_t> subcycle_offset;
    std::vector<Vec3> x_end, P_end, p_end, v_next, vbar_macro;
    VecField J_field, J_map;
    Field rho_end, rho_mid, phi_end, psi_end;
    VecField A_end, U_end;
    PathStats path_stats;
    Real local_scaled_residual_max = 0.0;
    int local_iterations_max = 0;
    Real local_iterations_mean = 0.0;
    int substeps_used = 0;
    std::size_t total_particle_substeps = 0;
    std::size_t refined_particles = 0;
};

struct LocalParticleStep {
    Vec3 vbar_map{0.0, 0.0, 0.0};
    Vec3 P_end{0.0, 0.0, 0.0};
    Vec3 p_end{0.0, 0.0, 0.0};
    Vec3 v_end{0.0, 0.0, 0.0};
};

struct LocalParticleSolveReport {
    int iterations = 0;
    LocalParticleFailureReason failure_reason =
        LocalParticleFailureReason::maximum_iterations;
    Real first_scaled_residual = std::numeric_limits<Real>::infinity();
    Real final_scaled_residual = std::numeric_limits<Real>::infinity();
    Real final_absolute_residual = std::numeric_limits<Real>::infinity();
    Vec3 last_guess{0.0, 0.0, 0.0};
    Vec3 last_map{0.0, 0.0, 0.0};
};

// ---------------------------------------------------------------------------
// Evaluate one frozen-field canonical GM-HC orbit map
//
// Inputs:
//   vbar_guess : const Vec3&; trial orbit/work velocities
//   x0 : const Vec3&; starting particle position(s)
//   P0 : const Vec3&; P0
//   q : Real; particle charge(s)
//   m : Real; particle mass
//   h : Real; local particle time step
//   theta0 : Real; theta0
//   theta1 : Real; theta1
//   st : const State&; accepted particle and mesh state
//   A_end : const VecField&; A end
//   grad_phi_mid : const VecField&; midpoint spectral scalar-potential gradient
//   g : const Grid&; periodic grid geometry and Fourier symbols
//   cfg : const Config&; simulation settings and solver tolerances
//   quad : const Quadrature&; Gauss nodes and weights on [0,1]
//
// Output:
//   LocalParticleStep: evaluate one frozen-field canonical GM-HC orbit map.
//
// Dependencies:
//   add3, gather_vector_time_interp_at, higuera_cary_rotate_with_K, kinetic_secant_velocity,
//   matvec3, mul3, orbit_discrete_gradient_A_interval, path_average_gather_vector_one,
//   skew_part_from_D, sub3, velocity_from_p, wrap_pos.
// ---------------------------------------------------------------------------
static LocalParticleStep evaluate_local_particle_substep(
        const Vec3& vbar_guess, const Vec3& x0, const Vec3& P0,
        Real q, Real m, Real h, Real theta0, Real theta1,
        const State& st, const VecField& A_end,
        const VecField& grad_phi_mid, const Grid& g,
        const Config& cfg, const Quadrature& quad) {
    LocalParticleStep out;
    const Vec3 dx_path = mul3(h, vbar_guess);
    const Vec3 x1 = wrap_pos(add3(x0, dx_path), g.L);
    const Vec3 A0p = gather_vector_time_interp_at(
        st.A, A_end, theta0, x0, g, cfg.spline_order);
    const Vec3 A1p = gather_vector_time_interp_at(
        st.A, A_end, theta1, x1, g, cfg.spline_order);
    const Vec3 Aep = mul3(0.5, add3(A0p, A1p));
    const Vec3 p_start = sub3(P0, mul3(q, A0p));
    const Vec3 grad_phi_bar = path_average_gather_vector_one(
        x0, dx_path, grad_phi_mid, g, cfg.spline_order, quad,
        cfg.split_orbit_at_knots);
    const Mat3 DA = orbit_discrete_gradient_A_interval(
        x0, dx_path, st.A, A_end, theta0, theta1,
        g, cfg.spline_order, quad, cfg.split_orbit_at_knots);
    const Vec3 Dvbar = matvec3(DA, vbar_guess);
    const Mat3 K = skew_part_from_D(DA);

    Vec3 p_minus = P0;
    p_minus = sub3(p_minus, mul3(q, Aep));
    p_minus = sub3(p_minus, mul3(0.5 * q * h, grad_phi_bar));
    p_minus = add3(p_minus, mul3(0.5 * q * h, Dvbar));
    const Vec3 p_plus = higuera_cary_rotate_with_K(
        p_minus, K, q, m, cfg.kappa, h);

    out.P_end = p_plus;
    out.P_end = add3(out.P_end, mul3(q, Aep));
    out.P_end = sub3(out.P_end, mul3(0.5 * q * h, grad_phi_bar));
    out.P_end = add3(out.P_end, mul3(0.5 * q * h, Dvbar));
    out.p_end = sub3(out.P_end, mul3(q, A1p));
    out.v_end = velocity_from_p(out.p_end, m, cfg.kappa);
    const Vec3 vEminus = kinetic_secant_velocity(p_minus, p_start, m, cfg.kappa);
    const Vec3 vEplus = kinetic_secant_velocity(out.p_end, p_plus, m, cfg.kappa);
    out.vbar_map = mul3(0.5, add3(vEminus, vEplus));
    return out;
}

// ---------------------------------------------------------------------------
// Finite vec3
//
// Inputs:
//   v : const Vec3&; v
//
// Output:
//   bool: success/predicate result; non-const reference arguments carry any results.
//
// Dependencies:
//   C++17 standard library (headers listed at the top of this file).
// ---------------------------------------------------------------------------
static bool finite_vec3(const Vec3& v) {
    return std::isfinite(v[0]) && std::isfinite(v[1]) && std::isfinite(v[2]);
}

// ---------------------------------------------------------------------------
// Converge one frozen-field GM-HC particle orbit with mixed residual tolerances
//
// Inputs:
//   initial_guess : const Vec3&; initial guess
//   x0 : const Vec3&; starting particle position(s)
//   P0 : const Vec3&; P0
//   q : Real; particle charge(s)
//   m : Real; particle mass
//   h : Real; local particle time step
//   theta0 : Real; theta0
//   theta1 : Real; theta1
//   st : const State&; accepted particle and mesh state
//   A_end : const VecField&; A end
//   grad_phi_mid : const VecField&; midpoint spectral scalar-potential gradient
//   g : const Grid&; periodic grid geometry and Fourier symbols
//   cfg : const Config&; simulation settings and solver tolerances
//   quad : const Quadrature&; Gauss nodes and weights on [0,1]
//   accepted_vbar : Vec3&; accepted vbar
//   accepted_map : LocalParticleStep&; accepted map
//   report : LocalParticleSolveReport&; report
//
// Output:
//   bool: success/predicate result; non-const reference arguments carry any results.
//
// Dependencies:
//   add3, evaluate_local_particle_substep, finite_vec3, mul3, norm3, sub3.
// ---------------------------------------------------------------------------
static bool solve_local_particle_substep(
        const Vec3& initial_guess, const Vec3& x0, const Vec3& P0,
        Real q, Real m, Real h, Real theta0, Real theta1,
        const State& st, const VecField& A_end,
        const VecField& grad_phi_mid, const Grid& g,
        const Config& cfg, const Quadrature& quad,
        Vec3& accepted_vbar, LocalParticleStep& accepted_map,
        LocalParticleSolveReport& report) {
    Vec3 guess = initial_guess;
    report = LocalParticleSolveReport{};
    report.last_guess = guess;
    for (int it = 0; it < cfg.uniform_local_max_iter; ++it) {
        const LocalParticleStep mapped = evaluate_local_particle_substep(
            guess, x0, P0, q, m, h, theta0, theta1,
            st, A_end, grad_phi_mid, g, cfg, quad);
        ++report.iterations;
        report.last_guess = guess;
        report.last_map = mapped.vbar_map;
        if (!finite_vec3(mapped.vbar_map)) {
            report.failure_reason = LocalParticleFailureReason::nonfinite_map;
            return false;
        }
        if (norm3(mapped.vbar_map) >= cfg.kappa) {
            report.failure_reason = LocalParticleFailureReason::superluminal_map;
            return false;
        }
        const Vec3 residual = sub3(mapped.vbar_map, guess);
        const Real scale = 0.5 * (norm3(mapped.vbar_map) + norm3(guess));
        report.final_absolute_residual = norm3(residual);
        report.final_scaled_residual = report.final_absolute_residual /
            (cfg.uniform_local_atol + cfg.uniform_local_rtol * scale);
        if (report.iterations == 1) {
            report.first_scaled_residual = report.final_scaled_residual;
        }
        if (report.final_scaled_residual <= 1.0) {
            accepted_vbar = guess;
            accepted_map = mapped;
            report.failure_reason = LocalParticleFailureReason::none;
            return true;
        }
        guess = add3(guess, mul3(cfg.uniform_local_damping, residual));
        report.last_guess = guess;
        if (!finite_vec3(guess)) {
            report.failure_reason = LocalParticleFailureReason::nonfinite_iterate;
            return false;
        }
        if (norm3(guess) >= cfg.kappa) {
            report.failure_reason = LocalParticleFailureReason::superluminal_iterate;
            return false;
        }
    }
    report.failure_reason = LocalParticleFailureReason::maximum_iterations;
    return false;
}

// ---------------------------------------------------------------------------
// Node vectors to vecfield
//
// Inputs:
//   values : const std::vector<Vec3>&; values
//   g : const Grid&; periodic grid geometry and Fourier symbols
//
// Output:
//   VecField: node vectors to vecfield.
//
// Dependencies:
//   zero_vecfield.
//   C++17 standard library (headers listed at the top of this file).
//   Optional OpenMP; serial loop fallback when OpenMP is disabled.
// ---------------------------------------------------------------------------
static VecField node_vectors_to_vecfield(const std::vector<Vec3>& values,
                                         const Grid& g) {
    if (values.size() != static_cast<std::size_t>(g.N)) {
        throw std::runtime_error("grid-current vector has the wrong size");
    }
    VecField out = zero_vecfield(g.N);
    #pragma omp parallel for schedule(static)
    for (int id = 0; id < g.N; ++id) {
        for (int d = 0; d < 3; ++d) out[d][id] = values[id][d];
    }
    return out;
}

// ---------------------------------------------------------------------------
// Vecfield to node vectors
//
// Inputs:
//   values : const VecField&; values
//   g : const Grid&; periodic grid geometry and Fourier symbols
//
// Output:
//   std::vector<Vec3>: vecfield to node vectors.
//
// Dependencies:
//   C++17 standard library (headers listed at the top of this file).
//   Optional OpenMP; serial loop fallback when OpenMP is disabled.
// ---------------------------------------------------------------------------
static std::vector<Vec3> vecfield_to_node_vectors(const VecField& values,
                                                  const Grid& g) {
    std::vector<Vec3> out(static_cast<std::size_t>(g.N), Vec3{0.0, 0.0, 0.0});
    #pragma omp parallel for schedule(static)
    for (int id = 0; id < g.N; ++id) {
        for (int d = 0; d < 3; ++d) out[id][d] = values[d][id];
    }
    return out;
}

// ---------------------------------------------------------------------------
// Finite node vector
//
// Inputs:
//   values : const std::vector<Vec3>&; values
//
// Output:
//   bool: success/predicate result; non-const reference arguments carry any results.
//
// Dependencies:
//   finite_vec3.
//   C++17 standard library (headers listed at the top of this file).
//   Optional OpenMP; serial loop fallback when OpenMP is disabled.
// ---------------------------------------------------------------------------
static bool finite_node_vector(const std::vector<Vec3>& values) {
    int finite = 1;
    #pragma omp parallel for reduction(&:finite) schedule(static)
    for (std::size_t i = 0; i < values.size(); ++i) {
        finite &= finite_vec3(values[i]);
    }
    return finite != 0;
}

// ---------------------------------------------------------------------------
// Evaluate uniform current map at substeps
//
// NOT NEEDED FOR MANUSCRIPT RUNS: retained optional/legacy branch.
//
// Inputs:
//   current_guess : const std::vector<Vec3>&; trial mesh current, stored as node vectors
//   st : const State&; accepted particle and mesh state
//   g : const Grid&; periodic grid geometry and Fourier symbols
//   cfg : const Config&; simulation settings and solver tolerances
//   grad_phi_n : const VecField&; old spectral scalar-potential gradient
//   wave_cache : const WaveSpectralState&; cached old-state Fourier transforms
//   quad : const Quadrature&; Gauss nodes and weights on [0,1]
//   requested_substeps : int; requested substeps
//
// Output:
//   UniformSubcycleTrial: evaluate uniform current map at substeps.
//
// Dependencies:
//   add3, cn_potential_update, continuity_charge_update, gather_vector_at, mul3,
//   node_vectors_to_vecfield, path_average_deposit_current, project_nyquist,
//   solve_local_particle_substep, spectral_gradient_scalar, sub3, vecfield_to_node_vectors,
//   wrap_pos, zero_vecfield.
//   C++17 standard library (headers listed at the top of this file).
//   Optional OpenMP; serial loop fallback when OpenMP is disabled.
// ---------------------------------------------------------------------------
static UniformSubcycleTrial evaluate_uniform_current_map_at_substeps(
        const std::vector<Vec3>& current_guess, const State& st,
        const Grid& g, const Config& cfg, const VecField& grad_phi_n,
        const WaveSpectralState& wave_cache, const Quadrature& quad,
        int requested_substeps) {
    const std::size_t np = st.x.size();
    const std::size_t nsub = static_cast<std::size_t>(requested_substeps);
    const std::size_t total_substeps = np * nsub;
    const Real h = cfg.dt / static_cast<Real>(nsub);
    UniformSubcycleTrial tr;
    tr.substeps_used = requested_substeps;
    tr.total_particle_substeps = total_substeps;
    tr.refined_particles = requested_substeps > 1 ? np : 0;
    tr.subcycle_offset.resize(np + 1);
    for (std::size_t p = 0; p <= np; ++p) {
        tr.subcycle_offset[p] = p * nsub;
    }
    tr.current_guess = current_guess;
    tr.J_field = node_vectors_to_vecfield(current_guess, g);
    project_nyquist(tr.J_field,g);
    tr.rho_end = continuity_charge_update(st.rho, tr.J_field, cfg.dt, g);
    tr.rho_mid.assign(g.N, 0.0);
    #pragma omp parallel for schedule(static)
    for (int id = 0; id < g.N; ++id) {
        tr.rho_mid[id] = 0.5 * (st.rho[id] + tr.rho_end[id]);
    }
    cn_potential_update(st, tr.rho_mid, tr.J_field, cfg.dt, g,
                        cfg.kappa, cfg.sigma1, cfg.sigma2,
                        tr.phi_end, tr.psi_end, tr.A_end, tr.U_end,
                        &wave_cache);
    const VecField grad_phi_end = spectral_gradient_scalar(tr.phi_end, g);
    VecField grad_phi_mid = zero_vecfield(g.N);
    for (int d = 0; d < 3; ++d) {
        #pragma omp parallel for schedule(static)
        for (int id = 0; id < g.N; ++id) {
            grad_phi_mid[d][id] = 0.5 * (grad_phi_n[d][id] + grad_phi_end[d][id]);
        }
    }

    tr.sub_x0.assign(total_substeps, Vec3{0.0, 0.0, 0.0});
    tr.sub_dx.assign(total_substeps, Vec3{0.0, 0.0, 0.0});
    tr.sub_vbar.assign(total_substeps, Vec3{0.0, 0.0, 0.0});
    tr.x_end.assign(np, Vec3{0.0, 0.0, 0.0});
    tr.P_end.assign(np, Vec3{0.0, 0.0, 0.0});
    tr.p_end.assign(np, Vec3{0.0, 0.0, 0.0});
    tr.v_next.assign(np, Vec3{0.0, 0.0, 0.0});
    tr.vbar_macro.assign(np, Vec3{0.0, 0.0, 0.0});
    std::vector<Real> q_weighted(total_substeps, 0.0);
    std::size_t failed_particles = 0;
    long long local_iterations_total = 0;
    int local_iterations_max = 0;
    Real worst_local_residual = 0.0;

    #pragma omp parallel for schedule(static) \
        reduction(+:failed_particles,local_iterations_total) \
        reduction(max:local_iterations_max,worst_local_residual)
    for (std::size_t p = 0; p < np; ++p) {
        Vec3 xpos = st.x[p];
        Vec3 Pcur = st.P[p];
        Vec3 vcur = st.v[p];
        Vec3 vmacro{0.0, 0.0, 0.0};
        bool particle_ok = true;
        for (std::size_t s = 0; s < nsub; ++s) {
            const std::size_t j = p * nsub + s;
            const Real theta0 = static_cast<Real>(s) / static_cast<Real>(nsub);
            const Real theta1 = static_cast<Real>(s + 1) / static_cast<Real>(nsub);
            Vec3 vbar;
            LocalParticleStep mapped;
            LocalParticleSolveReport report;
            const bool converged = solve_local_particle_substep(
                vcur, xpos, Pcur, st.q[p], st.m[p], h, theta0, theta1,
                st, tr.A_end, grad_phi_mid, g, cfg, quad,
                vbar, mapped, report);
            local_iterations_total += report.iterations;
            local_iterations_max = std::max(local_iterations_max, report.iterations);
            worst_local_residual = std::max(
                worst_local_residual, report.final_scaled_residual);
            if (!converged) {
                particle_ok = false;
                break;
            }
            tr.sub_x0[j] = xpos;
            tr.sub_vbar[j] = vbar;
            tr.sub_dx[j] = mul3(h, vbar);
            q_weighted[j] = st.q[p] / static_cast<Real>(nsub);
            xpos = wrap_pos(add3(xpos, tr.sub_dx[j]), g.L);
            Pcur = mapped.P_end;
            vcur = mapped.v_end;
            vmacro = add3(vmacro, mul3(1.0 / static_cast<Real>(nsub), vbar));
        }
        if (!particle_ok) {
            ++failed_particles;
            continue;
        }
        tr.x_end[p] = xpos;
        tr.P_end[p] = Pcur;
        tr.p_end[p] = sub3(Pcur, mul3(st.q[p], gather_vector_at(
            tr.A_end, xpos, g, cfg.spline_order)));
        tr.v_next[p] = vcur;
        tr.vbar_macro[p] = vmacro;
    }

    tr.local_iterations_max = local_iterations_max;
    tr.local_iterations_mean = total_substeps
        ? static_cast<Real>(local_iterations_total) / static_cast<Real>(total_substeps)
        : 0.0;
    if (failed_particles > 0) {
        std::ostringstream msg;
        msg << "uniform h=dt/" << requested_substeps
            << " local particle solve failed for " << failed_particles
            << " particles (worst scaled residual=" << worst_local_residual
            << ", local_max_iter=" << cfg.uniform_local_max_iter << ")";
        throw NonlinearSolveFailure(msg.str());
    }

    // q/nsub supplies h/dt, yielding the macro-time-averaged current of all
    // piecewise paths.
    tr.J_map = path_average_deposit_current(
        tr.sub_x0, tr.sub_dx, tr.sub_vbar, q_weighted, g,
        cfg.spline_order, quad, cfg.split_orbit_at_knots, &tr.path_stats);
    tr.current_map = vecfield_to_node_vectors(tr.J_map, g);
    return tr;
}

// ---------------------------------------------------------------------------
// Evaluate the particle-enslaved current map for a fixed per-particle substep layout
//
// Inputs:
//   current_guess : const std::vector<Vec3>&; trial mesh current, stored as node vectors
//   st : const State&; accepted particle and mesh state
//   g : const Grid&; periodic grid geometry and Fourier symbols
//   cfg : const Config&; simulation settings and solver tolerances
//   grad_phi_n : const VecField&; old spectral scalar-potential gradient
//   wave_cache : const WaveSpectralState&; cached old-state Fourier transforms
//   quad : const Quadrature&; Gauss nodes and weights on [0,1]
//   particle_substeps : const std::vector<int>&; per-particle local substep counts
//   map_evaluation : std::size_t; map evaluation
//
// Output:
//   UniformSubcycleTrial: evaluate the particle-enslaved current map for a fixed per-particle substep layout.
//
// Dependencies:
//   add3, cn_potential_update, continuity_charge_update, gather_vector_at, mul3,
//   node_vectors_to_vecfield, openmp_max_threads, openmp_thread_num,
//   path_average_deposit_current, project_nyquist, solve_local_particle_substep,
//   spectral_gradient_scalar, sub3, vecfield_to_node_vectors, wrap_pos, zero_vecfield.
//   C++17 standard library (headers listed at the top of this file).
//   Optional OpenMP; serial loop fallback when OpenMP is disabled.
// ---------------------------------------------------------------------------
static UniformSubcycleTrial evaluate_selective_current_map_at_layout(
        const std::vector<Vec3>& current_guess, const State& st,
        const Grid& g, const Config& cfg, const VecField& grad_phi_n,
        const WaveSpectralState& wave_cache, const Quadrature& quad,
        const std::vector<int>& particle_substeps,
        std::size_t map_evaluation) {
    const std::size_t np = st.x.size();
    if (particle_substeps.size() != np) {
        throw std::runtime_error("selective particle layout has the wrong size");
    }

    UniformSubcycleTrial tr;
    tr.current_guess = current_guess;
    tr.subcycle_offset.assign(np + 1, 0);
    std::vector<std::size_t> active_particles;
    active_particles.reserve(np / 16 + 1);
    for (std::size_t p = 0; p < np; ++p) {
        const int nsub = particle_substeps[p];
        if (nsub < 1 || (nsub & (nsub - 1)) != 0 ||
            nsub > cfg.selective_particle_max_substeps) {
            throw std::runtime_error("invalid selective particle substep level");
        }
        tr.substeps_used = std::max(tr.substeps_used, nsub);
        tr.subcycle_offset[p + 1] =
            tr.subcycle_offset[p] + static_cast<std::size_t>(nsub);
        if (nsub > 1) active_particles.push_back(p);
    }
    tr.total_particle_substeps = tr.subcycle_offset.back();
    tr.refined_particles = active_particles.size();

    tr.J_field = node_vectors_to_vecfield(current_guess, g);
    project_nyquist(tr.J_field,g);
    tr.rho_end = continuity_charge_update(st.rho, tr.J_field, cfg.dt, g);
    tr.rho_mid.assign(g.N, 0.0);
    #pragma omp parallel for schedule(static)
    for (int id = 0; id < g.N; ++id) {
        tr.rho_mid[id] = 0.5 * (st.rho[id] + tr.rho_end[id]);
    }
    cn_potential_update(st, tr.rho_mid, tr.J_field, cfg.dt, g,
                        cfg.kappa, cfg.sigma1, cfg.sigma2,
                        tr.phi_end, tr.psi_end, tr.A_end, tr.U_end,
                        &wave_cache);
    const VecField grad_phi_end = spectral_gradient_scalar(tr.phi_end, g);
    VecField grad_phi_mid = zero_vecfield(g.N);
    for (int d = 0; d < 3; ++d) {
        #pragma omp parallel for schedule(static)
        for (int id = 0; id < g.N; ++id) {
            grad_phi_mid[d][id] =
                0.5 * (grad_phi_n[d][id] + grad_phi_end[d][id]);
        }
    }

    const std::size_t total_substeps = tr.total_particle_substeps;
    tr.sub_x0.assign(total_substeps, Vec3{0.0, 0.0, 0.0});
    tr.sub_dx.assign(total_substeps, Vec3{0.0, 0.0, 0.0});
    tr.sub_vbar.assign(total_substeps, Vec3{0.0, 0.0, 0.0});
    tr.x_end.assign(np, Vec3{0.0, 0.0, 0.0});
    tr.P_end.assign(np, Vec3{0.0, 0.0, 0.0});
    tr.p_end.assign(np, Vec3{0.0, 0.0, 0.0});
    tr.v_next.assign(np, Vec3{0.0, 0.0, 0.0});
    tr.vbar_macro.assign(np, Vec3{0.0, 0.0, 0.0});
    std::vector<Real> q_weighted(total_substeps, 0.0);
    std::vector<long long> local_iterations(np, 0);
    std::vector<int> local_iterations_maximum(np, 0);
    std::vector<Real> final_residual_maximum(np, 0.0);
    std::vector<std::vector<FailedParticleDiagnostic>> thread_failures(
        static_cast<std::size_t>(std::max(1, openmp_max_threads())));

    auto push_particle = [&](std::size_t p) {
        const std::size_t begin = tr.subcycle_offset[p];
        const std::size_t nsub = static_cast<std::size_t>(particle_substeps[p]);
        const Real h = cfg.dt / static_cast<Real>(nsub);
        Vec3 xpos = st.x[p];
        Vec3 Pcur = st.P[p];
        Vec3 vcur = st.v[p];
        Vec3 vmacro{0.0, 0.0, 0.0};
        for (std::size_t s = 0; s < nsub; ++s) {
            const std::size_t j = begin + s;
            const Real theta0 = static_cast<Real>(s) / static_cast<Real>(nsub);
            const Real theta1 = static_cast<Real>(s + 1) / static_cast<Real>(nsub);
            Vec3 vbar;
            LocalParticleStep mapped;
            LocalParticleSolveReport report;
            const Vec3 initial_guess = vcur;
            const Vec3 substep_x0 = xpos;
            const Vec3 substep_P0 = Pcur;
            const bool converged = solve_local_particle_substep(
                vcur, xpos, Pcur, st.q[p], st.m[p], h, theta0, theta1,
                st, tr.A_end, grad_phi_mid, g, cfg, quad,
                vbar, mapped, report);
            local_iterations[p] += report.iterations;
            local_iterations_maximum[p] =
                std::max(local_iterations_maximum[p], report.iterations);
            final_residual_maximum[p] =
                std::max(final_residual_maximum[p], report.final_scaled_residual);
            if (!converged) {
                FailedParticleDiagnostic diagnostic;
                diagnostic.particle = p;
                diagnostic.map_evaluation = map_evaluation;
                diagnostic.assigned_substeps = static_cast<int>(nsub);
                diagnostic.failed_substep = static_cast<int>(s);
                diagnostic.theta0 = theta0;
                diagnostic.theta1 = theta1;
                diagnostic.h = h;
                diagnostic.local_iterations = report.iterations;
                diagnostic.reason = report.failure_reason;
                diagnostic.first_scaled_residual = report.first_scaled_residual;
                diagnostic.final_scaled_residual = report.final_scaled_residual;
                diagnostic.final_absolute_residual = report.final_absolute_residual;
                diagnostic.q = st.q[p];
                diagnostic.m = st.m[p];
                diagnostic.x0 = substep_x0;
                diagnostic.P0 = substep_P0;
                diagnostic.initial_guess = initial_guess;
                diagnostic.last_guess = report.last_guess;
                diagnostic.last_map = report.last_map;
                thread_failures[static_cast<std::size_t>(
                    openmp_thread_num())].push_back(std::move(diagnostic));
                return;
            }
            tr.sub_x0[j] = xpos;
            tr.sub_vbar[j] = vbar;
            tr.sub_dx[j] = mul3(h, vbar);
            q_weighted[j] = st.q[p] / static_cast<Real>(nsub);
            xpos = wrap_pos(add3(xpos, tr.sub_dx[j]), g.L);
            Pcur = mapped.P_end;
            vcur = mapped.v_end;
            vmacro = add3(vmacro, mul3(1.0 / static_cast<Real>(nsub), vbar));
        }
        tr.x_end[p] = xpos;
        tr.P_end[p] = Pcur;
        tr.p_end[p] = sub3(Pcur, mul3(st.q[p], gather_vector_at(
            tr.A_end, xpos, g, cfg.spline_order)));
        tr.v_next[p] = vcur;
        tr.vbar_macro[p] = vmacro;
    };

    // The large level-one population has uniform work and uses static
    // scheduling.  Only the compact refined list uses dynamic scheduling to
    // balance different substep counts without penalizing the ordinary pass.
    #pragma omp parallel for schedule(static)
    for (std::size_t p = 0; p < np; ++p) {
        if (particle_substeps[p] == 1) push_particle(p);
    }
    [[maybe_unused]] const int dynamic_chunk =
        cfg.selective_particle_dynamic_chunk;
    #pragma omp parallel for schedule(dynamic, dynamic_chunk)
    for (std::size_t i = 0; i < active_particles.size(); ++i) {
        push_particle(active_particles[i]);
    }

    std::vector<FailedParticleDiagnostic> failed_diagnostics;
    long long local_iterations_total = 0;
    Real worst_local_residual = 0.0;
    for (std::size_t p = 0; p < np; ++p) {
        local_iterations_total += local_iterations[p];
        tr.local_iterations_max =
            std::max(tr.local_iterations_max, local_iterations_maximum[p]);
        worst_local_residual =
            std::max(worst_local_residual, final_residual_maximum[p]);
    }
    for (auto& local : thread_failures) {
        failed_diagnostics.insert(
            failed_diagnostics.end(),
            std::make_move_iterator(local.begin()),
            std::make_move_iterator(local.end()));
    }
    std::sort(failed_diagnostics.begin(), failed_diagnostics.end(),
        [](const FailedParticleDiagnostic& a,
           const FailedParticleDiagnostic& b) {
            return a.particle < b.particle;
        });
    tr.local_iterations_mean = total_substeps
        ? static_cast<Real>(local_iterations_total) /
              static_cast<Real>(total_substeps)
        : 0.0;

    tr.local_scaled_residual_max = worst_local_residual;
    if (!failed_diagnostics.empty()) {
        std::ostringstream msg;
        msg << "selective local particle solve failed for "
            << failed_diagnostics.size() << " particles at their fixed levels"
            << " (max assigned substeps=" << tr.substeps_used
            << ", worst scaled residual=" << worst_local_residual
            << ", local_max_iter=" << cfg.uniform_local_max_iter << ")";
        throw SelectiveParticleRefinementNeeded(
            msg.str(), std::move(failed_diagnostics));
    }

    // q/nsub supplies h/dt independently for every particle, so mixed
    // particle levels still produce one macro-time-averaged current.
    tr.J_map = path_average_deposit_current(
        tr.sub_x0, tr.sub_dx, tr.sub_vbar, q_weighted, g,
        cfg.spline_order, quad, cfg.split_orbit_at_knots, &tr.path_stats);
    tr.current_map = vecfield_to_node_vectors(tr.J_map, g);
    return tr;
}

// ---------------------------------------------------------------------------
// Evaluate uniform current map
//
// NOT NEEDED FOR MANUSCRIPT RUNS: retained optional/legacy branch.
//
// Inputs:
//   current_guess : const std::vector<Vec3>&; trial mesh current, stored as node vectors
//   st : const State&; accepted particle and mesh state
//   g : const Grid&; periodic grid geometry and Fourier symbols
//   cfg : const Config&; simulation settings and solver tolerances
//   grad_phi_n : const VecField&; old spectral scalar-potential gradient
//   wave_cache : const WaveSpectralState&; cached old-state Fourier transforms
//   quad : const Quadrature&; Gauss nodes and weights on [0,1]
//   active_substeps : int&; current uniform resolution, updated on refinement
//
// Output:
//   UniformSubcycleTrial: evaluate uniform current map.
//
// Dependencies:
//   evaluate_uniform_current_map_at_substeps.
//   C++17 standard library (headers listed at the top of this file).
// ---------------------------------------------------------------------------
static UniformSubcycleTrial evaluate_uniform_current_map(
        const std::vector<Vec3>& current_guess, const State& st,
        const Grid& g, const Config& cfg, const VecField& grad_phi_n,
        const WaveSpectralState& wave_cache, const Quadrature& quad,
        int& active_substeps) {
    int nsub = active_substeps;
    std::string last_failure;
    for (;;) {
        try {
            UniformSubcycleTrial tr = evaluate_uniform_current_map_at_substeps(
                current_guess, st, g, cfg, grad_phi_n, wave_cache, quad, nsub);
            active_substeps = nsub;
            return tr;
        } catch (const NonlinearSolveFailure& e) {
            last_failure = e.what();
            if (nsub >= cfg.uniform_max_substeps) break;
            // During the scheduled easy phase the first pass may use one
            // macro-sized particle step.  If that assumption unexpectedly
            // fails, jump directly to the normal subcycling resolution before
            // applying the usual 10->20 refinement.
            nsub = (nsub < cfg.uniform_substeps)
                ? cfg.uniform_substeps
                : std::min(cfg.uniform_max_substeps, 2 * nsub);
        }
    }
    std::ostringstream msg;
    msg << last_failure << "; local refinement exhausted at h=dt/"
        << nsub << " while macro dt=" << cfg.dt << " remained fixed";
    throw NonlinearSolveFailure(msg.str());
}

struct NonlinearHistoryEntry {
    int iter = 0;
    Real rms_residual = std::numeric_limits<Real>::infinity();
    Real max_particle_residual = std::numeric_limits<Real>::infinity();
    std::string update_method = "none";
};

struct NonlinearSolveInfo {
    bool converged = false;
    bool hit_max_iter = false;
    int iterations = 0;
    Real final_rms_residual = std::numeric_limits<Real>::infinity();
    Real final_max_particle_residual = std::numeric_limits<Real>::infinity();
    Real final_absolute_rms_residual = std::numeric_limits<Real>::infinity();
    int anderson_restarts = 0;
    int anderson_steps = 0;
    int picard_fallback_steps = 0;
    // SG added: diagnostics used by the adaptive dt hysteresis controller.
    Real observed_convergence_rate = std::numeric_limits<Real>::infinity();
    Real estimated_map_contraction = std::numeric_limits<Real>::infinity();
    int backtracked_steps = 0;
    Real minimum_update_alpha = 1.0;
    std::string convergence_method = "none";
    int local_particle_iterations_max = 0;
    Real local_particle_iterations_mean = 0.0;
    int uniform_substeps_used = 0;
    bool selective_fallback_used = false;
    int direct_attempt_iterations = 0;
    int selective_refinement_restarts = 0;
    std::size_t selective_refined_particles = 0;
    std::size_t selective_total_particle_substeps = 0;
    Real selective_work_ratio = 1.0;
    std::array<std::size_t, 7> selective_level_counts{};
};

// SG added: geometric average residual reduction over one nonlinear solve.
// ---------------------------------------------------------------------------
// Observed nonlinear convergence rate
//
// Inputs:
//   hist : const std::vector<NonlinearHistoryEntry>&; outer residual/update history, overwritten
//   final_residual : Real; final residual
//
// Output:
//   Real: observed nonlinear convergence rate.
//
// Dependencies:
//   C++17 standard library (headers listed at the top of this file).
// ---------------------------------------------------------------------------
static Real observed_nonlinear_convergence_rate(
    const std::vector<NonlinearHistoryEntry>& hist, Real final_residual) {
    if (hist.size() < 2 || !(hist.front().rms_residual > 0.0) ||
        !(final_residual >= 0.0) || !std::isfinite(final_residual)) {
        return std::numeric_limits<Real>::infinity();
    }
    if (final_residual == 0.0) return 0.0;
    const Real ratio = final_residual / hist.front().rms_residual;
    if (!(ratio > 0.0) || !std::isfinite(ratio)) {
        return std::numeric_limits<Real>::infinity();
    }
    return std::pow(ratio, 1.0 / static_cast<Real>(hist.size() - 1));
}

struct NonlinearResidualNorms {
    Real rms_relative = std::numeric_limits<Real>::infinity();
    Real max_particle_relative = std::numeric_limits<Real>::infinity();
    Real rms_absolute = std::numeric_limits<Real>::infinity();
    // SG added: values <= 1 satisfy atol + rtol * velocity_scale.
    Real rms_scaled = std::numeric_limits<Real>::infinity();
    Real max_particle_scaled = std::numeric_limits<Real>::infinity();
};

// ---------------------------------------------------------------------------
// Nonlinear residual norms
//
// Inputs:
//   map_value : const std::vector<Vec3>&; map value
//   guess : const std::vector<Vec3>&; guess
//   kappa : Real; nondimensional light speed
//   rtol : Real; rtol
//   atol : Real; atol
//
// Output:
//   NonlinearResidualNorms: nonlinear residual norms.
//
// Dependencies:
//   norm3, rms_vec_particles, sub3.
//   C++17 standard library (headers listed at the top of this file).
//   Optional OpenMP; serial loop fallback when OpenMP is disabled.
// ---------------------------------------------------------------------------
static NonlinearResidualNorms nonlinear_residual_norms(const std::vector<Vec3>& map_value,
                                                        const std::vector<Vec3>& guess,
                                                        Real kappa, Real rtol, Real atol) {
    if (map_value.size() != guess.size()) throw std::runtime_error("nonlinear residual size mismatch");
    std::vector<Vec3> residual(guess.size());
    NonlinearResidualNorms norms;
    Real max_particle_relative = 0.0;
    // SG added: maximum mixed residual over all particles.
    Real max_particle_scaled = 0.0;
    const Real particle_floor = 64.0 * std::numeric_limits<Real>::epsilon() * kappa;
    // Each particle residual is independent. The long-double RMS accumulation
    // below remains serial so its floating-point order matches the reference.
    #pragma omp parallel for reduction(max:max_particle_relative,max_particle_scaled) schedule(static)
    for (std::size_t p = 0; p < guess.size(); ++p) {
        residual[p] = sub3(map_value[p], guess[p]);
        const Real residual_norm = norm3(residual[p]);
        const Real velocity_scale = 0.5 * (norm3(map_value[p]) + norm3(guess[p]));
        const Real relative_scale = std::max(particle_floor, velocity_scale);
        max_particle_relative = std::max(max_particle_relative, residual_norm / relative_scale);

        // SG added: use an absolute floor plus a relative velocity-scaled term.
        const Real mixed_scale = atol + rtol * velocity_scale;
        max_particle_scaled = std::max(max_particle_scaled, residual_norm / mixed_scale);
    }
    norms.max_particle_relative = max_particle_relative;
    norms.max_particle_scaled = max_particle_scaled; // SG added
    norms.rms_absolute = rms_vec_particles(residual);
    const Real map_rms = rms_vec_particles(map_value);
    const Real guess_rms = rms_vec_particles(guess);
    const Real velocity_rms_scale = 0.5 * (map_rms + guess_rms);
    norms.rms_relative = norms.rms_absolute / (velocity_rms_scale + particle_floor);
    // SG added: mixed RMS residual; convergence requires this value <= 1.
    norms.rms_scaled = norms.rms_absolute / (atol + rtol * velocity_rms_scale);
    return norms;
}

// ---------------------------------------------------------------------------
// Physical velocity vector
//
// NOT NEEDED: uncalled legacy/reference helper; retained for provenance.
//
// Inputs:
//   v : const std::vector<Vec3>&; v
//   kappa : Real; nondimensional light speed
//
// Output:
//   bool: success/predicate result; non-const reference arguments carry any results.
//
// Dependencies:
//   norm3.
//   C++17 standard library (headers listed at the top of this file).
//   Optional OpenMP; serial loop fallback when OpenMP is disabled.
// ---------------------------------------------------------------------------
[[maybe_unused]] static bool physical_velocity_vector(const std::vector<Vec3>& v, Real kappa) {
    int physical = 1;
    #pragma omp parallel for reduction(&:physical) schedule(static)
    for (std::size_t p = 0; p < v.size(); ++p) {
        const Vec3& vp = v[p];
        physical &= std::isfinite(vp[0]) && std::isfinite(vp[1]) && std::isfinite(vp[2]) &&
                    (norm3(vp) < kappa);
    }
    return physical != 0;
}

// ---------------------------------------------------------------------------
// Enforce initial velocity bound
//
// NOT NEEDED: uncalled legacy/reference helper; retained for provenance.
//
// Inputs:
//   v : std::vector<Vec3>&; v
//   kappa : Real; nondimensional light speed
//
// Output:
//   None; updates the non-const reference arguments in place.
//
// Dependencies:
//   mul3, norm3.
//   C++17 standard library (headers listed at the top of this file).
// ---------------------------------------------------------------------------
[[maybe_unused]] static void enforce_initial_velocity_bound(std::vector<Vec3>& v, Real kappa) {
    const Real bound = kappa * (1.0 - 64.0 * std::numeric_limits<Real>::epsilon());
    // Keep this serial because error handling must not throw across an OpenMP region.
    for (Vec3& vp : v) {
        const Real speed = norm3(vp);
        if (!std::isfinite(speed)) throw std::runtime_error("nonfinite nonlinear velocity predictor");
        if (speed >= bound) vp = mul3(bound / std::max(speed, std::numeric_limits<Real>::min()), vp);
    }
}

// ---------------------------------------------------------------------------
// Particle vector dot
//
// Inputs:
//   a : const std::vector<Vec3>&; a
//   b : const std::vector<Vec3>&; b
//
// Output:
//   long double: particle vector dot.
//
// Dependencies:
//   C++17 standard library (headers listed at the top of this file).
// ---------------------------------------------------------------------------
static long double particle_vector_dot(const std::vector<Vec3>& a, const std::vector<Vec3>& b) {
    if (a.size() != b.size()) throw std::runtime_error("Anderson vector size mismatch");
    // Deliberately serial: Anderson coefficients then follow the exact serial
    // floating-point accumulation order.
    long double sum = 0.0L;
    for (std::size_t p = 0; p < a.size(); ++p) {
        for (int d = 0; d < 3; ++d) {
            sum += static_cast<long double>(a[p][d]) * static_cast<long double>(b[p][d]);
        }
    }
    return sum;
}

// ---------------------------------------------------------------------------
// Solve the small dense Anderson system by pivoted elimination
//
// Inputs:
//   A : std::vector<std::vector<Real>>; nodal vector potential
//   b : std::vector<Real>; b
//   x : std::vector<Real>&; x
//
// Output:
//   bool: success/predicate result; non-const reference arguments carry any results.
//
// Dependencies:
//   C++17 standard library (headers listed at the top of this file).
// ---------------------------------------------------------------------------
static bool solve_small_dense_system(std::vector<std::vector<Real>> A, std::vector<Real> b,
                                     std::vector<Real>& x) {
    const int n = static_cast<int>(b.size());
    Real matrix_scale = 0.0;
    for (const auto& row : A) for (Real value : row) matrix_scale = std::max(matrix_scale, std::abs(value));
    if (!(matrix_scale > 0.0) || !std::isfinite(matrix_scale)) return false;
    Real largest_pivot = 0.0;
    Real smallest_pivot = std::numeric_limits<Real>::infinity();
    for (int col = 0; col < n; ++col) {
        int pivot_row = col;
        for (int row = col + 1; row < n; ++row) {
            if (std::abs(A[row][col]) > std::abs(A[pivot_row][col])) pivot_row = row;
        }
        const Real pivot = std::abs(A[pivot_row][col]);
        if (!std::isfinite(pivot) || pivot <= 100.0 * std::numeric_limits<Real>::epsilon() * matrix_scale) return false;
        largest_pivot = std::max(largest_pivot, pivot);
        smallest_pivot = std::min(smallest_pivot, pivot);
        if (pivot_row != col) {
            std::swap(A[pivot_row], A[col]);
            std::swap(b[pivot_row], b[col]);
        }
        for (int row = col + 1; row < n; ++row) {
            const Real factor = A[row][col] / A[col][col];
            for (int j = col; j < n; ++j) A[row][j] -= factor * A[col][j];
            b[row] -= factor * b[col];
        }
    }
    if (smallest_pivot <= 1.0e-12 * largest_pivot) return false;
    x.assign(static_cast<std::size_t>(n), 0.0);
    for (int row = n - 1; row >= 0; --row) {
        Real rhs = b[row];
        for (int j = row + 1; j < n; ++j) rhs -= A[row][j] * x[static_cast<std::size_t>(j)];
        x[static_cast<std::size_t>(row)] = rhs / A[row][row];
        if (!std::isfinite(x[static_cast<std::size_t>(row)])) return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Form a regularized limited-memory type-II Anderson candidate
//
// Inputs:
//   x_history : const std::vector<std::vector<Vec3>>&; x history
//   r_history : const std::vector<std::vector<Vec3>>&; r history
//   memory : int; memory
//   damping : Real; damping
//   regularization : Real; regularization
//   candidate : std::vector<Vec3>&; candidate
//
// Output:
//   bool: success/predicate result; non-const reference arguments carry any results.
//
// Dependencies:
//   add3, mul3, particle_vector_dot, solve_small_dense_system, sub3.
//   C++17 standard library (headers listed at the top of this file).
//   Optional OpenMP; serial loop fallback when OpenMP is disabled.
// ---------------------------------------------------------------------------
static bool anderson_type2_candidate(const std::vector<std::vector<Vec3>>& x_history,
                                     const std::vector<std::vector<Vec3>>& r_history,
                                     int memory, Real damping, Real regularization,
                                     std::vector<Vec3>& candidate) {
    // Damped type-II Anderson acceleration for r_k=G(x_k)-x_k.  The columns
    // contain only orbit velocities; fields and particle/mesh physics remain
    // inside the unchanged nonlinear map evaluation.
    const int available = static_cast<int>(x_history.size()) - 1;
    const int m = std::min(memory, available);
    if (m <= 0 || x_history.size() != r_history.size()) return false;
    const int first = static_cast<int>(x_history.size()) - m - 1;
    std::vector<std::vector<Vec3>> delta_x(static_cast<std::size_t>(m));
    std::vector<std::vector<Vec3>> delta_r(static_cast<std::size_t>(m));
    for (int j = 0; j < m; ++j) {
        const std::size_t a = static_cast<std::size_t>(first + j);
        const std::size_t b = a + 1;
        delta_x[static_cast<std::size_t>(j)].resize(x_history.back().size());
        delta_r[static_cast<std::size_t>(j)].resize(x_history.back().size());
        #pragma omp parallel for schedule(static)
        for (std::size_t p = 0; p < x_history.back().size(); ++p) {
            delta_x[static_cast<std::size_t>(j)][p] = sub3(x_history[b][p], x_history[a][p]);
            delta_r[static_cast<std::size_t>(j)][p] = sub3(r_history[b][p], r_history[a][p]);
        }
    }

    std::vector<std::vector<Real>> gram(static_cast<std::size_t>(m), std::vector<Real>(static_cast<std::size_t>(m), 0.0));
    std::vector<Real> rhs(static_cast<std::size_t>(m), 0.0);
    Real trace = 0.0;
    for (int i = 0; i < m; ++i) {
        rhs[static_cast<std::size_t>(i)] = static_cast<Real>(particle_vector_dot(delta_r[static_cast<std::size_t>(i)], r_history.back()));
        for (int j = 0; j <= i; ++j) {
            const Real value = static_cast<Real>(particle_vector_dot(delta_r[static_cast<std::size_t>(i)], delta_r[static_cast<std::size_t>(j)]));
            gram[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] = value;
            gram[static_cast<std::size_t>(j)][static_cast<std::size_t>(i)] = value;
        }
        trace += gram[static_cast<std::size_t>(i)][static_cast<std::size_t>(i)];
    }
    if (!(trace > 0.0) || !std::isfinite(trace)) return false;
    // Test the unregularized normal matrix first. Tikhonov regularization
    // stabilizes a sound multisecant system; it does not conceal singular data.
    std::vector<Real> unregularized_solution;
    if (!solve_small_dense_system(gram, rhs, unregularized_solution)) return false;
    const Real regularization_scale = trace / static_cast<Real>(m);
    for (int i = 0; i < m; ++i) {
        gram[static_cast<std::size_t>(i)][static_cast<std::size_t>(i)] += regularization * regularization_scale;
    }
    std::vector<Real> gamma;
    if (!solve_small_dense_system(std::move(gram), std::move(rhs), gamma)) return false;

    candidate = x_history.back();
    #pragma omp parallel for schedule(static)
    for (std::size_t p = 0; p < candidate.size(); ++p) {
        candidate[p] = add3(candidate[p], mul3(damping, r_history.back()[p]));
        for (int j = 0; j < m; ++j) {
            const Vec3 correction = add3(delta_x[static_cast<std::size_t>(j)][p],
                                         mul3(damping, delta_r[static_cast<std::size_t>(j)][p]));
            candidate[p] = sub3(candidate[p], mul3(gamma[static_cast<std::size_t>(j)], correction));
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// Damped picard candidate
//
// Inputs:
//   guess : const std::vector<Vec3>&; guess
//   residual : const std::vector<Vec3>&; residual
//   damping : Real; damping
//
// Output:
//   std::vector<Vec3>: damped picard candidate.
//
// Dependencies:
//   add3, mul3.
//   C++17 standard library (headers listed at the top of this file).
//   Optional OpenMP; serial loop fallback when OpenMP is disabled.
// ---------------------------------------------------------------------------
static std::vector<Vec3> damped_picard_candidate(const std::vector<Vec3>& guess,
                                                 const std::vector<Vec3>& residual,
                                                 Real damping) {
    std::vector<Vec3> candidate(guess.size());
    #pragma omp parallel for schedule(static)
    for (std::size_t p = 0; p < guess.size(); ++p) {
        candidate[p] = add3(guess[p], mul3(damping, residual[p]));
    }
    return candidate;
}

// ---------------------------------------------------------------------------
// Converge the outer mesh-current map with frozen-field inner particle solves
//
// Inputs:
//   st : const State&; accepted particle and mesh state
//   g : const Grid&; periodic grid geometry and Fourier symbols
//   cfg : const Config&; simulation settings and solver tolerances
//   quad : const Quadrature&; Gauss nodes and weights on [0,1]
//   hist : std::vector<NonlinearHistoryEntry>&; outer residual/update history, overwritten
//   solve_info_out : NonlinearSolveInfo*; optional output solver statistics pointer
//   time_n : Real; time at the start of the field step
//   selective_layout : const std::vector<int>*; optional fixed per-particle substep counts (one means full dt)
//   particle_history : std::ostream*; optional output stream for every inner-map evaluation
//
// Output:
//   State: new particle/field state; input state remains unchanged.
//
// Dependencies:
//   add3, anderson_type2_candidate, damped_picard_candidate,
//   evaluate_selective_current_map_at_layout, evaluate_uniform_current_map,
//   finite_node_vector, make_wave_spectral_state, mul3, nonlinear_residual_norms,
//   observed_nonlinear_convergence_rate, rms_vec_particles, spectral_gradient_scalar, sub3,
//   vecfield_to_node_vectors.
//   C++17 standard library (headers listed at the top of this file).
//   Optional OpenMP; serial loop fallback when OpenMP is disabled.
// ---------------------------------------------------------------------------
static State nonlinear_step_uniform_subcycled(
        const State& st, const Grid& g, const Config& cfg,
        const Quadrature& quad, std::vector<NonlinearHistoryEntry>& hist,
        NonlinearSolveInfo* solve_info_out, Real time_n,
        const std::vector<int>* selective_layout = nullptr,
        std::ostream* particle_history = nullptr) {
    const VecField grad_phi_n = spectral_gradient_scalar(st.phi, g);
    const WaveSpectralState wave_cache = make_wave_spectral_state(
        st.phi, st.psi, st.A, st.U, g);
    std::vector<Vec3> current_guess = vecfield_to_node_vectors(st.J, g);
    if (!finite_node_vector(current_guess)) {
        throw NonlinearSolveFailure("nonfinite initial macro-current predictor");
    }

    UniformSubcycleTrial tr;
    NonlinearSolveInfo sinfo;
    sinfo.selective_fallback_used = selective_layout != nullptr;
    hist.clear();
    std::vector<std::vector<Vec3>> x_history, r_history;
    Real previous_rms_residual = std::numeric_limits<Real>::infinity();
    std::string last_update_method =
        (cfg.nonlinear_solver == "picard") ? "picard" : "picard_fallback";
    bool apparent_convergence = false;
    bool have_prefetched_trial = false;
    // Once any outer evaluation needs finer local resolution, retain that
    // resolution for the rest of this macro solve.  This avoids repeating a
    // known-failing dt/10 pass and keeps the outer current map consistent.
    const Real switch_epsilon =
        128.0 * std::numeric_limits<Real>::epsilon() *
        std::max<Real>(1.0, std::abs(cfg.uniform_subcycling_start_time));
    const bool initial_easy_phase =
        !cfg.uniform_refine_on_failure &&
        cfg.uniform_subcycling_start_time > 0.0 &&
        time_n + switch_epsilon < cfg.uniform_subcycling_start_time;
    int active_substeps = cfg.uniform_refine_on_failure
        ? cfg.uniform_initial_substeps
        : (initial_easy_phase ? cfg.uniform_initial_substeps : cfg.uniform_substeps);
    std::size_t map_evaluation = 0;
    auto evaluate_current_map = [&](const std::vector<Vec3>& guess) {
        ++map_evaluation;
        if (selective_layout) {
            try {
                auto trial = evaluate_selective_current_map_at_layout(
                    guess, st, g, cfg, grad_phi_n, wave_cache, quad,
                    *selective_layout, map_evaluation);
                if (particle_history) {
                    *particle_history << std::setprecision(17) << time_n << ","
                        << cfg.dt << "," << map_evaluation << ",1,0,"
                        << trial.local_scaled_residual_max << ","
                        << trial.local_iterations_mean << ","
                        << trial.local_iterations_max << "\n";
                    particle_history->flush();
                }
                return trial;
            } catch (const SelectiveParticleRefinementNeeded& e) {
                if (particle_history) {
                    Real worst = 0.0;
                    int iterations = 0;
                    for (const auto& d : e.failed_diagnostics) {
                        worst = std::isfinite(d.final_scaled_residual)
                            ? std::max(worst, d.final_scaled_residual)
                            : std::numeric_limits<Real>::infinity();
                        iterations = std::max(iterations, d.local_iterations);
                    }
                    *particle_history << std::setprecision(17) << time_n << ","
                        << cfg.dt << "," << map_evaluation << ",0,"
                        << e.failed_diagnostics.size() << "," << worst
                        << ",nan," << iterations << "\n";
                    particle_history->flush();
                }
                throw;
            }
        }
        return evaluate_uniform_current_map(
            guess, st, g, cfg, grad_phi_n, wave_cache, quad,
            active_substeps);
    };

    for (int it = 0; it < cfg.nonlinear_max_iter; ++it) {
        try {
            if (have_prefetched_trial) {
                have_prefetched_trial = false;
            } else {
                tr = evaluate_current_map(current_guess);
            }
        } catch (const NonlinearSolveFailure&) {
            sinfo.iterations = it + 1;
            if (solve_info_out) *solve_info_out = sinfo;
            throw;
        }
        if (!finite_node_vector(tr.current_map)) {
            sinfo.iterations = it + 1;
            if (solve_info_out) *solve_info_out = sinfo;
            throw NonlinearSolveFailure(
                "uniform particle map returned a nonfinite macro current");
        }

        std::vector<Vec3> residual(current_guess.size());
        #pragma omp parallel for schedule(static)
        for (std::size_t i = 0; i < current_guess.size(); ++i) {
            residual[i] = sub3(tr.current_map[i], current_guess[i]);
        }
        const NonlinearResidualNorms norms = nonlinear_residual_norms(
            tr.current_map, current_guess, cfg.kappa,
            cfg.nonlinear_rtol, cfg.nonlinear_atol);

        if (!x_history.empty()) {
            std::vector<Vec3> delta_x(current_guess.size());
            std::vector<Vec3> delta_G(current_guess.size());
            #pragma omp parallel for schedule(static)
            for (std::size_t i = 0; i < current_guess.size(); ++i) {
                delta_x[i] = sub3(current_guess[i], x_history.back()[i]);
                delta_G[i] = add3(delta_x[i], sub3(residual[i], r_history.back()[i]));
            }
            const Real dx_rms = rms_vec_particles(delta_x);
            if (dx_rms > 64.0 * std::numeric_limits<Real>::epsilon()) {
                sinfo.estimated_map_contraction =
                    rms_vec_particles(delta_G) / dx_rms;
            }
        }

        NonlinearHistoryEntry he;
        he.iter = it;
        he.rms_residual = norms.rms_relative;
        he.max_particle_residual = norms.max_particle_relative;
        if (norms.rms_scaled <= 1.0 && norms.max_particle_scaled <= 1.0) {
            apparent_convergence = true;
            he.update_method = last_update_method;
            hist.push_back(he);
            sinfo.iterations = it + 1;
            sinfo.convergence_method = last_update_method;
            break;
        }

        if (cfg.nonlinear_solver == "anderson_quasi_newton" &&
            std::isfinite(previous_rms_residual) &&
            norms.rms_relative > cfg.anderson_growth_limit * previous_rms_residual) {
            x_history.clear();
            r_history.clear();
            ++sinfo.anderson_restarts;
        }
        x_history.push_back(current_guess);
        r_history.push_back(residual);
        const std::size_t keep = static_cast<std::size_t>(cfg.anderson_memory + 1);
        if (x_history.size() > keep) {
            x_history.erase(x_history.begin());
            r_history.erase(r_history.begin());
        }

        std::vector<Vec3> candidate;
        bool used_anderson = false;
        if (cfg.nonlinear_solver == "anderson_quasi_newton" &&
            it >= cfg.anderson_start && x_history.size() >= 2) {
            used_anderson = anderson_type2_candidate(
                x_history, r_history, cfg.anderson_memory,
                cfg.anderson_damping, cfg.anderson_regularization, candidate);
            if (!used_anderson || !finite_node_vector(candidate)) {
                x_history.assign(1, current_guess);
                r_history.assign(1, residual);
                ++sinfo.anderson_restarts;
                used_anderson = false;
            }
        }

        const Real current_rms_merit = norms.rms_scaled;
        const Real current_max_merit = norms.max_particle_scaled;
        auto try_backtracked_update =
            [&](const std::vector<Vec3>& full_candidate,
                std::vector<Vec3>& accepted_candidate,
                UniformSubcycleTrial& accepted_trial,
                Real& accepted_alpha) -> bool {
                constexpr int max_backtracks = 6;
                constexpr Real sufficient_decrease = 1.0e-4;
                std::vector<Vec3> trial_guess(current_guess.size());
                for (int bt = 0; bt <= max_backtracks; ++bt) {
                    const Real alpha = std::ldexp(1.0, -bt);
                    #pragma omp parallel for schedule(static)
                    for (std::size_t i = 0; i < current_guess.size(); ++i) {
                        trial_guess[i] = add3(
                            current_guess[i],
                            mul3(alpha, sub3(full_candidate[i], current_guess[i])));
                    }
                    if (!finite_node_vector(trial_guess)) continue;
                    UniformSubcycleTrial trial_map;
                    try {
                        trial_map = evaluate_current_map(trial_guess);
                    } catch (const SelectiveParticleRefinementNeeded&) {
                        throw;
                    } catch (const NonlinearSolveFailure&) {
                        continue;
                    }
                    const NonlinearResidualNorms trial_norms = nonlinear_residual_norms(
                        trial_map.current_map, trial_guess, cfg.kappa,
                        cfg.nonlinear_rtol, cfg.nonlinear_atol);
                    const bool rms_decreased = trial_norms.rms_scaled <=
                        (1.0 - sufficient_decrease * alpha) * current_rms_merit;
                    const bool max_safeguarded = trial_norms.max_particle_scaled <=
                        cfg.anderson_growth_limit * current_max_merit;
                    if (rms_decreased && max_safeguarded) {
                        accepted_candidate = trial_guess;
                        accepted_trial = std::move(trial_map);
                        accepted_alpha = alpha;
                        return true;
                    }
                }
                return false;
            };

        bool accepted = false;
        bool accepted_as_anderson = false;
        Real accepted_alpha = 0.0;
        UniformSubcycleTrial accepted_trial;
        if (used_anderson) {
            accepted = try_backtracked_update(
                candidate, candidate, accepted_trial, accepted_alpha);
            accepted_as_anderson = accepted;
            if (!accepted) {
                x_history.assign(1, current_guess);
                r_history.assign(1, residual);
                ++sinfo.anderson_restarts;
            }
        }
        if (!accepted) {
            const Real damping = (cfg.nonlinear_solver == "picard")
                ? 1.0 : cfg.anderson_damping;
            const std::vector<Vec3> picard_candidate = damped_picard_candidate(
                current_guess, residual, damping);
            accepted = try_backtracked_update(
                picard_candidate, candidate, accepted_trial, accepted_alpha);
            accepted_as_anderson = false;
        }
        if (!accepted) {
            sinfo.iterations = it + 1;
            sinfo.final_rms_residual = norms.rms_relative;
            sinfo.final_max_particle_residual = norms.max_particle_relative;
            sinfo.final_absolute_rms_residual = norms.rms_absolute;
            hist.push_back(he);
            sinfo.observed_convergence_rate =
                observed_nonlinear_convergence_rate(hist, norms.rms_relative);
            if (solve_info_out) *solve_info_out = sinfo;
            throw NonlinearSolveFailure(
                "outer current Anderson/Picard backtracking failed at the prescribed macro dt");
        }

        if (accepted_as_anderson) {
            ++sinfo.anderson_steps;
            he.update_method = accepted_alpha < 1.0
                ? "current_anderson_backtracked" : "current_anderson";
        } else {
            ++sinfo.picard_fallback_steps;
            he.update_method = accepted_alpha < 1.0
                ? "current_picard_backtracked" : "current_picard";
        }
        if (accepted_alpha < 1.0) {
            ++sinfo.backtracked_steps;
            sinfo.minimum_update_alpha =
                std::min(sinfo.minimum_update_alpha, accepted_alpha);
        }
        hist.push_back(he);
        last_update_method = he.update_method;
        previous_rms_residual = norms.rms_relative;
        current_guess = std::move(candidate);
        tr = std::move(accepted_trial);
        have_prefetched_trial = true;
        sinfo.iterations = it + 1;
    }

    if (!apparent_convergence) {
        sinfo.hit_max_iter = true;
        if (!hist.empty()) {
            sinfo.final_rms_residual = hist.back().rms_residual;
            sinfo.final_max_particle_residual = hist.back().max_particle_residual;
            sinfo.observed_convergence_rate = observed_nonlinear_convergence_rate(
                hist, sinfo.final_rms_residual);
        }
        if (solve_info_out) *solve_info_out = sinfo;
        std::ostringstream msg;
        msg << "outer macro-current solve failed at dt=" << cfg.dt
            << " after " << sinfo.iterations << " iterations";
        throw NonlinearSolveFailure(msg.str());
    }

    tr = evaluate_current_map(current_guess);
    const NonlinearResidualNorms final_norms = nonlinear_residual_norms(
        tr.current_map, current_guess, cfg.kappa,
        cfg.nonlinear_rtol, cfg.nonlinear_atol);
    sinfo.final_rms_residual = final_norms.rms_relative;
    sinfo.final_max_particle_residual = final_norms.max_particle_relative;
    sinfo.final_absolute_rms_residual = final_norms.rms_absolute;
    sinfo.observed_convergence_rate = observed_nonlinear_convergence_rate(
        hist, final_norms.rms_relative);
    sinfo.local_particle_iterations_max = tr.local_iterations_max;
    sinfo.local_particle_iterations_mean = tr.local_iterations_mean;
    sinfo.uniform_substeps_used = tr.substeps_used;
    if (selective_layout) {
        sinfo.selective_refined_particles = tr.refined_particles;
        sinfo.selective_total_particle_substeps = tr.total_particle_substeps;
        sinfo.selective_work_ratio = st.x.empty() ? 0.0
            : static_cast<Real>(tr.total_particle_substeps) /
                  static_cast<Real>(st.x.size());
        for (int nsub : *selective_layout) {
            int index = 0;
            for (int value = nsub; value > 1; value /= 2) ++index;
            index = std::min<int>(index,
                static_cast<int>(sinfo.selective_level_counts.size()) - 1);
            ++sinfo.selective_level_counts[static_cast<std::size_t>(index)];
        }
        sinfo.convergence_method = "selective_" + sinfo.convergence_method;
    }
    sinfo.converged = finite_node_vector(tr.current_map) &&
        final_norms.rms_scaled <= 1.0 && final_norms.max_particle_scaled <= 1.0;
    if (solve_info_out) *solve_info_out = sinfo;
    if (!sinfo.converged) {
        throw NonlinearSolveFailure(
            "final outer macro-current consistency evaluation failed at the prescribed macro dt");
    }

    State out;
    out.x = std::move(tr.x_end);
    out.v = std::move(tr.v_next);
    out.v_prev = st.v;
    out.vbar_last = std::move(tr.vbar_macro);
    out.P = std::move(tr.P_end);
    out.q = st.q;
    out.m = st.m;
    out.phi = std::move(tr.phi_end);
    out.psi = std::move(tr.psi_end);
    out.A = std::move(tr.A_end);
    out.U = std::move(tr.U_end);
    out.rho = std::move(tr.rho_end);
    out.J = std::move(tr.J_map);
    out.dt_last = cfg.dt;
    out.uniform_substeps_active = tr.substeps_used;
    out.subcycle_offset_last = std::move(tr.subcycle_offset);
    out.subcycle_x0_last = std::move(tr.sub_x0);
    out.subcycle_vbar_last = std::move(tr.sub_vbar);
    return out;
}

// ---------------------------------------------------------------------------
// Write failed particle diagnostics
//
// Inputs:
//   out : std::ostream*; output stream or result populated in place (see type)
//   time_n : Real; time at the start of the field step
//   refinement_restart : int; refinement restart
//   cfg : const Config&; simulation settings and solver tolerances
//   diagnostics : const std::vector<FailedParticleDiagnostic>&; diagnostics
//   event_exhausted : bool; event exhausted
//
// Output:
//   None; writes records to the supplied stream/file.
//
// Dependencies:
//   C++17 standard library (headers listed at the top of this file).
// ---------------------------------------------------------------------------
static void write_failed_particle_diagnostics(
        std::ostream* out, Real time_n, int refinement_restart,
        const Config& cfg,
        const std::vector<FailedParticleDiagnostic>& diagnostics,
        bool event_exhausted) {
    if (!out || diagnostics.empty() || !cfg.selective_failure_diagnostics) return;

    std::vector<std::size_t> order(diagnostics.size());
    for (std::size_t i = 0; i < order.size(); ++i) order[i] = i;
    std::sort(order.begin(), order.end(),
        [&](std::size_t ia, std::size_t ib) {
            const Real a = diagnostics[ia].final_scaled_residual;
            const Real b = diagnostics[ib].final_scaled_residual;
            if (!std::isfinite(a)) return std::isfinite(b);
            if (!std::isfinite(b)) return false;
            return a > b;
        });

    std::vector<unsigned char> selected(diagnostics.size(), 0);
    const std::size_t limit = std::min<std::size_t>(
        diagnostics.size(),
        static_cast<std::size_t>(cfg.selective_failure_diagnostic_limit));
    for (std::size_t i = 0; i < limit; ++i) selected[order[i]] = 1;
    if (event_exhausted) {
        std::size_t worst_exhausted = diagnostics.size();
        for (std::size_t i = 0; i < diagnostics.size(); ++i) {
            if (diagnostics[i].assigned_substeps <
                cfg.selective_particle_max_substeps) continue;
            if (worst_exhausted == diagnostics.size()) {
                worst_exhausted = i;
                continue;
            }
            const Real candidate = diagnostics[i].final_scaled_residual;
            const Real current =
                diagnostics[worst_exhausted].final_scaled_residual;
            if ((!std::isfinite(candidate) && std::isfinite(current)) ||
                (std::isfinite(candidate) && std::isfinite(current) &&
                 candidate > current)) {
                worst_exhausted = i;
            }
        }
        if (worst_exhausted < diagnostics.size()) selected[worst_exhausted] = 1;
    }

    for (std::size_t i = 0; i < diagnostics.size(); ++i) {
        if (!selected[i]) continue;
        const FailedParticleDiagnostic& d = diagnostics[i];
        *out << std::setprecision(17)
             << time_n << "," << refinement_restart << ","
             << d.map_evaluation << "," << (event_exhausted ? 1 : 0) << ","
             << diagnostics.size() << "," << d.particle << ","
             << d.assigned_substeps << "," << d.failed_substep << ","
             << d.theta0 << "," << d.theta1 << "," << d.h << ","
             << d.local_iterations << ","
             << local_particle_failure_reason_name(d.reason) << ","
             << d.first_scaled_residual << ","
             << d.final_scaled_residual << ","
             << d.final_absolute_residual << ","
             << d.q << "," << d.m << ","
             << d.x0[0] << "," << d.x0[1] << "," << d.x0[2] << ","
             << d.P0[0] << "," << d.P0[1] << "," << d.P0[2] << ","
             << d.initial_guess[0] << "," << d.initial_guess[1] << ","
             << d.initial_guess[2] << ","
             << d.last_guess[0] << "," << d.last_guess[1] << ","
             << d.last_guess[2] << ","
             << d.last_map[0] << "," << d.last_map[1] << ","
             << d.last_map[2] << "\n";
    }
    out->flush();
}

// ---------------------------------------------------------------------------
// Retry a failed full-step solve with selective particle refinement
//
// Inputs:
//   st : const State&; accepted particle and mesh state
//   g : const Grid&; periodic grid geometry and Fourier symbols
//   cfg : const Config&; simulation settings and solver tolerances
//   quad : const Quadrature&; Gauss nodes and weights on [0,1]
//   hist : std::vector<NonlinearHistoryEntry>&; outer residual/update history, overwritten
//   solve_info_out : NonlinearSolveInfo*; optional output solver statistics pointer
//   time_n : Real; time at the start of the field step
//   direct_info : const NonlinearSolveInfo&; direct info
//   direct_failure : const std::string&; direct failure
//   failed_particle_diagnostics : std::ostream*; optional stream for failed-particle records
//   particle_history : std::ostream*; optional output stream for every inner-map evaluation
//
// Output:
//   State: new particle/field state; input state remains unchanged.
//
// Dependencies:
//   nonlinear_step_uniform_subcycled, write_failed_particle_diagnostics.
//   C++17 standard library (headers listed at the top of this file).
// ---------------------------------------------------------------------------
static State nonlinear_step_selective_particle_fallback(
        const State& st, const Grid& g, const Config& cfg,
        const Quadrature& quad, std::vector<NonlinearHistoryEntry>& hist,
        NonlinearSolveInfo* solve_info_out, Real time_n,
        const NonlinearSolveInfo& direct_info,
        const std::string& direct_failure,
        std::ostream* failed_particle_diagnostics = nullptr,
        std::ostream* particle_history = nullptr) {
    const std::size_t np = st.x.size();
    std::vector<int> particle_substeps(np, 1);
    int refinement_restarts = 0;

    for (;;) {
        NonlinearSolveInfo selective_info;
        try {
            State out = nonlinear_step_uniform_subcycled(
                st, g, cfg, quad, hist, &selective_info, time_n,
                &particle_substeps, particle_history);
            selective_info.selective_fallback_used = true;
            selective_info.direct_attempt_iterations = direct_info.iterations;
            selective_info.selective_refinement_restarts =
                refinement_restarts;
            if (solve_info_out) *solve_info_out = selective_info;
            return out;
        } catch (const SelectiveParticleRefinementNeeded& e) {
            selective_info.selective_fallback_used = true;
            selective_info.direct_attempt_iterations = direct_info.iterations;
            selective_info.selective_refinement_restarts =
                refinement_restarts;

            std::size_t exhausted = 0;
            const FailedParticleDiagnostic* worst_exhausted = nullptr;
            for (const FailedParticleDiagnostic& diagnostic :
                 e.failed_diagnostics) {
                if (particle_substeps[diagnostic.particle] >=
                    cfg.selective_particle_max_substeps) {
                    ++exhausted;
                    if (!worst_exhausted ||
                        (!std::isfinite(diagnostic.final_scaled_residual) &&
                         std::isfinite(
                             worst_exhausted->final_scaled_residual)) ||
                        (std::isfinite(diagnostic.final_scaled_residual) &&
                         std::isfinite(
                             worst_exhausted->final_scaled_residual) &&
                         diagnostic.final_scaled_residual >
                             worst_exhausted->final_scaled_residual)) {
                        worst_exhausted = &diagnostic;
                    }
                }
            }
            write_failed_particle_diagnostics(
                failed_particle_diagnostics, time_n, refinement_restarts,
                cfg, e.failed_diagnostics, exhausted > 0);
            if (exhausted > 0) {
                if (solve_info_out) *solve_info_out = selective_info;
                std::ostringstream msg;
                msg << "direct macro solve failed (" << direct_failure
                    << "); selective particle fallback reached h=dt/"
                    << cfg.selective_particle_max_substeps << " with "
                    << exhausted << " particles still unconverged; macro dt="
                    << cfg.dt << " was not changed";
                if (worst_exhausted) {
                    msg << "; worst exhausted particle="
                        << worst_exhausted->particle
                        << ", failed_substep_zero_based="
                        << worst_exhausted->failed_substep
                        << ", local_iterations="
                        << worst_exhausted->local_iterations
                        << ", reason="
                        << local_particle_failure_reason_name(
                               worst_exhausted->reason)
                        << ", scaled_residual="
                        << worst_exhausted->final_scaled_residual;
                }
                throw NonlinearSolveFailure(msg.str());
            }

            for (const FailedParticleDiagnostic& diagnostic :
                 e.failed_diagnostics) {
                const std::size_t p = diagnostic.particle;
                particle_substeps[p] = std::min(
                    cfg.selective_particle_max_substeps,
                    2 * particle_substeps[p]);
            }
            ++refinement_restarts;

            std::array<std::size_t, 7> counts{};
            std::size_t total_substeps = 0;
            int maximum_substeps = 1;
            for (int nsub : particle_substeps) {
                int index = 0;
                for (int value = nsub; value > 1; value /= 2) ++index;
                index = std::min<int>(index,
                    static_cast<int>(counts.size()) - 1);
                ++counts[static_cast<std::size_t>(index)];
                total_substeps += static_cast<std::size_t>(nsub);
                maximum_substeps = std::max(maximum_substeps, nsub);
            }
            const Real work_ratio = np
                ? static_cast<Real>(total_substeps) / static_cast<Real>(np)
                : 0.0;
            std::cout << "selective particle refinement at t=" << time_n
                      << ": " << e.failed_diagnostics.size()
                      << " failed particles increased one level; max=dt/"
                      << maximum_substeps
                      << ", work_ratio=" << work_ratio
                      << ", counts[n=1,2,4,8,16,32,64]="
                      << counts[0] << "," << counts[1] << ","
                      << counts[2] << "," << counts[3] << ","
                      << counts[4] << "," << counts[5] << ","
                      << counts[6]
                      << "; restarting outer current Anderson\n";
        } catch (const NonlinearSolveFailure& e) {
            selective_info.selective_fallback_used = true;
            selective_info.direct_attempt_iterations = direct_info.iterations;
            selective_info.selective_refinement_restarts =
                refinement_restarts;
            if (solve_info_out) *solve_info_out = selective_info;
            std::ostringstream msg;
            msg << "direct macro solve failed (" << direct_failure
                << "); selective outer current solve also failed without a "
                   "refinable local-particle set (" << e.what()
                << "); macro dt=" << cfg.dt << " was not changed";
            throw NonlinearSolveFailure(msg.str());
        }
    }
}

// ---------------------------------------------------------------------------
// Advance one fixed field step; selectively refine failed particles only when enabled
//
// Inputs:
//   st : const State&; accepted particle and mesh state
//   g : const Grid&; periodic grid geometry and Fourier symbols
//   cfg : const Config&; simulation settings and solver tolerances
//   quad : const Quadrature&; Gauss nodes and weights on [0,1]
//   hist : std::vector<NonlinearHistoryEntry>&; outer residual/update history, overwritten
//   solve_info_out : NonlinearSolveInfo*; optional output solver statistics pointer
//   time_n : Real; time at the start of the field step
//   failed_particle_diagnostics : std::ostream*; optional stream for failed-particle records
//
// Output:
//   State: new particle/field state; input state remains unchanged.
//
// Dependencies:
//   nonlinear_step_selective_particle_fallback, nonlinear_step_uniform_subcycled,
//   write_failed_particle_diagnostics.
//   C++17 standard library (headers listed at the top of this file).
// ---------------------------------------------------------------------------
static State nonlinear_step_orbit(const State& st, const Grid& g, const Config& cfg, const Quadrature& quad,
                                  std::vector<NonlinearHistoryEntry>& hist,
                                  NonlinearSolveInfo* solve_info_out = nullptr,
                                  Real time_n = std::numeric_limits<Real>::infinity(),
                                  std::ostream* failed_particle_diagnostics = nullptr) {
    // Every particle first solves the full [t_n,t_n+dt]
    // equation with frozen trial fields before the outer current map exists.
    // A level-one layout reuses the verified particle kernel without any
    // temporal subdivision or direct coupled-solver attempt.
    {
        if (cfg.adaptive_dt || cfg.uniform_particle_subcycling) {
            throw std::runtime_error("particle-inner experiment requires adaptive_dt=false, "
                "uniform_particle_subcycling=false");
        }
        const std::vector<int> full_step_layout(st.x.size(), 1);
        std::ofstream particle_history(cfg.output_prefix + "_particle_inner_history.csv",
                                      std::ios::app);
        if (!particle_history) throw std::runtime_error("cannot open particle inner history");
        NonlinearSolveInfo info;
        try {
            State out = nonlinear_step_uniform_subcycled(
                st, g, cfg, quad, hist, &info, time_n,
                &full_step_layout, &particle_history);
            info.selective_fallback_used = false;
            info.convergence_method = "nested_full_dt_" +
                info.convergence_method.substr(std::string("selective_").size());
            if (solve_info_out) *solve_info_out = info;
            return out;
        } catch (const SelectiveParticleRefinementNeeded& e) {
            // AUXILIARY SPLINE COMPARISON: retry from the unchanged state.
            // Only particles whose local solve fails are refined. The existing
            // controller doubles their substeps and restarts outer Anderson.
            if (cfg.selective_particle_fallback) {
                return nonlinear_step_selective_particle_fallback(
                    st, g, cfg, quad, hist, solve_info_out, time_n, info,
                    e.what(), failed_particle_diagnostics, &particle_history);
            }
            write_failed_particle_diagnostics(failed_particle_diagnostics,
                time_n, 0, cfg, e.failed_diagnostics, true);
            info.selective_fallback_used = false;
            if (solve_info_out) *solve_info_out = info;
            throw NonlinearSolveFailure(std::string("inner particle solve failed over full dt; ") +
                "time and dt unchanged: " + e.what());
        } catch (const NonlinearSolveFailure&) {
            info.selective_fallback_used = false;
            if (solve_info_out) *solve_info_out = info;
            throw;
        }
    }
}

// -----------------------------------------------------------------------------
// Unit tests
// -----------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Rms vecfield components
//
// Inputs:
//   V : const VecField&; V
//
// Output:
//   Real: rms vecfield components.
//
// Dependencies:
//   C++17 standard library (headers listed at the top of this file).
// ---------------------------------------------------------------------------
static Real rms_vecfield_components(const VecField& V) {
    long double s = 0.0L;
    std::size_t count = 0;
    for (int d = 0; d < 3; ++d) {
        for (Real x : V[d]) {
            s += static_cast<long double>(x) * static_cast<long double>(x);
            ++count;
        }
    }
    return std::sqrt(static_cast<Real>(s / std::max<std::size_t>(1, count)));
}

// ---------------------------------------------------------------------------
// Regression test: spectral nyquist operator compatibility
//
// VALIDATION ONLY: called when run_unit_tests=true.
//
// Inputs:
//   os : std::ostream&; test report stream
//
// Output:
//   bool: true when the stated numerical regression passes; writes measured residuals.
//
// Dependencies:
//   fft_real, ifft_real, make_grid, rms_field, rms_vecfield_components, spectral_curl,
//   spectral_divergence_vector, spectral_gradient_scalar, zero_vecfield.
//   C++17 standard library (headers listed at the top of this file).
// ---------------------------------------------------------------------------
static bool test_spectral_nyquist_operator_compatibility(std::ostream& os) {
    // Regression test for the even-grid Nyquist bug.  The real-valued spectral
    // derivative has zero represented derivative at the Nyquist index, so the
    // Laplacian used by Poisson/CN must be built from the same derivative
    // symbols.  The mixed mode (-1)^i cos(2*pi*j/ny) is deliberately placed on
    // an x-Nyquist plane but has a nonzero y derivative; this is exactly the
    // class of particle-current noise that exposed the long-time Gauss drift.
    Config cfg;
    cfg.nx = cfg.ny = cfg.nz = 8;
    cfg.Lx = cfg.Ly = cfg.Lz = 2.0 * PI;
    Grid g = make_grid(cfg);

    Field f(g.N, 0.0);
    VecField V = zero_vecfield(g.N);
    for (int i = 0; i < g.nx; ++i) {
        const Real x = 2.0 * PI * static_cast<Real>(i) / static_cast<Real>(g.nx);
        const Real sxnyq = (i % 2 == 0) ? 1.0 : -1.0;
        for (int j = 0; j < g.ny; ++j) {
            const Real y = 2.0 * PI * static_cast<Real>(j) / static_cast<Real>(g.ny);
            for (int k = 0; k < g.nz; ++k) {
                const Real z = 2.0 * PI * static_cast<Real>(k) / static_cast<Real>(g.nz);
                const int id = g.index(i, j, k);
                f[id] = sxnyq * std::cos(y) + 0.25 * std::sin(x) * std::cos(2.0 * z);
                V[0][id] = 0.17 * sxnyq * std::sin(y) + 0.11 * std::cos(z);
                V[1][id] = sxnyq * std::sin(y) + 0.13 * std::cos(x) * std::sin(z);
                V[2][id] = 0.19 * sxnyq * std::cos(y) + 0.07 * std::sin(x);
            }
        }
    }

    const VecField grad_f = spectral_gradient_scalar(f, g);
    const Field div_grad_f = spectral_divergence_vector(grad_f, g);
    CField fhat = fft_real(f, g);
    CField laphat(g.N);
    for (int id = 0; id < g.N; ++id) laphat[id] = -g.k2[id] * fhat[id];
    const Field lap_f = ifft_real(std::move(laphat), g);
    Field diff(g.N, 0.0);
    for (int id = 0; id < g.N; ++id) diff[id] = div_grad_f[id] - lap_f[id];
    const Real lap_compat_abs = rms_field(diff);
    const Real lap_scale = rms_field(lap_f) + 1.0e-300;

    const VecField curl_grad_f = spectral_curl(grad_f, g);
    const VecField curl_V = spectral_curl(V, g);
    const Field div_curl_V = spectral_divergence_vector(curl_V, g);
    const Real curl_grad_abs = rms_vecfield_components(curl_grad_f);
    const Real div_curl_abs = rms_field(div_curl_V);

    os << std::scientific << std::setprecision(6)
       << "Spectral Nyquist compatibility unit test: |divgrad-lap|=" << lap_compat_abs
       << " rel=" << lap_compat_abs / lap_scale
       << " |curlgrad|=" << curl_grad_abs
       << " |divcurl|=" << div_curl_abs << "\n";
    if (!(lap_compat_abs / lap_scale < 5.0e-13 &&
          curl_grad_abs < 5.0e-13 &&
          div_curl_abs < 5.0e-13)) {
        os << "FAILED: spectral grad/div/curl/Laplacian symbols are not mutually compatible at Nyquist\n";
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Regression test: field continuity gauss nyquist regression
//
// VALIDATION ONLY: called when run_unit_tests=true.
//
// Inputs:
//   os : std::ostream&; test report stream
//
// Output:
//   bool: true when the stated numerical regression passes; writes measured residuals.
//
// Dependencies:
//   cn_potential_update, compute_diagnostics, compute_field_energy,
//   continuity_charge_update, make_grid, spectral_gradient_scalar, zero_vecfield.
//   C++17 standard library (headers listed at the top of this file).
// ---------------------------------------------------------------------------
static bool test_field_continuity_gauss_nyquist_regression(std::ostream& os) {
    // Field-only source-ordering test with no particles.  A current containing
    // mixed Nyquist-plane content is accepted, charge is advanced from the same
    // spectral divergence, and then the CN potential equations are solved.  If
    // k2 is not the Laplacian induced by the represented grad/div symbols, one
    // step leaves an O(1e-3--1e-1) Gauss defect on this small grid.  With the
    // compatible Nyquist treatment it is roundoff-level.
    Config cfg;
    cfg.nx = cfg.ny = cfg.nz = 8;
    cfg.Lx = cfg.Ly = cfg.Lz = 2.0 * PI;
    cfg.dt = 0.17;
    cfg.kappa = 1.0;
    cfg.sigma1 = 1.0;
    cfg.sigma2 = 1.0;
    Grid g = make_grid(cfg);

    State s0;
    s0.phi.assign(g.N, 0.0);
    s0.psi.assign(g.N, 0.0);
    s0.rho.assign(g.N, 0.0);
    s0.A = zero_vecfield(g.N);
    s0.U = zero_vecfield(g.N);
    s0.J = zero_vecfield(g.N);

    VecField J = zero_vecfield(g.N);
    for (int i = 0; i < g.nx; ++i) {
        const Real x = 2.0 * PI * static_cast<Real>(i) / static_cast<Real>(g.nx);
        const Real sxnyq = (i % 2 == 0) ? 1.0 : -1.0;
        for (int j = 0; j < g.ny; ++j) {
            const Real y = 2.0 * PI * static_cast<Real>(j) / static_cast<Real>(g.ny);
            for (int k = 0; k < g.nz; ++k) {
                const Real z = 2.0 * PI * static_cast<Real>(k) / static_cast<Real>(g.nz);
                const int id = g.index(i, j, k);
                J[0][id] = 0.21 * std::sin(x) + 0.07 * sxnyq * std::cos(z);
                J[1][id] = sxnyq * std::sin(y) + 0.09 * std::cos(x) * std::sin(z);
                J[2][id] = 0.15 * sxnyq * std::cos(y) + 0.05 * std::sin(z);
            }
        }
    }

    Field rho_end = continuity_charge_update(s0.rho, J, cfg.dt, g);
    Field rho_mid(g.N, 0.0);
    for (int id = 0; id < g.N; ++id) rho_mid[id] = 0.5 * (rho_end[id] + s0.rho[id]);

    Field phi1, psi1;
    VecField A1, U1;
    cn_potential_update(s0, rho_mid, J, cfg.dt, g, cfg.kappa, cfg.sigma1, cfg.sigma2,
                        phi1, psi1, A1, U1);

    State s1 = s0;
    s1.phi = std::move(phi1);
    s1.psi = std::move(psi1);
    s1.rho = std::move(rho_end);
    s1.A = std::move(A1);
    s1.U = std::move(U1);
    s1.J = J;

    const Diagnostics d = compute_diagnostics(s1, g, cfg.sigma1, cfg.sigma2, cfg.kappa, cfg.perturbation_mode);

    Field phi_mid(g.N, 0.0);
    VecField U_mid = zero_vecfield(g.N);
    for (int id = 0; id < g.N; ++id) phi_mid[id] = 0.5 * s1.phi[id];
    for (int m = 0; m < 3; ++m) for (int id = 0; id < g.N; ++id) U_mid[m][id] = 0.5 * s1.U[m][id];
    const VecField grad_phi_mid = spectral_gradient_scalar(phi_mid, g);
    VecField E_mid = zero_vecfield(g.N);
    for (int m = 0; m < 3; ++m) {
        for (int id = 0; id < g.N; ++id) E_mid[m][id] = -grad_phi_mid[m][id] - U_mid[m][id];
    }
    long double grid_power = 0.0L;
    for (int m = 0; m < 3; ++m) for (int id = 0; id < g.N; ++id) grid_power += static_cast<long double>(J[m][id]) * E_mid[m][id];
    const Real field_energy_residual = compute_field_energy(s1, g, cfg.sigma1, cfg.sigma2)
                                     + static_cast<Real>(cfg.dt * grid_power * g.dV);

    os << std::scientific << std::setprecision(6)
       << "Field-continuity Nyquist Gauss unit test: gauss_rms=" << d.gauss_rms
       << " gauge_rms=" << d.gauge_rms
       << " field_energy_residual=" << field_energy_residual << "\n";
    if (!(d.gauss_rms < 5.0e-13 && d.gauge_rms < 5.0e-13 &&
          std::abs(field_energy_residual) < 5.0e-12)) {
        os << "FAILED: CN field/continuity update did not preserve constraints with Nyquist-plane current\n";
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Regression test: relativistic hc static B
//
// VALIDATION ONLY: called when run_unit_tests=true.
//
// Inputs:
//   os : std::ostream&; test report stream
//
// Output:
//   bool: true when the stated numerical regression passes; writes measured residuals.
//
// Dependencies:
//   dot3, higuera_cary_rotate_with_K, matvec3, momentum_from_v, norm3, velocity_from_p,
//   zero_mat3.
//   C++17 standard library (headers listed at the top of this file).
// ---------------------------------------------------------------------------
static bool test_relativistic_hc_static_B(std::ostream& os) {
    // Particle-only prescribed-field test: E=0, uniform B_z.  The HC magnetic
    // rotation generated by a skew K must preserve |p| and therefore gamma and
    // kinetic energy.  This is the static-field test that fixes the sign
    // convention for K v = v x B.
    const Real kappa = 1.0;
    const Real m = 1.0;
    const Real q = -1.0;
    const Real dt = 0.05;
    const Real Bz = 0.7;
    Mat3 K = zero_mat3();
    K[0][1] = Bz;
    K[1][0] = -Bz;
    Vec3 p0 = momentum_from_v(Vec3{0.35, 0.12, 0.0}, m, kappa);
    Vec3 p1 = higuera_cary_rotate_with_K(p0, K, q, m, kappa, dt);
    const Real rel_norm_err = std::abs(norm3(p1) - norm3(p0)) / (norm3(p0) + 1.0e-300);
    const Vec3 v0 = velocity_from_p(p0, m, kappa);
    const Real no_work = dot3(v0, matvec3(K, v0));
    os << std::scientific << std::setprecision(6)
       << "Relativistic HC static-B unit test: rel_norm_err=" << rel_norm_err
       << " v.Kv=" << no_work << "\n";
    if (!(rel_norm_err < 5.0e-14 && std::abs(no_work) < 5.0e-15)) {
        os << "FAILED: HC static-B rotation did not preserve momentum norm or skew no-work\n";
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Regression test: relativistic static E secant work
//
// VALIDATION ONLY: called when run_unit_tests=true.
//
// Inputs:
//   os : std::ostream&; test report stream
//
// Output:
//   bool: true when the stated numerical regression passes; writes measured residuals.
//
// Dependencies:
//   add3, dot3, kinetic_energy_from_p, kinetic_secant_velocity, momentum_from_v, mul3.
//   C++17 standard library (headers listed at the top of this file).
// ---------------------------------------------------------------------------
static bool test_relativistic_static_E_secant_work(std::ostream& os) {
    // Prescribed uniform electric field, no magnetic field.  The relativistic
    // discrete-gradient velocity must make the particle work exactly equal the
    // kinetic-energy change for the electric kick.
    const Real kappa = 1.0;
    const Real m = 1.0;
    const Real q = -1.0;
    const Real dt = 0.03;
    const Vec3 E{0.2, -0.1, 0.05};
    const Vec3 p0 = momentum_from_v(Vec3{0.25, 0.05, -0.02}, m, kappa);
    const Vec3 p1 = add3(p0, mul3(q * dt, E));
    const Vec3 vdg = kinetic_secant_velocity(p1, p0, m, kappa);
    const Real dK = kinetic_energy_from_p(p1, m, kappa) - kinetic_energy_from_p(p0, m, kappa);
    const Real work = q * dt * dot3(E, vdg);
    const Real err = std::abs(dK - work);
    os << std::scientific << std::setprecision(6)
       << "Relativistic static-E secant unit test: |dK-work|=" << err << "\n";
    if (!(err < 5.0e-15)) {
        os << "FAILED: relativistic electric secant work identity failed\n";
        return false;
    }
    return true;
}


// ---------------------------------------------------------------------------
// Regression test: skew K to Beff sign
//
// VALIDATION ONLY: called when run_unit_tests=true.
//
// Inputs:
//   os : std::ostream&; test report stream
//
// Output:
//   bool: true when the stated numerical regression passes; writes measured residuals.
//
// Dependencies:
//   beff_from_skew_K, cross3, higuera_cary_rotate_with_K, matvec3, momentum_from_v, mul3,
//   norm3, sub3, zero_mat3.
//   C++17 standard library (headers listed at the top of this file).
// ---------------------------------------------------------------------------
static bool test_skew_K_to_Beff_sign(std::ostream& os) {
    // Sign-convention regression test.  For the gauge A=(0,Bx,0), the
    // orbit-discrete-gradient has D_yx=B and K=D^T-D satisfies K v = v x B zhat.
    // This test checks both the axial-vector extraction and the first-order
    // direction of the HC rotation.
    const Real kappa = 1.0;
    const Real m = 1.0;
    const Real q = 1.0;
    const Real Bz = 0.7;
    Mat3 K = zero_mat3();
    K[0][1] = Bz;
    K[1][0] = -Bz;
    const Vec3 v{0.20, 0.10, -0.03};
    const Vec3 Kv = matvec3(K, v);
    const Vec3 B = beff_from_skew_K(K);
    const Vec3 v_cross_B = cross3(v, Vec3{0.0, 0.0, Bz});
    const Real beff_err = norm3(sub3(B, Vec3{0.0, 0.0, Bz}));
    const Real cross_err = norm3(sub3(Kv, v_cross_B));

    const Real dt = 1.0e-7;
    const Vec3 p0 = momentum_from_v(v, m, kappa);
    const Vec3 p1 = higuera_cary_rotate_with_K(p0, K, q, m, kappa, dt);
    const Vec3 dpdt_num = mul3(1.0 / dt, sub3(p1, p0));
    const Vec3 dpdt_ref = mul3(q, Kv);
    const Real orientation_err = norm3(sub3(dpdt_num, dpdt_ref)) / (norm3(dpdt_ref) + 1.0e-300);

    os << std::scientific << std::setprecision(6)
       << "K-to-Beff sign unit test: beff_err=" << beff_err
       << " K_cross_err=" << cross_err
       << " hc_orientation_rel_err=" << orientation_err << "\n";
    if (!(beff_err < 1.0e-14 && cross_err < 1.0e-14 && orientation_err < 5.0e-7)) {
        os << "FAILED: K-to-Beff sign convention or HC rotation orientation failed\n";
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Regression test: canonical no At equivalence
//
// VALIDATION ONLY: called when run_unit_tests=true.
//
// Inputs:
//   os : std::ostream&; test report stream
//
// Output:
//   bool: true when the stated numerical regression passes; writes measured residuals.
//
// Dependencies:
//   add3, matvec3, mul3, norm3, sub3, zero_mat3.
//   C++17 standard library (headers listed at the top of this file).
// ---------------------------------------------------------------------------
static bool test_canonical_no_At_equivalence(std::ostream& os) {
    // Algebraic no-explicit-A_t test.  Mechanical half kicks using
    // Ebar=-grad(phi)-U must be exactly equivalent to the canonical endpoint
    // formulas when A^+-A^-=dt(U+D vbar).  This is the implementation-level
    // check for the cancellation of the explicit U=A_t force gather.
    const Real q = -0.8;
    const Real dt = 0.037;
    const Vec3 Pn{0.12, -0.31, 0.21};
    const Vec3 Aminus{0.05, -0.02, 0.03};
    const Vec3 G{0.11, -0.07, 0.02};
    const Vec3 U{0.03, 0.04, -0.01};
    const Vec3 vbar{0.20, -0.15, 0.10};
    Mat3 D = zero_mat3();
    D[0][0] = 0.20; D[0][1] = -0.10; D[0][2] = 0.05;
    D[1][0] = 0.07; D[1][1] = 0.03;  D[1][2] = -0.04;
    D[2][0] = -0.02; D[2][1] = 0.09; D[2][2] = 0.01;
    const Vec3 Dv = matvec3(D, vbar);
    const Vec3 Aplus = add3(Aminus, mul3(dt, add3(U, Dv)));
    const Vec3 Aep = mul3(0.5, add3(Aminus, Aplus));

    const Vec3 pn = sub3(Pn, mul3(q, Aminus));
    const Vec3 Ebar = mul3(-1.0, add3(G, U));
    const Vec3 pminus_mech = add3(pn, mul3(0.5 * q * dt, Ebar));

    Vec3 pminus_can = Pn;
    pminus_can = sub3(pminus_can, mul3(q, Aep));
    pminus_can = sub3(pminus_can, mul3(0.5 * q * dt, G));
    pminus_can = add3(pminus_can, mul3(0.5 * q * dt, Dv));

    const Vec3 pplus{0.09, -0.28, 0.24};
    const Vec3 pn1_mech = add3(pplus, mul3(0.5 * q * dt, Ebar));
    const Vec3 Pn1_mech = add3(pn1_mech, mul3(q, Aplus));

    Vec3 Pn1_can = pplus;
    Pn1_can = add3(Pn1_can, mul3(q, Aep));
    Pn1_can = sub3(Pn1_can, mul3(0.5 * q * dt, G));
    Pn1_can = add3(Pn1_can, mul3(0.5 * q * dt, Dv));

    const Real pre_err = norm3(sub3(pminus_can, pminus_mech));
    const Real post_err = norm3(sub3(Pn1_can, Pn1_mech));
    os << std::scientific << std::setprecision(6)
       << "Canonical no-At equivalence unit test: pre_err=" << pre_err
       << " post_err=" << post_err << "\n";
    if (!(pre_err < 5.0e-16 && post_err < 5.0e-16)) {
        os << "FAILED: canonical no-At formulas are not equivalent to mechanical E=-grad(phi)-U kicks\n";
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Regression test: orbit deposit gather adjointness
//
// VALIDATION ONLY: called when run_unit_tests=true.
//
// Inputs:
//   os : std::ostream&; test report stream
//
// Output:
//   bool: true when the stated numerical regression passes; writes measured residuals.
//
// Dependencies:
//   dot3, gauss_legendre_01, make_grid, mul3, path_average_deposit_current,
//   path_average_gather_vector, zero_vecfield.
//   C++17 standard library (headers listed at the top of this file).
// ---------------------------------------------------------------------------
static bool test_orbit_deposit_gather_adjointness(std::ostream& os) {
    // Standalone scatter/gather power identity.  The same orbit weights must be
    // used in current deposition and field gather, otherwise particle work and
    // mesh work cannot cancel in the field energy equation.
    Config cfg;
    cfg.nx = cfg.ny = cfg.nz = 8;
    cfg.Lx = cfg.Ly = cfg.Lz = 2.0 * PI;
    cfg.dt = 0.20;
    cfg.spline_order = 2;
    cfg.orbit_quad_order = 18;
    Grid g = make_grid(cfg);
    Quadrature quad = gauss_legendre_01(cfg.orbit_quad_order);

    std::mt19937_64 rng(23);
    std::uniform_real_distribution<Real> uni(-0.5, 0.5);
    std::normal_distribution<Real> normal(0.0, 1.0);
    const std::size_t np = 64;
    std::vector<Vec3> x0(np), vbar(np), dx_path(np);
    std::vector<Real> q(np);
    for (std::size_t p = 0; p < np; ++p) {
        x0[p] = Vec3{cfg.Lx * uni(rng), cfg.Ly * uni(rng), cfg.Lz * uni(rng)};
        vbar[p] = Vec3{0.35 * normal(rng), 0.20 * normal(rng), 0.15 * normal(rng)};
        dx_path[p] = mul3(cfg.dt, vbar[p]);
        q[p] = 0.1 * normal(rng);
    }
    VecField E = zero_vecfield(g.N);
    for (int d = 0; d < 3; ++d) for (int id = 0; id < g.N; ++id) E[d][id] = normal(rng);

    const auto Ebar = path_average_gather_vector(x0, dx_path, E, g, cfg.spline_order, quad, true);
    const VecField Jbar = path_average_deposit_current(x0, dx_path, vbar, q, g, cfg.spline_order, quad, true);
    long double particle_power = 0.0L, mesh_power = 0.0L;
    for (std::size_t p = 0; p < np; ++p) particle_power += static_cast<long double>(q[p]) * dot3(vbar[p], Ebar[p]);
    for (int d = 0; d < 3; ++d) for (int id = 0; id < g.N; ++id) mesh_power += static_cast<long double>(Jbar[d][id]) * E[d][id];
    const Real residual = static_cast<Real>(particle_power - mesh_power * g.dV);
    const Real scale = std::abs(static_cast<Real>(particle_power)) + std::abs(static_cast<Real>(mesh_power * g.dV)) + 1.0;
    os << std::scientific << std::setprecision(6)
       << "Orbit deposit/gather adjoint unit test: residual=" << residual
       << " rel=" << std::abs(residual) / scale << "\n";
    if (!(std::abs(residual) / scale < 5.0e-14)) {
        os << "FAILED: orbit scatter/gather work identity failed\n";
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Regression test: RA orbit chain rule
//
// VALIDATION ONLY: called when run_unit_tests=true.
//
// Inputs:
//   os : std::ostream&; test report stream
//
// Output:
//   bool: true when the stated numerical regression passes; writes measured residuals.
//
// Dependencies:
//   compute_RA_orbit_residual, gauss_legendre_01, make_grid, rms_from_vec3s, zero_vecfield.
//   C++17 standard library (headers listed at the top of this file).
// ---------------------------------------------------------------------------
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


// ---------------------------------------------------------------------------
// Regression test: RA orbit chain rule with crossings
//
// VALIDATION ONLY: called when run_unit_tests=true.
//
// Inputs:
//   os : std::ostream&; test report stream
//
// Output:
//   bool: true when the stated numerical regression passes; writes measured residuals.
//
// Dependencies:
//   compute_RA_orbit_residual, compute_path_stats, gauss_legendre_01, make_grid,
//   rms_from_vec3s, zero_vecfield.
//   C++17 standard library (headers listed at the top of this file).
// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
// Regression test: initial gauss law
//
// VALIDATION ONLY: called when run_unit_tests=true.
//
// Inputs:
//   os : std::ostream&; test report stream
//
// Output:
//   bool: true when the stated numerical regression passes; writes measured residuals.
//
// Dependencies:
//   compute_diagnostics, initialize_two_stream_3d_linear_x, make_grid.
//   C++17 standard library (headers listed at the top of this file).
// ---------------------------------------------------------------------------
static bool test_initial_gauss_law(std::ostream& os) {
    Config cfg;
    cfg.nx = cfg.ny = cfg.nz = 8;
    cfg.Lx = cfg.Ly = cfg.Lz = 2.0 * PI;
    cfg.dt = 0.00625;
    cfg.spline_order = 2;
    cfg.particles_per_cell_pair = 2;
    Grid g = make_grid(cfg);
    State st = initialize_two_stream_3d_linear_x(cfg, g);
    Diagnostics d = compute_diagnostics(st, g, cfg.sigma1, cfg.sigma2, cfg.kappa, cfg.perturbation_mode);
    os << std::scientific << std::setprecision(6)
       << "Initial Gauss-law unit test: gauss_rms=" << d.gauss_rms << "\n";
    if (!(d.gauss_rms < 5.0e-11)) {
        os << "FAILED: initial Gauss residual is too large\n";
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Regression test: one step gauss law and RA
//
// VALIDATION ONLY: called when run_unit_tests=true.
//
// Inputs:
//   os : std::ostream&; test report stream
//
// Output:
//   bool: true when the stated numerical regression passes; writes measured residuals.
//
// Dependencies:
//   compute_diagnostics, compute_step_diagnostics, gauss_legendre_01,
//   initialize_two_stream_3d_linear_x, make_grid, nonlinear_step_orbit.
//   C++17 standard library (headers listed at the top of this file).
// ---------------------------------------------------------------------------
static bool test_one_step_gauss_law_and_RA(std::ostream& os) {
    Config cfg;
    cfg.nx = cfg.ny = cfg.nz = 4;
    cfg.Lx = cfg.Ly = cfg.Lz = 2.0 * PI;
    cfg.dt = 0.00625;
    cfg.n_steps = 1;
    cfg.spline_order = 2;
    cfg.particles_per_cell_pair = 2;
    cfg.orbit_quad_order = 16;
    cfg.nonlinear_rtol = 1.0e-11; // SG added
    cfg.nonlinear_max_iter = 16;
    Grid g = make_grid(cfg);
    Quadrature quad = gauss_legendre_01(cfg.orbit_quad_order);
    State s0 = initialize_two_stream_3d_linear_x(cfg, g);
    std::vector<NonlinearHistoryEntry> hist;
    State s1 = nonlinear_step_orbit(s0, g, cfg, quad, hist);
    Diagnostics d1 = compute_diagnostics(s1, g, cfg.sigma1, cfg.sigma2, cfg.kappa, cfg.perturbation_mode);
    StepDiagnostics sd = compute_step_diagnostics(s0, s1, g, cfg.dt, cfg.spline_order, quad, cfg.sigma1, cfg.sigma2, cfg.kappa);
    os << std::scientific << std::setprecision(6)
       << "One-step Gauss-law unit test: gauss_rms=" << d1.gauss_rms << "\n"
       << "One-step R_A,p unit test: A_chain_abs_rms=" << sd.A_chain_abs_rms << "\n"
       << "One-step deposit/gather work residual=" << sd.deposit_gather_work_residual << "\n"
       << "One-step particle-energy residual=" << sd.particle_energy_residual << "\n"
       << "One-step field-energy residual=" << sd.field_energy_residual << "\n"
       << "One-step total-energy residual=" << sd.delta_total_energy << "\n";
    if (!(d1.gauss_rms < 5.0e-10 &&
          sd.A_chain_abs_rms < 5.0e-10 &&
          std::abs(sd.deposit_gather_work_residual) < 5.0e-13 &&
          std::abs(sd.particle_energy_residual) < 5.0e-10 &&
          std::abs(sd.field_energy_residual) < 5.0e-10 &&
          std::abs(sd.delta_total_energy) < 5.0e-10)) {
        os << "FAILED: one-step Gauss, R_A,p, work, or energy-balance test failed\n";
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Regression test: small multistep energy conservation
//
// VALIDATION ONLY: called when run_unit_tests=true.
//
// Inputs:
//   os : std::ostream&; test report stream
//
// Output:
//   bool: true when the stated numerical regression passes; writes measured residuals.
//
// Dependencies:
//   compute_diagnostics, compute_step_diagnostics, gauss_legendre_01,
//   initialize_two_stream_3d_linear_x, make_grid, nonlinear_step_orbit.
//   C++17 standard library (headers listed at the top of this file).
// ---------------------------------------------------------------------------
static bool test_small_multistep_energy_conservation(std::ostream& os) {
    // End-to-end relativistic coupled PIC regression: a tiny 3D two-stream run
    // must keep total energy at the level dictated by solver tolerance while
    // maintaining the chain-rule and scatter/gather residuals.
    Config cfg;
    cfg.nx = cfg.ny = cfg.nz = 4;
    cfg.Lx = cfg.Ly = cfg.Lz = 2.0 * PI;
    cfg.dt = 0.00625;
    cfg.n_steps = 4;
    cfg.spline_order = 2;
    cfg.particles_per_cell_pair = 2;
    cfg.orbit_quad_order = 16;
    cfg.nonlinear_rtol = 1.0e-11; // SG added
    cfg.nonlinear_max_iter = 16;
    cfg.split_orbit_at_knots = true;
    cfg.two_stream_density_alpha = 0.01;
    cfg.perturbation = 0.0;
    Grid g = make_grid(cfg);
    Quadrature quad = gauss_legendre_01(cfg.orbit_quad_order);
    State st = initialize_two_stream_3d_linear_x(cfg, g);
    const Real E0 = compute_diagnostics(st, g, cfg.sigma1, cfg.sigma2, cfg.kappa, cfg.perturbation_mode).total_energy;
    Real max_rel_drift = 0.0;
    Real max_step_dE = 0.0;
    Real max_A_chain = 0.0;
    Real max_particle_res = 0.0;
    Real max_field_res = 0.0;
    for (int n = 0; n < cfg.n_steps; ++n) {
        State old = st;
        std::vector<NonlinearHistoryEntry> hist;
        st = nonlinear_step_orbit(old, g, cfg, quad, hist);
        StepDiagnostics sd = compute_step_diagnostics(old, st, g, cfg.dt, cfg.spline_order, quad, cfg.sigma1, cfg.sigma2, cfg.kappa, cfg.split_orbit_at_knots);
        Diagnostics d = compute_diagnostics(st, g, cfg.sigma1, cfg.sigma2, cfg.kappa, cfg.perturbation_mode);
        max_rel_drift = std::max(max_rel_drift, std::abs((d.total_energy - E0) / E0));
        max_step_dE = std::max(max_step_dE, std::abs(sd.delta_total_energy));
        max_A_chain = std::max(max_A_chain, sd.A_chain_abs_rms);
        max_particle_res = std::max(max_particle_res, std::abs(sd.particle_energy_residual));
        max_field_res = std::max(max_field_res, std::abs(sd.field_energy_residual));
    }
    os << std::scientific << std::setprecision(6)
       << "Small multistep energy unit test: max_rel_drift=" << max_rel_drift
       << " max_step_dE=" << max_step_dE
       << " max_A_chain=" << max_A_chain
       << " max_particle_res=" << max_particle_res
       << " max_field_res=" << max_field_res << "\n";
    if (!(max_rel_drift < 5.0e-11 && max_step_dE < 5.0e-10 &&
          max_A_chain < 5.0e-10 && max_particle_res < 5.0e-10 && max_field_res < 5.0e-10)) {
        os << "FAILED: small multistep energy-conservation regression failed\n";
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Regression test: instability initialization and theory
//
// VALIDATION ONLY: called when run_unit_tests=true.
//
// Inputs:
//   os : std::ostream&; test report stream
//
// Output:
//   bool: true when the stated numerical regression passes; writes measured residuals.
//
// Dependencies:
//   cold_two_stream_growth_rate, cold_weibel_growth_rate, compute_diagnostics,
//   initialize_two_stream_3d_linear_x, initialize_weibel_filamentation_3d, make_grid,
//   rms_field, sqr.
//   C++17 standard library (headers listed at the top of this file).
// ---------------------------------------------------------------------------
static bool test_instability_initialization_and_theory(std::ostream& os) {
    // Test the two physical benchmark setup paths without doing a long run.
    // Two-stream: neutral periodic density, positive cold relativistic growth rate.
    // Weibel: neutral total charge, nonzero eigenmode J_x, and a nonzero seeded B_z(k_y).
    Config ts;
    ts.test_case = "cold_relativistic_two_stream";
    ts.nx = ts.ny = ts.nz = 4;
    ts.Lx = 8.0 * PI;
    ts.Ly = ts.Lz = 2.0 * PI;
    ts.spline_order = 1;
    ts.particles_per_cell_pair = 2;
    ts.v0 = 0.9;
    ts.perturbation = 0.0;
    ts.two_stream_density_alpha = 0.01;
    Grid gts = make_grid(ts);
    State sts = initialize_two_stream_3d_linear_x(ts, gts);
    Diagnostics dts = compute_diagnostics(sts, gts, ts.sigma1, ts.sigma2, ts.kappa, ts.perturbation_mode);
    const Real gamma_ts = cold_two_stream_growth_rate(ts, gts);

    Config wb;
    wb.test_case = "cold_relativistic_weibel";
    wb.nx = wb.ny = wb.nz = 4;
    wb.Lx = wb.Ly = wb.Lz = 2.0 * PI;
    wb.spline_order = 1;
    wb.particles_per_cell_pair = 1;
    wb.v0 = 0.9;
    wb.weibel_B0 = 1.0e-4;
    Grid gwb = make_grid(wb);
    State swb = initialize_weibel_filamentation_3d(wb, gwb);
    Diagnostics dwb = compute_diagnostics(swb, gwb, wb.sigma1, wb.sigma2, wb.kappa, wb.perturbation_mode);
    const Real gamma_wb = cold_weibel_growth_rate(wb, gwb);
    const Real J_rms = std::sqrt((sqr(rms_field(swb.J[0])) + sqr(rms_field(swb.J[1])) + sqr(rms_field(swb.J[2]))) / 3.0);

    os << std::scientific << std::setprecision(6)
       << "Instability IC/theory unit test: ts_gamma=" << gamma_ts
       << " ts_gauss=" << dts.gauss_rms
       << " wb_gamma=" << gamma_wb
       << " wb_Bz_mode=" << dwb.Bz_y_mode_abs
       << " wb_J_rms=" << J_rms << "\n";
    if (!(gamma_ts > 0.0 && dts.gauss_rms < 5.0e-10 &&
          gamma_wb > 0.0 && dwb.Bz_y_mode_abs > 1.0e-6 && J_rms > 1.0e-8 && dwb.gauss_rms < 5.0e-10)) {
        os << "FAILED: instability initialization or cold-theory reference failed\n";
        return false;
    }
    return true;
}



// ---------------------------------------------------------------------------
// Regression test: two stream eigenmode growth
//
// VALIDATION ONLY: called when run_unit_tests=true.
//
// Inputs:
//   os : std::ostream&; test report stream
//
// Output:
//   bool: true when the stated numerical regression passes; writes measured residuals.
//
// Dependencies:
//   cold_two_stream_growth_rate, compute_diagnostics, gauss_legendre_01,
//   initialize_two_stream_3d_linear_x, make_grid, nonlinear_step_orbit.
//   C++17 standard library (headers listed at the top of this file).
// ---------------------------------------------------------------------------
static bool test_two_stream_eigenmode_growth(std::ostream& os) {
    // Linear-physics regression for the bug that originally motivated this
    // repair.  The previous two-stream deck used a density-only seed and a
    // continuum rate on a finite grid; the mode amplitude decayed initially even
    // though the code conserved energy exactly.  This test initializes the
    // growing cold relativistic eigenvector and fits |E_x(k)| over a short 1D
    // embedded run.  The fitted growth must match the same finite-grid cold-fluid
    // rate written to the diagnostics.
    Config cfg;
    cfg.test_case = "cold_relativistic_two_stream";
    cfg.nx = 16;
    cfg.ny = 1;
    cfg.nz = 1;
    cfg.Lx = 8.0 * PI;
    cfg.Ly = 2.0 * PI;
    cfg.Lz = 2.0 * PI;
    cfg.dt = 0.05;
    cfg.n_steps = 80;
    cfg.spline_order = 1;
    cfg.orbit_quad_order = 6;
    cfg.particles_per_cell_pair = 4;
    cfg.nonlinear_rtol = 1.0e-10; // SG added
    cfg.nonlinear_max_iter = 20;
    cfg.v0 = 0.9;
    cfg.perturbation = 0.0;
    cfg.two_stream_density_alpha = 1.0e-3;
    cfg.two_stream_eigenmode = true;
    cfg.two_stream_finite_grid_theory = true;
    cfg.perturbation_mode = 1;

    Grid g = make_grid(cfg);
    Quadrature quad = gauss_legendre_01(cfg.orbit_quad_order);
    State st = initialize_two_stream_3d_linear_x(cfg, g);
    const Real gamma_ref = cold_two_stream_growth_rate(cfg, g);
    const Real E0 = compute_diagnostics(st, g, cfg.sigma1, cfg.sigma2, cfg.kappa, cfg.perturbation_mode).E_mode_abs;
    if (!(E0 > 0.0 && gamma_ref > 0.0)) {
        os << "FAILED: two-stream eigenmode growth test has zero seed or nonpositive theory rate\n";
        return false;
    }

    long double S0 = 0.0L, S1 = 0.0L, S2 = 0.0L, Sy = 0.0L, Sty = 0.0L;
    int count = 0;
    Real max_rel_energy = 0.0;
    Real max_mode_over_theory_error = 0.0;
    const Real Etot0 = compute_diagnostics(st, g, cfg.sigma1, cfg.sigma2, cfg.kappa, cfg.perturbation_mode).total_energy;
    auto accumulate = [&](Real t, Real amp) {
        const Real y = std::log(std::max<Real>(amp, 1.0e-300));
        S0 += 1.0L;
        S1 += t;
        S2 += static_cast<long double>(t) * t;
        Sy += y;
        Sty += static_cast<long double>(t) * y;
        ++count;
    };
    accumulate(0.0, E0);

    for (int n = 0; n < cfg.n_steps; ++n) {
        State old = st;
        std::vector<NonlinearHistoryEntry> hist;
        st = nonlinear_step_orbit(old, g, cfg, quad, hist);
        const Real t = (n + 1) * cfg.dt;
        const Diagnostics d = compute_diagnostics(st, g, cfg.sigma1, cfg.sigma2, cfg.kappa, cfg.perturbation_mode);
        accumulate(t, d.E_mode_abs);
        const Real theory_amp = E0 * std::exp(gamma_ref * t);
        max_mode_over_theory_error = std::max(max_mode_over_theory_error, std::abs(d.E_mode_abs / theory_amp - 1.0));
        max_rel_energy = std::max(max_rel_energy, std::abs((d.total_energy - Etot0) / Etot0));
    }

    const long double denom = S0 * S2 - S1 * S1;
    const Real gamma_fit = static_cast<Real>((S0 * Sty - S1 * Sy) / denom);
    const Real rel_growth_error = std::abs(gamma_fit - gamma_ref) / gamma_ref;
    os << std::scientific << std::setprecision(6)
       << "Two-stream eigenmode growth unit test: gamma_fit=" << gamma_fit
       << " gamma_ref=" << gamma_ref
       << " rel_growth_error=" << rel_growth_error
       << " max_mode_over_theory_error=" << max_mode_over_theory_error
       << " max_rel_energy=" << max_rel_energy << "\n";
    if (!(rel_growth_error < 5.0e-3 && max_mode_over_theory_error < 5.0e-3 && max_rel_energy < 5.0e-10)) {
        os << "FAILED: two-stream eigenmode growth no longer matches cold-fluid theory\n";
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Regression test: weibel one step gauss RA and energy
//
// VALIDATION ONLY: called when run_unit_tests=true.
//
// Inputs:
//   os : std::ostream&; test report stream
//
// Output:
//   bool: true when the stated numerical regression passes; writes measured residuals.
//
// Dependencies:
//   cold_weibel_growth_rate, compute_diagnostics, compute_step_diagnostics,
//   gauss_legendre_01, initialize_weibel_filamentation_3d, make_grid, nonlinear_step_orbit.
//   C++17 standard library (headers listed at the top of this file).
// ---------------------------------------------------------------------------
static bool test_weibel_one_step_gauss_RA_and_energy(std::ostream& os) {
    // One-step coupled regression for the 3D cold relativistic Weibel deck.  The
    // two-stream one-step test is longitudinal and electrostatic-dominated; this
    // one starts from a magnetic B_z(k_y) seed and verifies the same energy and
    // constraint identities in the transverse/filamentation setup.
    Config cfg;
    cfg.test_case = "cold_relativistic_weibel";
    cfg.nx = cfg.ny = cfg.nz = 4;
    cfg.Lx = cfg.Ly = cfg.Lz = 2.0 * PI;
    cfg.dt = 0.005;
    cfg.n_steps = 1;
    cfg.spline_order = 1;
    cfg.particles_per_cell_pair = 1;
    cfg.orbit_quad_order = 8;
    cfg.nonlinear_rtol = 1.0e-9; // SG added
    cfg.nonlinear_max_iter = 12;
    cfg.v0 = 0.5;
    cfg.weibel_B0 = 1.0e-5;
    cfg.perturbation_mode = 1;
    Grid g = make_grid(cfg);
    Quadrature quad = gauss_legendre_01(cfg.orbit_quad_order);
    State s0 = initialize_weibel_filamentation_3d(cfg, g);
    const Real gamma_ref = cold_weibel_growth_rate(cfg, g);
    const Diagnostics d0 = compute_diagnostics(s0, g, cfg.sigma1, cfg.sigma2, cfg.kappa, cfg.perturbation_mode);
    std::vector<NonlinearHistoryEntry> hist;
    State s1 = nonlinear_step_orbit(s0, g, cfg, quad, hist);
    const Diagnostics d1 = compute_diagnostics(s1, g, cfg.sigma1, cfg.sigma2, cfg.kappa, cfg.perturbation_mode);
    const StepDiagnostics sd = compute_step_diagnostics(s0, s1, g, cfg.dt, cfg.spline_order, quad, cfg.sigma1, cfg.sigma2, cfg.kappa, cfg.split_orbit_at_knots);
    os << std::scientific << std::setprecision(6)
       << "Weibel one-step coupled unit test: gamma_ref=" << gamma_ref
       << " Bz_mode0=" << d0.Bz_y_mode_abs
       << " Bz_mode1=" << d1.Bz_y_mode_abs
       << " gauss_rms=" << d1.gauss_rms
       << " gauge_rms=" << d1.gauge_rms
       << " A_chain_abs_rms=" << sd.A_chain_abs_rms
       << " deposit_gather=" << sd.deposit_gather_work_residual
       << " particle_res=" << sd.particle_energy_residual
       << " field_res=" << sd.field_energy_residual
       << " dE=" << sd.delta_total_energy << "\n";
    if (!(std::isfinite(gamma_ref) && gamma_ref > 0.0 && d0.Bz_y_mode_abs > 0.0 &&
          d1.gauss_rms < 5.0e-9 && d1.gauge_rms < 5.0e-9 &&
          sd.A_chain_abs_rms < 5.0e-9 &&
          std::abs(sd.deposit_gather_work_residual) < 5.0e-9 &&
          std::abs(sd.particle_energy_residual) < 5.0e-9 &&
          std::abs(sd.field_energy_residual) < 5.0e-9 &&
          std::abs(sd.delta_total_energy) < 5.0e-9)) {
        os << "FAILED: coupled Weibel one-step conservation identities failed\n";
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Regression test: selective particle fallback layout
//
// VALIDATION ONLY: called when run_unit_tests=true.
//
// Inputs:
//   os : std::ostream&; test report stream
//
// Output:
//   bool: true when the stated numerical regression passes; writes measured residuals.
//
// Dependencies:
//   compute_diagnostics, compute_step_diagnostics, gauss_legendre_01,
//   initialize_weibel_filamentation_3d, make_grid,
//   nonlinear_step_selective_particle_fallback, nonlinear_step_uniform_subcycled.
//   C++17 standard library (headers listed at the top of this file).
// ---------------------------------------------------------------------------
static bool test_selective_particle_fallback_layout(std::ostream& os) {
    Config cfg;
    cfg.test_case = "cold_relativistic_weibel";
    cfg.nx = cfg.ny = cfg.nz = 4;
    cfg.Lx = cfg.Ly = cfg.Lz = 2.0 * PI;
    cfg.dt = 1.0;
    cfg.n_steps = 1;
    cfg.particles_per_cell_pair = 1;
    cfg.spline_order = 1;
    cfg.orbit_quad_order = 8;
    cfg.v0 = 0.9;
    cfg.weibel_B0 = 0.2;
    cfg.weibel_eigenmode = false;
    cfg.nonlinear_rtol = 1.0e-9;
    cfg.nonlinear_atol = 1.0e-11;
    cfg.nonlinear_max_iter = 32;
    cfg.anderson_memory = 4;
    cfg.anderson_start = 2;
    cfg.anderson_damping = 0.5;
    cfg.uniform_local_rtol = 2.0e-3;
    cfg.uniform_local_atol = 1.0e-10;
    cfg.uniform_local_max_iter = 1;
    cfg.uniform_local_damping = 1.0;
    cfg.selective_particle_max_substeps = 64;
    cfg.selective_particle_dynamic_chunk = 2;
    cfg.adaptive_dt = false;

    const Grid g = make_grid(cfg);
    const Quadrature quad = gauss_legendre_01(cfg.orbit_quad_order);
    const State s0 = initialize_weibel_filamentation_3d(cfg, g);
    std::vector<NonlinearHistoryEntry> hist;
    NonlinearSolveInfo direct_info;
    direct_info.iterations = 7;
    NonlinearSolveInfo info;
    State s1;
    std::ostringstream failure_diagnostics;
    try {
        s1 = nonlinear_step_selective_particle_fallback(
            s0, g, cfg, quad, hist, &info, 0.0, direct_info,
            "forced unit-test direct failure", &failure_diagnostics);
    } catch (const std::exception& e) {
        os << "FAILED: selective fallback threw unexpectedly: "
           << e.what() << "\n";
        return false;
    }

    const StepDiagnostics sd = compute_step_diagnostics(
        s0, s1, g, cfg.dt, cfg.spline_order, quad, cfg.sigma1,
        cfg.sigma2, cfg.kappa, cfg.split_orbit_at_knots);
    const Diagnostics d1 = compute_diagnostics(
        s1, g, cfg.sigma1, cfg.sigma2, cfg.kappa,
        cfg.perturbation_mode);
    const bool offsets_ok = s1.subcycle_offset_last.size() == s0.x.size() + 1 &&
        !s1.subcycle_offset_last.empty() &&
        s1.subcycle_offset_last.back() == s1.subcycle_x0_last.size() &&
        s1.subcycle_x0_last.size() == s1.subcycle_vbar_last.size();
    const bool failure_diagnostics_ok =
        failure_diagnostics.str().find("maximum_iterations") !=
        std::string::npos;
    os << std::scientific << std::setprecision(6)
       << "Selective particle fallback unit test: layout_restarts="
       << info.selective_refinement_restarts
       << " refined_particles=" << info.selective_refined_particles
       << " max_substeps=" << info.uniform_substeps_used
       << " work_ratio=" << info.selective_work_ratio
       << " gauss=" << d1.gauss_rms
       << " deposit_gather=" << sd.deposit_gather_work_residual
       << " dE=" << sd.delta_total_energy << "\n";
    const bool controller_ok = info.selective_fallback_used &&
          info.direct_attempt_iterations == direct_info.iterations &&
          info.selective_refinement_restarts > 0 &&
          info.selective_refined_particles > 0 &&
          info.uniform_substeps_used > 1 &&
          info.uniform_substeps_used <= cfg.selective_particle_max_substeps &&
          info.selective_total_particle_substeps ==
              s1.subcycle_x0_last.size() &&
          offsets_ok && failure_diagnostics_ok && d1.gauss_rms < 5.0e-9 &&
          std::abs(sd.deposit_gather_work_residual) < 5.0e-9;

    // A separate tight solve checks conservation for a deliberately mixed
    // fixed layout.  The loose one-iteration controller case above is only a
    // deterministic way to force several layout restarts and is not expected
    // to have a small particle fixed-point energy defect.
    Config accurate_cfg = cfg;
    accurate_cfg.dt = 0.02;
    accurate_cfg.weibel_B0 = 1.0e-4;
    accurate_cfg.uniform_local_rtol = 1.0e-12;
    accurate_cfg.uniform_local_atol = 1.0e-14;
    accurate_cfg.uniform_local_max_iter = 64;
    const Grid accurate_grid = make_grid(accurate_cfg);
    const Quadrature accurate_quad =
        gauss_legendre_01(accurate_cfg.orbit_quad_order);
    const State a0 = initialize_weibel_filamentation_3d(
        accurate_cfg, accurate_grid);
    std::vector<int> mixed_layout(a0.x.size(), 1);
    if (!mixed_layout.empty()) mixed_layout[0] = 4;
    if (mixed_layout.size() > 1) mixed_layout[1] = 2;
    std::vector<NonlinearHistoryEntry> accurate_hist;
    NonlinearSolveInfo accurate_info;
    State a1;
    try {
        a1 = nonlinear_step_uniform_subcycled(
            a0, accurate_grid, accurate_cfg, accurate_quad,
            accurate_hist, &accurate_info, 0.0, &mixed_layout);
    } catch (const std::exception& e) {
        os << "FAILED: tight mixed-layout solve threw unexpectedly: "
           << e.what() << "\n";
        return false;
    }
    const StepDiagnostics accurate_sd = compute_step_diagnostics(
        a0, a1, accurate_grid, accurate_cfg.dt,
        accurate_cfg.spline_order, accurate_quad, accurate_cfg.sigma1,
        accurate_cfg.sigma2, accurate_cfg.kappa,
        accurate_cfg.split_orbit_at_knots);
    const std::size_t expected_substeps = a0.x.size() + 4;
    const bool accurate_ok = accurate_info.selective_fallback_used &&
        accurate_info.selective_refined_particles == 2 &&
        accurate_info.selective_total_particle_substeps == expected_substeps &&
        a1.subcycle_offset_last.back() == expected_substeps &&
        std::abs(accurate_sd.deposit_gather_work_residual) < 5.0e-9 &&
        std::abs(accurate_sd.particle_energy_residual) < 5.0e-9 &&
        std::abs(accurate_sd.field_energy_residual) < 5.0e-9 &&
        std::abs(accurate_sd.delta_total_energy) < 5.0e-9;
    os << "Selective mixed-layout conservation test: refined_particles="
       << accurate_info.selective_refined_particles
       << " total_substeps="
       << accurate_info.selective_total_particle_substeps
       << " deposit_gather="
       << accurate_sd.deposit_gather_work_residual
       << " particle_res=" << accurate_sd.particle_energy_residual
       << " field_res=" << accurate_sd.field_energy_residual
       << " dE=" << accurate_sd.delta_total_energy << "\n";

    if (!(controller_ok && accurate_ok)) {
        os << "FAILED: selective layout/current replacement regression failed\n";
        return false;
    }
    return true;
}


// -----------------------------------------------------------------------------
// Output helpers
// -----------------------------------------------------------------------------

// All-particle mass-weighted moments, independent of the bounded plot sample.
// ---------------------------------------------------------------------------
// Write all-particle mass-weighted velocity means, RMS, and variances
//
// Inputs:
//   out : std::ostream&; output stream or result populated in place (see type)
//   st : const State&; accepted particle and mesh state
//   step : int; accepted field-step index
//   time : Real; accepted endpoint time
//
// Output:
//   None; writes records to the supplied stream/file.
//
// Dependencies:
//   C++17 standard library (headers listed at the top of this file).
// ---------------------------------------------------------------------------
static void write_transverse_moments(std::ostream& out, const State& st,
                                     int step, Real time) {
    long double mass = 0.0L;
    std::array<long double, 3> mean{}, square{}, variance{};
    for (std::size_t p = 0; p < st.v.size(); ++p) {
        mass += st.m[p];
        for (int d = 0; d < 3; ++d) {
            mean[d] += st.m[p] * static_cast<long double>(st.v[p][d]);
            square[d] += st.m[p] * static_cast<long double>(st.v[p][d]) * st.v[p][d];
        }
    }
    if (!(mass > 0.0L)) throw std::runtime_error("nonpositive mass in velocity moments");
    for (int d = 0; d < 3; ++d) mean[d] /= mass;
    for (std::size_t p = 0; p < st.v.size(); ++p)
        for (int d = 0; d < 3; ++d) {
            const long double dv = st.v[p][d] - mean[d];
            variance[d] += st.m[p] * dv * dv;
        }
    out << step << ',' << std::setprecision(17) << time << ',' << st.v.size() << ',' << mass;
    for (int d = 0; d < 3; ++d) out << ',' << mean[d];
    for (int d = 0; d < 3; ++d) out << ',' << std::sqrt(square[d] / mass);
    for (int d = 0; d < 3; ++d) out << ',' << variance[d] / mass;
    out << '\n';
}

// Fold +/- Fourier indices without dropping conjugates: summing all output
// bins gives the real-space mean square (Parseval), component by component.
// Use raw FFT indices here: derivative symbols intentionally zero Nyquist.
// ---------------------------------------------------------------------------
// Write folded Fourier-mode powers without dropping conjugate partners
//
// Inputs:
//   out : std::ostream&; output stream or result populated in place (see type)
//   st : const State&; accepted particle and mesh state
//   g : const Grid&; periodic grid geometry and Fourier symbols
//   step : int; accepted field-step index
//   time : Real; accepted endpoint time
//
// Output:
//   None; writes records to the supplied stream/file.
//
// Dependencies:
//   compute_E, fft_real, spectral_curl.
//   C++17 standard library (headers listed at the top of this file).
// ---------------------------------------------------------------------------
static void write_transverse_spectrum(std::ostream& out, const State& st,
                                      const Grid& g, int step, Real time) {
    const VecField E = compute_E(st, g);
    const VecField B = spectral_curl(st.A, g);
    const int sx = g.nx / 2 + 1, sy = g.ny / 2 + 1, sz = g.nz / 2 + 1;
    std::vector<std::array<Real, 6>> power(sx * sy * sz);
    const Real scale = 1.0 / (static_cast<Real>(g.N) * g.N);
    for (int d = 0; d < 6; ++d) {
        const CField h = fft_real(d < 3 ? E[d] : B[d - 3], g);
        for (int i = 0; i < g.nx; ++i)
            for (int j = 0; j < g.ny; ++j)
                for (int k = 0; k < g.nz; ++k) {
                    const int a = std::min(i, g.nx - i);
                    const int b = std::min(j, g.ny - j);
                    const int c = std::min(k, g.nz - k);
                    power[(a * sy + b) * sz + c][d] += std::norm(h[g.index(i,j,k)]) * scale;
                }
    }
    for (int a = 0; a < sx; ++a)
        for (int b = 0; b < sy; ++b)
            for (int c = 0; c < sz; ++c) {
                const char* kind = (a == 0 && b == 0 && c == 0) ? "zero" :
                    (b == 0 && c == 0) ? "longitudinal" : a == 0 ? "transverse" : "oblique";
                out << step << ',' << std::setprecision(17) << time << ',' << a << ',' << b << ',' << c
                    << ',' << kind;
                for (Real value : power[(a * sy + b) * sz + c]) out << ',' << value;
                out << '\n';
            }
}

// ---------------------------------------------------------------------------
// Write a bounded deterministic sample of the final particle state
//
// Inputs:
//   path : const std::string&; input/output filename
//   st : const State&; accepted particle and mesh state
//   max_particles : std::size_t; max particles
//
// Output:
//   None; writes records to the supplied stream/file.
//
// Dependencies:
//   C++17 standard library (headers listed at the top of this file).
// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
// Write every particle at an accepted endpoint for a requested snapshot time
//
// Inputs:
//   cfg : const Config&; simulation settings and solver tolerances
//   st : const State&; accepted particle and mesh state
//   step : int; accepted field-step index
//   time : Real; accepted endpoint time
//   requested_time : Real; desired physical snapshot time
//
// Output:
//   None; writes records to the supplied stream/file.
//
// Dependencies:
//   C++17 standard library (headers listed at the top of this file).
// ---------------------------------------------------------------------------
static void write_particle_snapshot(const Config& cfg, const State& st,
                                    int step, Real time, Real requested_time) {
    std::ostringstream name;
    name << cfg.output_prefix << "_particles_t" << std::setprecision(17)
         << requested_time << ".csv";
    std::ofstream out(name.str());
    if (!out) throw std::runtime_error("could not open particle snapshot: " + name.str());
    out << "step,time,requested_time,particle,x,y,z,vx,vy,vz,q,m\n";
    out << std::setprecision(17);
    for (std::size_t p = 0; p < st.x.size(); ++p) {
        out << step << ',' << time << ',' << requested_time << ',' << p;
        for (Real x : st.x[p]) out << ',' << x;
        for (Real v : st.v[p]) out << ',' << v;
        out << ',' << st.q[p] << ',' << st.m[p] << '\n';
    }
    out.close();
    if (!out) throw std::runtime_error("failed writing particle snapshot: " + name.str());
    std::cout << "particle snapshot = " << name.str() << " (accepted time "
              << std::setprecision(17) << time << ", particles " << st.x.size() << ")\n";
}

// ---------------------------------------------------------------------------
// Field probe axis index
//
// Inputs:
//   cfg : const Config&; simulation settings and solver tolerances
//
// Output:
//   int: field probe axis index.
//
// Dependencies:
//   is_weibel_case, lowercase.
//   C++17 standard library (headers listed at the top of this file).
// ---------------------------------------------------------------------------
static int field_probe_axis_index(const Config& cfg) {
    const std::string axis = lowercase(cfg.field_probe_axis);
    if (axis == "x") return 0;
    if (axis == "y") return 1;
    if (axis == "z") return 2;
    return is_weibel_case(cfg) ? 1 : 0;
}

static const char* axis_label(int axis) {
    return axis == 0 ? "x" : (axis == 1 ? "y" : "z");
}

// ---------------------------------------------------------------------------
// Curl from shape gradient
//
// Inputs:
//   gradA : const Mat3&; gradA
//
// Output:
//   Vec3: curl from shape gradient.
//
// Dependencies:
//   Local scalar/vector types; no external library.
// ---------------------------------------------------------------------------
static Vec3 curl_from_shape_gradient(const Mat3& gradA) {
    // gradA[component][direction] = d A_component / d x_direction.
    return Vec3{
        gradA[2][1] - gradA[1][2],
        gradA[0][2] - gradA[2][0],
        gradA[1][0] - gradA[0][1]
    };
}

// ---------------------------------------------------------------------------
// Matrix difference norm
//
// Inputs:
//   a : const Mat3&; a
//   b : const Mat3&; b
//
// Output:
//   Real: matrix difference norm.
//
// Dependencies:
//   C++17 standard library (headers listed at the top of this file).
// ---------------------------------------------------------------------------
static Real matrix_difference_norm(const Mat3& a, const Mat3& b) {
    long double sum = 0.0L;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            const long double d = static_cast<long double>(a[i][j]) -
                                  static_cast<long double>(b[i][j]);
            sum += d * d;
        }
    }
    return std::sqrt(static_cast<Real>(sum));
}

struct ReconstructedFieldProbe {
    Real phi_h = 0.0;
    Real psi_h = 0.0;
    Real rho_h = 0.0;
    Vec3 A_h{0.0, 0.0, 0.0};
    Vec3 U_h{0.0, 0.0, 0.0};
    Vec3 J_h{0.0, 0.0, 0.0};
    Vec3 E_grid_gather{0.0, 0.0, 0.0};
    Vec3 E_shape_phi{0.0, 0.0, 0.0};
    Vec3 B_grid_gather{0.0, 0.0, 0.0};
    Vec3 B_shape_A{0.0, 0.0, 0.0};
    Mat3 gradA{};
};

// Two reconstructions are deliberately recorded.  "grid_gather" first takes
// the spectral mesh derivative and then applies the particle shape S_p.  The
// "shape" version differentiates the spline interpolant itself:
//
//   E_grid_gather = S_p[-grad_spec(phi)-U],
//   E_shape_phi   = -grad(S_p phi)-S_p U,
//   B_grid_gather = S_p[curl_spec(A)],
//   B_shape_A     = curl(S_p A).
//
// B_shape_A is the relevant continuity diagnostic for the canonical pusher,
// whose magnetic skew term is built from grad(S_p A).  Degree 1 permits jumps
// in this derivative; degree 2 makes it C0.
// ---------------------------------------------------------------------------
// Compare gathered mesh fields with shape-derived field reconstructions
//
// Inputs:
//   st : const State&; accepted particle and mesh state
//   g : const Grid&; periodic grid geometry and Fourier symbols
//   spline_order : int; spline order
//   E_grid : const VecField&; E grid
//   B_grid : const VecField&; B grid
//   pos : const Vec3&; particle/probe position(s)
//
// Output:
//   ReconstructedFieldProbe: compare gathered mesh fields with shape-derived field reconstructions.
//
// Dependencies:
//   add3, curl_from_shape_gradient, gather_scalar_at, gather_shape_gradient_scalar_at,
//   gather_shape_gradient_vector_interp_at, gather_vector_at, mul3.
// ---------------------------------------------------------------------------
static ReconstructedFieldProbe evaluate_reconstructed_field_probe(
        const State& st, const Grid& g, int spline_order, const VecField& E_grid,
        const VecField& B_grid, const Vec3& pos) {
    ReconstructedFieldProbe p;
    p.phi_h = gather_scalar_at(st.phi, pos, g, spline_order);
    p.psi_h = gather_scalar_at(st.psi, pos, g, spline_order);
    p.rho_h = gather_scalar_at(st.rho, pos, g, spline_order);
    p.A_h = gather_vector_at(st.A, pos, g, spline_order);
    p.U_h = gather_vector_at(st.U, pos, g, spline_order);
    p.J_h = gather_vector_at(st.J, pos, g, spline_order);
    p.E_grid_gather = gather_vector_at(E_grid, pos, g, spline_order);
    p.B_grid_gather = gather_vector_at(B_grid, pos, g, spline_order);
    const Vec3 grad_phi_shape =
        gather_shape_gradient_scalar_at(st.phi, pos, g, spline_order);
    p.E_shape_phi = mul3(-1.0, add3(grad_phi_shape, p.U_h));
    p.gradA = gather_shape_gradient_vector_interp_at(
        st.A, st.A, 0.0, pos, g, spline_order);
    p.B_shape_A = curl_from_shape_gradient(p.gradA);
    return p;
}

// ---------------------------------------------------------------------------
// Write field line probe header
//
// Inputs:
//   out : std::ostream&; output stream or result populated in place (see type)
//
// Output:
//   None; writes records to the supplied stream/file.
//
// Dependencies:
//   Local scalar/vector types; no external library.
// ---------------------------------------------------------------------------
static void write_field_line_probe_header(std::ostream& out) {
    out << "step,time,spline_order,probe_axis,sample,coordinate,x,y,z,"
        << "phi_h,psi_h,rho_h,Ax_h,Ay_h,Az_h,Ux_h,Uy_h,Uz_h,Jx_h,Jy_h,Jz_h,"
        << "Ex_grid_gather,Ey_grid_gather,Ez_grid_gather,"
        << "Ex_shape_phi,Ey_shape_phi,Ez_shape_phi,"
        << "Bx_grid_gather,By_grid_gather,Bz_grid_gather,"
        << "Bx_shape_A,By_shape_A,Bz_shape_A,"
        << "dAx_dx,dAx_dy,dAx_dz,dAy_dx,dAy_dy,dAy_dz,dAz_dx,dAz_dy,dAz_dz,"
        << "E_representation_difference,B_representation_difference\n";
}

// ---------------------------------------------------------------------------
// Write field knot jump header
//
// Inputs:
//   out : std::ostream&; output stream or result populated in place (see type)
//
// Output:
//   None; writes records to the supplied stream/file.
//
// Dependencies:
//   Local scalar/vector types; no external library.
// ---------------------------------------------------------------------------
static void write_field_knot_jump_header(std::ostream& out) {
    out << "step,time,spline_order,probe_axis,knot,knot_coordinate,epsilon,"
        << "A_jump,E_grid_gather_jump,E_shape_phi_jump,"
        << "B_grid_gather_jump,B_shape_A_jump,gradA_jump,"
        << "Bx_shape_left,By_shape_left,Bz_shape_left,"
        << "Bx_shape_right,By_shape_right,Bz_shape_right,"
        << "Ex_shape_left,Ey_shape_left,Ez_shape_left,"
        << "Ex_shape_right,Ey_shape_right,Ez_shape_right\n";
}

// ---------------------------------------------------------------------------
// Write line probes and two-sided finite-offset spline-knot differences
//
// Inputs:
//   line_out : std::ostream&; line out
//   jump_out : std::ostream&; jump out
//   st : const State&; accepted particle and mesh state
//   g : const Grid&; periodic grid geometry and Fourier symbols
//   cfg : const Config&; simulation settings and solver tolerances
//   step : int; accepted field-step index
//   time : Real; accepted endpoint time
//
// Output:
//   None; writes records to the supplied stream/file.
//
// Dependencies:
//   compute_E, evaluate_reconstructed_field_probe, field_probe_axis_index,
//   matrix_difference_norm, norm3, spectral_curl, sub3.
//   C++17 standard library (headers listed at the top of this file).
// ---------------------------------------------------------------------------
static void write_field_line_and_knot_snapshot(
        std::ostream& line_out, std::ostream& jump_out, const State& st,
        const Grid& g, const Config& cfg, int step, Real time) {
    const int axis = field_probe_axis_index(cfg);
    const int n_axis = axis == 0 ? g.nx : (axis == 1 ? g.ny : g.nz);
    const int n_samples = n_axis * cfg.field_probe_points_per_cell;
    const Real L_axis = g.L[axis];
    const Real dx_axis = g.dx[axis];
    const VecField E_grid = compute_E(st, g);
    const VecField B_grid = spectral_curl(st.A, g);

    for (int sample = 0; sample < n_samples; ++sample) {
        Vec3 pos{0.0, 0.0, 0.0};
        const Real coordinate = -0.5 * L_axis +
            (static_cast<Real>(sample) + 0.5) * L_axis /
                static_cast<Real>(n_samples);
        pos[axis] = coordinate;
        const ReconstructedFieldProbe p = evaluate_reconstructed_field_probe(
            st, g, cfg.spline_order, E_grid, B_grid, pos);
        line_out << step << "," << std::setprecision(17) << time << ","
                 << cfg.spline_order << "," << axis_label(axis) << ","
                 << sample << "," << coordinate << ","
                 << pos[0] << "," << pos[1] << "," << pos[2] << ","
                 << p.phi_h << "," << p.psi_h << "," << p.rho_h << ","
                 << p.A_h[0] << "," << p.A_h[1] << "," << p.A_h[2] << ","
                 << p.U_h[0] << "," << p.U_h[1] << "," << p.U_h[2] << ","
                 << p.J_h[0] << "," << p.J_h[1] << "," << p.J_h[2] << ","
                 << p.E_grid_gather[0] << "," << p.E_grid_gather[1] << ","
                 << p.E_grid_gather[2] << ","
                 << p.E_shape_phi[0] << "," << p.E_shape_phi[1] << ","
                 << p.E_shape_phi[2] << ","
                 << p.B_grid_gather[0] << "," << p.B_grid_gather[1] << ","
                 << p.B_grid_gather[2] << ","
                 << p.B_shape_A[0] << "," << p.B_shape_A[1] << ","
                 << p.B_shape_A[2] << ",";
        for (int comp = 0; comp < 3; ++comp) {
            for (int direction = 0; direction < 3; ++direction) {
                line_out << p.gradA[comp][direction] << ",";
            }
        }
        line_out << norm3(sub3(p.E_shape_phi, p.E_grid_gather)) << ","
                 << norm3(sub3(p.B_shape_A, p.B_grid_gather)) << "\n";
    }

    // Measure the actual left/right jump at every knot of the selected
    // centered B-spline.  For degree 1, grad(A_h) and curl(A_h) can have a
    // finite jump.  For degree 2, those jumps should decrease with epsilon.
    const Real offset = 0.5 * static_cast<Real>(cfg.spline_order - 1);
    const Real epsilon = 1.0e-8 * dx_axis;
    for (int knot = 0; knot < n_axis; ++knot) {
        Real u_knot = std::fmod(static_cast<Real>(knot) + offset,
                               static_cast<Real>(n_axis));
        if (u_knot < 0.0) u_knot += static_cast<Real>(n_axis);
        const Real coordinate = -0.5 * L_axis + u_knot * dx_axis;
        Vec3 left{0.0, 0.0, 0.0};
        Vec3 right{0.0, 0.0, 0.0};
        left[axis] = coordinate - epsilon;
        right[axis] = coordinate + epsilon;
        const ReconstructedFieldProbe pl = evaluate_reconstructed_field_probe(
            st, g, cfg.spline_order, E_grid, B_grid, left);
        const ReconstructedFieldProbe pr = evaluate_reconstructed_field_probe(
            st, g, cfg.spline_order, E_grid, B_grid, right);
        jump_out << step << "," << std::setprecision(17) << time << ","
                 << cfg.spline_order << "," << axis_label(axis) << ","
                 << knot << "," << coordinate << "," << epsilon << ","
                 << norm3(sub3(pr.A_h, pl.A_h)) << ","
                 << norm3(sub3(pr.E_grid_gather, pl.E_grid_gather)) << ","
                 << norm3(sub3(pr.E_shape_phi, pl.E_shape_phi)) << ","
                 << norm3(sub3(pr.B_grid_gather, pl.B_grid_gather)) << ","
                 << norm3(sub3(pr.B_shape_A, pl.B_shape_A)) << ","
                 << matrix_difference_norm(pr.gradA, pl.gradA) << ","
                 << pl.B_shape_A[0] << "," << pl.B_shape_A[1] << ","
                 << pl.B_shape_A[2] << ","
                 << pr.B_shape_A[0] << "," << pr.B_shape_A[1] << ","
                 << pr.B_shape_A[2] << ","
                 << pl.E_shape_phi[0] << "," << pl.E_shape_phi[1] << ","
                 << pl.E_shape_phi[2] << ","
                 << pr.E_shape_phi[0] << "," << pr.E_shape_phi[1] << ","
                 << pr.E_shape_phi[2] << "\n";
    }
}

// ---------------------------------------------------------------------------
// Write field grid header
//
// Inputs:
//   out : std::ostream&; output stream or result populated in place (see type)
//
// Output:
//   None; writes records to the supplied stream/file.
//
// Dependencies:
//   Local scalar/vector types; no external library.
// ---------------------------------------------------------------------------
static void write_field_grid_header(std::ostream& out) {
    out << "step,time,spline_order,ix,iy,iz,x,y,z,phi,psi,rho,"
        << "Ax,Ay,Az,Ux,Uy,Uz,Jx,Jy,Jz,Ex,Ey,Ez,Bx,By,Bz\n";
}

// ---------------------------------------------------------------------------
// Write field grid snapshot
//
// Inputs:
//   out : std::ostream&; output stream or result populated in place (see type)
//   st : const State&; accepted particle and mesh state
//   g : const Grid&; periodic grid geometry and Fourier symbols
//   cfg : const Config&; simulation settings and solver tolerances
//   step : int; accepted field-step index
//   time : Real; accepted endpoint time
//
// Output:
//   None; writes records to the supplied stream/file.
//
// Dependencies:
//   compute_E, spectral_curl.
//   C++17 standard library (headers listed at the top of this file).
// ---------------------------------------------------------------------------
static void write_field_grid_snapshot(std::ostream& out, const State& st,
                                      const Grid& g, const Config& cfg,
                                      int step, Real time) {
    const VecField E = compute_E(st, g);
    const VecField B = spectral_curl(st.A, g);
    for (int i = 0; i < g.nx; ++i) {
        for (int j = 0; j < g.ny; ++j) {
            for (int k = 0; k < g.nz; ++k) {
                const int id = g.index(i, j, k);
                const Real x = -0.5 * g.L[0] + static_cast<Real>(i) * g.dx[0];
                const Real y = -0.5 * g.L[1] + static_cast<Real>(j) * g.dx[1];
                const Real z = -0.5 * g.L[2] + static_cast<Real>(k) * g.dx[2];
                out << step << "," << std::setprecision(17) << time << ","
                    << cfg.spline_order << "," << i << "," << j << "," << k << ","
                    << x << "," << y << "," << z << ","
                    << st.phi[id] << "," << st.psi[id] << "," << st.rho[id] << ","
                    << st.A[0][id] << "," << st.A[1][id] << "," << st.A[2][id] << ","
                    << st.U[0][id] << "," << st.U[1][id] << "," << st.U[2][id] << ","
                    << st.J[0][id] << "," << st.J[1][id] << "," << st.J[2][id] << ","
                    << E[0][id] << "," << E[1][id] << "," << E[2][id] << ","
                    << B[0][id] << "," << B[1][id] << "," << B[2][id] << "\n";
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Write effective settings and derived quantities for reproducibility
//
// Inputs:
//   path : const std::string&; input/output filename
//   cfg : const Config&; simulation settings and solver tolerances
//   g : const Grid&; periodic grid geometry and Fourier symbols
//   np : std::size_t; total particle count
//
// Output:
//   None; writes records to the supplied stream/file.
//
// Dependencies:
//   effective_adaptive_dt_min, landau_density_amplitude, selected_theory_growth_rate,
//   weibel_seed_B0.
//   C++17 standard library (headers listed at the top of this file).
// ---------------------------------------------------------------------------
static void write_config_echo(const std::string& path, const Config& cfg, const Grid& g, std::size_t np) {
    std::ofstream out(path);
    out << "test_case=" << cfg.test_case << "\n";
    out << "transverse_diagnostics=" << (cfg.transverse_diagnostics ? "true" : "false") << "\n";
    out << "transverse_spectrum_interval=" << std::setprecision(17) << cfg.transverse_spectrum_interval << "\n";
    out << "tsi_transverse_seed=" << cfg.tsi_transverse_seed << "\n";
    out << "tsi_transverse_mode=" << cfg.tsi_transverse_mode << "\n";
    out << "nyquist_projection=" << cfg.nyquist_projection << "\n";
    out << "nx=" << cfg.nx << "\n";
    out << "ny=" << cfg.ny << "\n";
    out << "nz=" << cfg.nz << "\n";
    out << "Lx=" << std::setprecision(17) << cfg.Lx << "\n";
    out << "Ly=" << std::setprecision(17) << cfg.Ly << "\n";
    out << "Lz=" << std::setprecision(17) << cfg.Lz << "\n";
    out << "n_steps=" << cfg.n_steps << "\n";
    out << "dt=" << std::setprecision(17) << cfg.dt << "\n";
    // SG added: n_steps and nominal dt define the unchanged target time.
    out << "target_time=" << (static_cast<Real>(cfg.n_steps) * cfg.dt) << "\n";
    out << "particles=" << np << "\n";
    out << "particles_per_cell_pair_or_velocity_samples=" << cfg.particles_per_cell_pair << "\n";
    out << "spline_order=" << cfg.spline_order << "\n";
    out << "orbit_quad_order=" << cfg.orbit_quad_order << "\n";
    out << "split_orbit_at_knots=" << (cfg.split_orbit_at_knots ? "true" : "false") << "\n";
    out << "macro_time_step=" << (cfg.adaptive_dt ? "adaptive_rejection" : "fixed") << "\n";
    out << "adaptive_dt=" << (cfg.adaptive_dt ? "true" : "false") << "\n";
    out << "adaptive_dt_min=" << effective_adaptive_dt_min(cfg) << "\n";
    out << "adaptive_dt_shrink=" << cfg.adaptive_dt_shrink << "\n";
    out << "adaptive_dt_growth=" << cfg.adaptive_dt_growth << "\n";
    out << "adaptive_dt_growth_interval=" << cfg.adaptive_dt_growth_interval << "\n";
    out << "adaptive_max_attempts=" << cfg.adaptive_max_attempts << "\n";
    out << "nonlinear_solver=" << cfg.nonlinear_solver << "\n";
    // SG added: echo both parts of the mixed nonlinear tolerance.
    out << "nonlinear_rtol=" << cfg.nonlinear_rtol << "\n";
    out << "nonlinear_atol=" << cfg.nonlinear_atol << "\n";
    out << "nonlinear_max_iter=" << cfg.nonlinear_max_iter << "\n";
    out << "anderson_memory=" << cfg.anderson_memory << "\n";
    out << "anderson_start=" << cfg.anderson_start << "\n";
    out << "anderson_damping=" << cfg.anderson_damping << "\n";
    out << "anderson_regularization=" << cfg.anderson_regularization << "\n";
    out << "anderson_growth_limit=" << cfg.anderson_growth_limit << "\n";
    out << "uniform_particle_subcycling="
        << (cfg.uniform_particle_subcycling ? "true" : "false") << "\n";
    out << "uniform_refine_on_failure="
        << (cfg.uniform_refine_on_failure ? "true" : "false") << "\n";
    out << "uniform_initial_substeps=" << cfg.uniform_initial_substeps << "\n";
    out << "uniform_substeps=" << cfg.uniform_substeps << "\n";
    out << "uniform_max_substeps=" << cfg.uniform_max_substeps << "\n";
    out << "uniform_subcycling_start_time="
        << cfg.uniform_subcycling_start_time << "\n";
    out << "particle_local_rtol=" << cfg.uniform_local_rtol << "\n";
    out << "particle_local_atol=" << cfg.uniform_local_atol << "\n";
    out << "particle_local_max_iter=" << cfg.uniform_local_max_iter << "\n";
    out << "uniform_local_rtol=" << cfg.uniform_local_rtol << "\n";
    out << "uniform_local_atol=" << cfg.uniform_local_atol << "\n";
    out << "uniform_local_max_iter=" << cfg.uniform_local_max_iter << "\n";
    out << "uniform_local_damping=" << cfg.uniform_local_damping << "\n";
    out << "selective_particle_fallback="
        << (cfg.selective_particle_fallback ? "true" : "false") << "\n";
    out << "selective_particle_max_substeps="
        << cfg.selective_particle_max_substeps << "\n";
    out << "selective_particle_dynamic_chunk="
        << cfg.selective_particle_dynamic_chunk << "\n";
    out << "selective_failure_diagnostics="
        << (cfg.selective_failure_diagnostics ? "true" : "false") << "\n";
    out << "selective_failure_diagnostic_limit="
        << cfg.selective_failure_diagnostic_limit << "\n";
    out << "particle_snapshot_times=";
    for (Real t : cfg.particle_snapshot_times) out << t << ' ';
    out << "\n";
    out << "field_diagnostics="
        << (cfg.field_diagnostics ? "true" : "false") << "\n";
    out << "field_diagnostics_interval="
        << cfg.field_diagnostics_interval << "\n";
    out << "field_probe_points_per_cell="
        << cfg.field_probe_points_per_cell << "\n";
    out << "field_probe_axis=" << cfg.field_probe_axis << "\n";
    out << "field_grid_snapshots="
        << (cfg.field_grid_snapshots ? "true" : "false") << "\n";
    out << "kappa=" << cfg.kappa << "\n";
    out << "sigma1=" << cfg.sigma1 << "\n";
    out << "sigma2=" << cfg.sigma2 << "\n";
    out << "sigma1_minus_kappa2_sigma2=" << (cfg.sigma1 - cfg.kappa * cfg.kappa * cfg.sigma2) << "\n";
    out << "rho0=" << cfg.n0 << "\n";
    out << "v0=" << cfg.v0 << "\n";
    out << "perturbation=" << cfg.perturbation << "\n";
    out << "two_stream_density_alpha=" << cfg.two_stream_density_alpha << "\n";
    out << "two_stream_eigenmode=" << (cfg.two_stream_eigenmode ? "true" : "false") << "\n";
    out << "two_stream_finite_grid_theory=" << (cfg.two_stream_finite_grid_theory ? "true" : "false") << "\n";
    out << "weibel_B0=" << weibel_seed_B0(cfg) << "\n";
    out << "weibel_eigenmode=" << (cfg.weibel_eigenmode ? "true" : "false") << "\n";
    out << "weibel_finite_grid_theory=" << (cfg.weibel_finite_grid_theory ? "true" : "false") << "\n";
    out << "landau_alpha=" << landau_density_amplitude(cfg) << "\n";
    out << "thermal_velocity=" << cfg.thermal_velocity << "\n";
    out << "perturbation_mode=" << cfg.perturbation_mode << "\n";
    out << "theory_omega=" << cfg.theory_omega << "\n";
    out << "theory_gamma=" << cfg.theory_gamma << "\n";
    out << "auto_theory_growth_rate=" << selected_theory_growth_rate(cfg, g) << "\n";
    out << "dV=" << g.dV << "\n";
}

// -----------------------------------------------------------------------------
// Main driver
// -----------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Run input validation, optional regression tests, fixed-step evolution, and CSV output
//
// Inputs:
//   argc : int; number of command-line arguments
//   argv : char**; executable name and input-file path
//
// Output:
//   int: zero on success; nonzero on invalid input, failed tests, or failed solve.
//
// Dependencies:
//   compute_diagnostics, compute_step_diagnostics, effective_adaptive_dt_min,
//   field_probe_axis_index, format_wall_duration, gauss_legendre_01, initialize_state,
//   is_weibel_case, make_grid, nonlinear_step_orbit, read_config,
//   selected_theory_growth_rate, test_RA_orbit_chain_rule,
//   test_RA_orbit_chain_rule_with_crossings, test_canonical_no_At_equivalence,
//   test_field_continuity_gauss_nyquist_regression, test_initial_gauss_law,
//   test_instability_initialization_and_theory, test_one_step_gauss_law_and_RA,
//   test_orbit_deposit_gather_adjointness, test_relativistic_hc_static_B,
//   test_relativistic_static_E_secant_work, test_selective_particle_fallback_layout,
//   test_skew_K_to_Beff_sign, test_small_multistep_energy_conservation,
//   test_spectral_nyquist_operator_compatibility, test_two_stream_eigenmode_growth,
//   test_weibel_one_step_gauss_RA_and_energy, wall_seconds_since, write_config_echo,
//   write_field_grid_header, write_field_grid_snapshot, write_field_knot_jump_header,
//   write_field_line_and_knot_snapshot, write_field_line_probe_header,
//   write_particle_sample, write_particle_snapshot, write_transverse_moments,
//   write_transverse_spectrum.
//   C++17 standard library (headers listed at the top of this file).
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    const WallClock::time_point program_start = WallClock::now();
    try {
        if (argc < 2) {
            std::cerr << "usage: " << argv[0] << " input.txt\n";
            return 2;
        }
        const Config cfg = read_config(argv[1]);
        if (cfg.adaptive_dt || cfg.uniform_particle_subcycling)
            throw std::runtime_error("manuscript runs require adaptive_dt=false, uniform_particle_subcycling=false");
        {
            std::ofstream inner(cfg.output_prefix + "_particle_inner_history.csv");
            if (!inner) throw std::runtime_error("cannot open particle inner history");
            inner << "time_n,dt,map_evaluation,particles_converged,failed_particles,max_scaled_particle_residual,local_iterations_mean,local_iterations_max\n";
        }
        std::cout << "Nested particle solve: frozen-field inner iterations, then outer current Anderson; "
                  << (cfg.selective_particle_fallback ? "selective refinement on local failure\n" : "full dt, no temporal substeps\n");
        const Grid g = make_grid(cfg);
        const Quadrature quad = gauss_legendre_01(cfg.orbit_quad_order);

        std::cout << "OpenMP teaching code: serial-reference GM-HC mathematics with parallel kernels\n";
        std::cout << "test_case = " << cfg.test_case << " (shared-memory OpenMP, no MPI)\n";
#ifdef _OPENMP
        std::cout << "OpenMP max threads = " << omp_get_max_threads() << "\n";
#else
        std::cout << "OpenMP not enabled at compile time; running serial fallback pragmas\n";
#endif
        std::cout << "grid = " << cfg.nx << " x " << cfg.ny << " x " << cfg.nz
                  << ", dt = " << cfg.dt << ", steps = " << cfg.n_steps << "\n";
        std::cout << "wall timer = steady_clock; elapsed times include setup, solves, diagnostics, and output\n";
        if (cfg.uniform_particle_subcycling) {
            std::cout << "uniform particle subcycling: outer grid-current Anderson, ";
            if (cfg.uniform_refine_on_failure) {
                std::cout << cfg.uniform_initial_substeps
                          << " local step(s) initially; after the first failed macro solve retry with "
                          << cfg.uniform_substeps << " and retain it, then refine to "
                          << cfg.uniform_max_substeps << " if needed";
            } else if (cfg.uniform_subcycling_start_time > 0.0) {
                std::cout << cfg.uniform_initial_substeps << " local step(s) before t="
                          << cfg.uniform_subcycling_start_time << ", then ";
                std::cout << cfg.uniform_substeps << " local steps per macro step"
                          << " (refine to at most " << cfg.uniform_max_substeps << ")";
            } else {
                std::cout << cfg.uniform_substeps << " local steps per macro step"
                          << " (refine to at most " << cfg.uniform_max_substeps << ")";
            }
            std::cout << ", local_max_iter=" << cfg.uniform_local_max_iter << "\n";
        }
        if (cfg.selective_particle_fallback) {
            std::cout << "direct-first selective particle fallback: fixed macro dt; "
                      << "failed particles refine through powers of two to dt/"
                      << cfg.selective_particle_max_substeps
                      << ", outer grid-current Anderson restarts after every "
                         "layout change, local_max_iter="
                      << cfg.uniform_local_max_iter << "\n";
        }

        std::ofstream unit_log(cfg.output_prefix + "_unit_tests.txt");
        bool tests_ok = true;
        if (cfg.run_unit_tests) {
            std::ostringstream results;
            tests_ok = test_spectral_nyquist_operator_compatibility(results) && tests_ok;
            tests_ok = test_field_continuity_gauss_nyquist_regression(results) && tests_ok;
            tests_ok = test_relativistic_hc_static_B(results) && tests_ok;
            tests_ok = test_skew_K_to_Beff_sign(results) && tests_ok;
            tests_ok = test_relativistic_static_E_secant_work(results) && tests_ok;
            tests_ok = test_canonical_no_At_equivalence(results) && tests_ok;
            tests_ok = test_orbit_deposit_gather_adjointness(results) && tests_ok;
            tests_ok = test_RA_orbit_chain_rule(results) && tests_ok;
            tests_ok = test_RA_orbit_chain_rule_with_crossings(results) && tests_ok;
            tests_ok = test_initial_gauss_law(results) && tests_ok;
            tests_ok = test_one_step_gauss_law_and_RA(results) && tests_ok;
            tests_ok = test_small_multistep_energy_conservation(results) && tests_ok;
            tests_ok = test_instability_initialization_and_theory(results) && tests_ok;
            tests_ok = test_two_stream_eigenmode_growth(results) && tests_ok;
            tests_ok = test_weibel_one_step_gauss_RA_and_energy(results) && tests_ok;
            tests_ok = test_selective_particle_fallback_layout(results) && tests_ok;
            std::cout << results.str();
            unit_log << results.str();
        }
        if (!tests_ok) {
            std::cerr << "one or more unit tests failed\n";
            return 3;
        }
        if (cfg.run_unit_tests) std::cout << "Unit tests passed.\n";

        State state = initialize_state(cfg, g);
        write_config_echo(cfg.output_prefix + "_config_echo.txt", cfg, g, state.x.size());

        const Diagnostics initial_diag = compute_diagnostics(state, g, cfg.sigma1, cfg.sigma2,
                                                              cfg.kappa, cfg.perturbation_mode);
        const Real theory_growth = selected_theory_growth_rate(cfg, g);
        const Real theory_E0 = initial_diag.E_mode_abs;
        const Real theory_B0 = initial_diag.Bz_y_mode_abs;
        const bool use_B_theory = is_weibel_case(cfg);
        const Real theory_amp0 = use_B_theory ? theory_B0 : theory_E0;
        const bool has_theory_reference =
            (std::abs(theory_growth) > 0.0 || std::abs(cfg.theory_omega) > 0.0) && theory_amp0 > 0.0;
        if (std::abs(theory_growth) > 0.0) {
            std::cout << std::scientific << std::setprecision(6)
                      << "linear theory growth/damping rate = " << theory_growth
                      << (use_B_theory ? " using |B_z(k_y)|" : " using |E_x(k_x)|") << "\n";
            if (!has_theory_reference) {
                std::cout << "initial diagnostic mode is zero; use tools/summarize_diagnostics.py "
                          << "with a fit window to compare the measured slope to theory\n";
            }
        }

        std::ofstream diag(cfg.output_prefix + "_diagnostics.csv");
        diag << "step,time,gauss_rms,gauss_max,divE_rms,sigma1_rho_rms,"
             << "gauss_normalization_rms,gauss_relative,"
             << "gauge_rms,gauge_max,psi_over_kappa2_rms,divA_rms,"
             << "gauge_normalization_rms,gauge_relative,"
             << "field_energy,kinetic_energy,total_energy,"
             << "rho_mode,E_mode_re,E_mode_im,E_mode_abs,Bz_y_mode_re,Bz_y_mode_im,Bz_y_mode_abs,"
             << "theory_growth_rate,theory_mode_abs,mode_over_theory,mean_rho,"
             << "nonlinear_iterations,nonlinear_converged,nonlinear_rms_residual,"
             << "nonlinear_max_particle_residual,nonlinear_absolute_rms_residual,"
             << "anderson_restarts,anderson_steps,picard_fallback_steps,convergence_method,"
             << "observed_convergence_rate,estimated_map_contraction,backtracked_steps,minimum_update_alpha,"
             << "local_particle_iterations_mean,local_particle_iterations_max,uniform_substeps_used,"
             << "selective_fallback_used,direct_attempt_iterations,selective_refinement_restarts,"
             << "selective_refined_particles,selective_total_particle_substeps,selective_work_ratio,"
             << "selective_n1,selective_n2,selective_n4,selective_n8,selective_n16,selective_n32,selective_n64,"
             << "crossing_particles,avg_path_segments,max_path_segments,dt,elapsed_wall_seconds\n";

        std::ofstream stepcsv(cfg.output_prefix + "_step_diagnostics.csv");
        stepcsv << "step,time,nonlinear_iterations,nonlinear_rms_residual,nonlinear_max_particle_residual,"
                << "anderson_restarts,convergence_method,observed_convergence_rate,"
                << "estimated_map_contraction,backtracked_steps,minimum_update_alpha,"
                << "local_particle_iterations_mean,local_particle_iterations_max,uniform_substeps_used,"
                << "selective_fallback_used,direct_attempt_iterations,selective_refinement_restarts,"
                << "selective_refined_particles,selective_total_particle_substeps,selective_work_ratio,"
                << "selective_n1,selective_n2,selective_n4,selective_n8,selective_n16,selective_n32,selective_n64,"
                << "A_chain_abs_rms,A_chain_rel_rms,A_chain_max_norm,"
                << "A_chain_mean_norm,A_chain_work_defect,delta_K,delta_W,delta_total_energy,particle_work,"
                << "grid_work,deposit_gather_work_residual,particle_energy_residual,field_energy_residual,"
                << "crossing_particles,total_crossings,avg_path_segments,max_path_segments,dt,elapsed_wall_seconds\n";

        // SG added: every rejected and accepted nonlinear attempt is recorded.
        std::ofstream adaptivecsv(cfg.output_prefix + "_adaptive_diagnostics.csv");
        adaptivecsv << "accepted_step,time_before,attempt,dt,outcome,nonlinear_iterations,"
                    << "nonlinear_rms_residual,nonlinear_max_particle_residual,"
                    << "observed_convergence_rate,estimated_map_contraction,"
                    << "backtracked_steps,minimum_update_alpha,"
                    << "local_particle_iterations_mean,local_particle_iterations_max,uniform_substeps_used,"
                    << "selective_fallback_used,direct_attempt_iterations,selective_refinement_restarts,"
                    << "selective_refined_particles,selective_total_particle_substeps,selective_work_ratio,"
                    << "selective_n1,selective_n2,selective_n4,selective_n8,selective_n16,selective_n32,selective_n64,"
                    << "elapsed_wall_seconds\n";

        std::ofstream failed_particle_csv;
        std::ostream* failed_particle_diagnostics = nullptr;
        if (cfg.selective_failure_diagnostics) {
            const std::string failure_path =
                cfg.output_prefix + "_failed_particle_diagnostics.csv";
            failed_particle_csv.open(failure_path);
            if (!failed_particle_csv) {
                throw std::runtime_error(
                    "could not open failed-particle diagnostics file: " +
                    failure_path);
            }
            failed_particle_csv
                << "time_before,layout_refinement_restart,map_evaluation,event_exhausted,"
                << "failed_particles_in_event,particle_index,assigned_substeps,failed_substep_zero_based,"
                << "theta0,theta1,h,local_iterations,failure_reason,first_scaled_residual,"
                << "final_scaled_residual,final_absolute_residual,q,m,"
                << "x0_x,x0_y,x0_z,P0_x,P0_y,P0_z,"
                << "initial_guess_x,initial_guess_y,initial_guess_z,"
                << "last_guess_x,last_guess_y,last_guess_z,"
                << "last_map_x,last_map_y,last_map_z\n";
            failed_particle_csv.flush();
            failed_particle_diagnostics = &failed_particle_csv;
            std::cout << "failed-particle diagnostics = " << failure_path
                      << " (up to "
                      << cfg.selective_failure_diagnostic_limit
                      << " worst particles per refinement event; the worst "
                         "exhausted particle is always retained)\n";
        }

        std::ofstream field_line_csv;
        std::ofstream field_jump_csv;
        std::ofstream field_grid_csv;
        if (cfg.field_diagnostics) {
            const std::string line_path =
                cfg.output_prefix + "_field_line_probe.csv";
            const std::string jump_path =
                cfg.output_prefix + "_field_knot_jumps.csv";
            field_line_csv.open(line_path);
            field_jump_csv.open(jump_path);
            if (!field_line_csv || !field_jump_csv) {
                throw std::runtime_error(
                    "could not open field reconstruction diagnostics files");
            }
            write_field_line_probe_header(field_line_csv);
            write_field_knot_jump_header(field_jump_csv);
            std::cout << "field reconstruction diagnostics = " << line_path
                      << " and " << jump_path << " (axis="
                      << axis_label(field_probe_axis_index(cfg))
                      << ", samples_per_cell="
                      << cfg.field_probe_points_per_cell << ")\n";
            if (cfg.field_grid_snapshots) {
                const std::string grid_path =
                    cfg.output_prefix + "_field_grid_snapshots.csv";
                field_grid_csv.open(grid_path);
                if (!field_grid_csv) {
                    throw std::runtime_error(
                        "could not open full-grid field diagnostics file: " +
                        grid_path);
                }
                write_field_grid_header(field_grid_csv);
                std::cout << "full-grid field snapshots = " << grid_path << "\n";
            }
        }

        std::ofstream transverse_moments, transverse_spectrum;
        Real next_transverse_spectrum_time = 0.0;
        if (cfg.transverse_diagnostics) {
            transverse_moments.open(cfg.output_prefix + "_velocity_moments.csv");
            transverse_spectrum.open(cfg.output_prefix + "_field_spectrum.csv");
            if (!transverse_moments || !transverse_spectrum)
                throw std::runtime_error("cannot open transverse diagnostic files");
            transverse_moments << "step,time,particles,total_mass,mean_vx,mean_vy,mean_vz,rms_vx,rms_vy,rms_vz,var_vx,var_vy,var_vz\n";
            transverse_spectrum << "step,time,abs_mx,abs_my,abs_mz,wavevector_class,Ex_power,Ey_power,Ez_power,Bx_power,By_power,Bz_power\n";
        }
        auto write_diag = [&](int step, Real time, const NonlinearSolveInfo& ninfo, const PathStats& ps,
                              const Diagnostics& d, Real step_dt) {
            if (cfg.transverse_diagnostics) {
                write_transverse_moments(transverse_moments, state, step, time);
                const Real end = cfg.n_steps * cfg.dt;
                const Real eps = 128.0 * std::numeric_limits<Real>::epsilon() * std::max(1.0, end);
                if (time + eps >= next_transverse_spectrum_time || time + eps >= end) {
                    write_transverse_spectrum(transverse_spectrum, state, g, step, time);
                    next_transverse_spectrum_time =
                        (std::floor((time + eps) / cfg.transverse_spectrum_interval) + 1.0) *
                        cfg.transverse_spectrum_interval;
                    transverse_spectrum.flush();
                }
                transverse_moments.flush();
            }
            Real theory_mode_abs = std::numeric_limits<Real>::quiet_NaN();
            Real mode_over_theory = std::numeric_limits<Real>::quiet_NaN();
            const Real measured_mode_abs = use_B_theory ? d.Bz_y_mode_abs : d.E_mode_abs;
            if (has_theory_reference) {
                theory_mode_abs = theory_amp0 * std::exp(theory_growth * time);
                mode_over_theory = measured_mode_abs / (theory_mode_abs + 1.0e-300);
            }
            diag << step << "," << std::setprecision(17) << time << ","
                 << d.gauss_rms << "," << d.gauss_max << ","
                 << d.divE_rms << "," << d.sigma1_rho_rms << ","
                 << d.gauss_normalization_rms << "," << d.gauss_relative << ","
                 << d.gauge_rms << "," << d.gauge_max << ","
                 << d.psi_over_kappa2_rms << "," << d.divA_rms << ","
                 << d.gauge_normalization_rms << "," << d.gauge_relative << ","
                 << d.field_energy << "," << d.kinetic_energy << "," << d.total_energy << ","
                 << d.rho_mode << "," << d.E_mode_re << "," << d.E_mode_im << ","
                 << d.E_mode_abs << "," << d.Bz_y_mode_re << "," << d.Bz_y_mode_im << ","
                 << d.Bz_y_mode_abs << "," << theory_growth << "," << theory_mode_abs << ","
                 << mode_over_theory << "," << d.mean_rho << ","
                 << ninfo.iterations << "," << (ninfo.converged ? 1 : 0) << ","
                 << ninfo.final_rms_residual << "," << ninfo.final_max_particle_residual << ","
                 << ninfo.final_absolute_rms_residual << "," << ninfo.anderson_restarts << ","
                 << ninfo.anderson_steps << "," << ninfo.picard_fallback_steps << ","
                 << ninfo.convergence_method << ","
                 << ninfo.observed_convergence_rate << ","
                 << ninfo.estimated_map_contraction << ","
                 << ninfo.backtracked_steps << "," << ninfo.minimum_update_alpha << ","
                 << ninfo.local_particle_iterations_mean << ","
                 << ninfo.local_particle_iterations_max << ","
                 << ninfo.uniform_substeps_used << ","
                 << (ninfo.selective_fallback_used ? 1 : 0) << ","
                 << ninfo.direct_attempt_iterations << ","
                 << ninfo.selective_refinement_restarts << ","
                 << ninfo.selective_refined_particles << ","
                 << ninfo.selective_total_particle_substeps << ","
                 << ninfo.selective_work_ratio << ","
                 << ninfo.selective_level_counts[0] << ","
                 << ninfo.selective_level_counts[1] << ","
                 << ninfo.selective_level_counts[2] << ","
                 << ninfo.selective_level_counts[3] << ","
                 << ninfo.selective_level_counts[4] << ","
                 << ninfo.selective_level_counts[5] << ","
                 << ninfo.selective_level_counts[6] << ","
                 << ps.crossing_particles << ","
                 << (ps.n_particles ? static_cast<Real>(ps.total_segments) /
                                      static_cast<Real>(ps.n_particles) : 0.0) << ","
                 << ps.max_segments << "," << step_dt << ","
                 << wall_seconds_since(program_start) << "\n";
        };

        NonlinearSolveInfo initial_nonlinear_info;
        initial_nonlinear_info.converged = true;
        initial_nonlinear_info.iterations = 0;
        initial_nonlinear_info.final_rms_residual = std::numeric_limits<Real>::quiet_NaN();
        initial_nonlinear_info.final_max_particle_residual = std::numeric_limits<Real>::quiet_NaN();
        initial_nonlinear_info.final_absolute_rms_residual = std::numeric_limits<Real>::quiet_NaN();
        initial_nonlinear_info.convergence_method = "initial_state";
        write_diag(0, 0.0, initial_nonlinear_info, PathStats{}, initial_diag, 0.0);
        Diagnostics latest_diag = initial_diag;

        // SG added: n_steps retains its input-deck meaning by defining the same
        // target time, while accepted_step may increase after dt is reduced.
        const Real target_time = static_cast<Real>(cfg.n_steps) * cfg.dt;
        const Real dt_min = effective_adaptive_dt_min(cfg);
        const Real time_epsilon =
            128.0 * std::numeric_limits<Real>::epsilon() *
            std::max<Real>(1.0, std::abs(target_time));
        Real time = 0.0;
        Real last_field_diagnostic_time =
            -std::numeric_limits<Real>::infinity();
        Real next_field_diagnostic_time =
            cfg.field_diagnostics_interval > 0.0
                ? cfg.field_diagnostics_interval
                : std::numeric_limits<Real>::infinity();
        if (cfg.field_diagnostics) {
            write_field_line_and_knot_snapshot(
                field_line_csv, field_jump_csv, state, g, cfg, 0, 0.0);
            if (cfg.field_grid_snapshots) {
                write_field_grid_snapshot(field_grid_csv, state, g, cfg, 0, 0.0);
            }
            field_line_csv.flush();
            field_jump_csv.flush();
            if (cfg.field_grid_snapshots) field_grid_csv.flush();
            last_field_diagnostic_time = 0.0;
        }
        std::size_t next_particle_snapshot = 0;
        auto save_due_particle_snapshots = [&](int step) {
            while (next_particle_snapshot < cfg.particle_snapshot_times.size() &&
                   cfg.particle_snapshot_times[next_particle_snapshot] <= time + time_epsilon) {
                write_particle_snapshot(cfg, state, step, time,
                    cfg.particle_snapshot_times[next_particle_snapshot++]);
            }
        };
        for (Real t : cfg.particle_snapshot_times) {
            if (t > target_time + time_epsilon)
                std::cerr << "warning: particle snapshot time " << t
                          << " exceeds end time " << target_time << "\n";
        }
        save_due_particle_snapshots(0);
        Real current_dt = cfg.dt;
        Real next_progress_time = (target_time > 0.0) ? target_time / 5.0 : 0.0;
        const Real progress_interval = next_progress_time;
        int accepted_step = 0;
        // NOT NEEDED FOR MANUSCRIPT: adaptive-step state retained for reference.
        // SG added: hysteresis state for cautious dt regrowth.
        int consecutive_easy_steps = 0;
        int growth_cooldown_remaining = 0;
        int failed_growth_attempts = 0;
        bool growth_trial_pending = false;

        std::cout << std::scientific << std::setprecision(3)
                  << (cfg.adaptive_dt ? "adaptive rejection" : "fixed stepping")
                  << ": nominal dt=" << cfg.dt
                  << ", dt_min=" << dt_min
                  << ", target_time=" << target_time
                  << ", nonlinear_solver=" << cfg.nonlinear_solver << "\n";

        while (time + time_epsilon < target_time) {
            // Avoid copying O(Np*uniform_substeps) accepted-path diagnostics.
            // They have already been reported and are not inputs to the next
            // solve; state_n remains the unchanged retry anchor for this step.
            State state_n = std::move(state);
            std::vector<std::size_t>().swap(state_n.subcycle_offset_last);
            std::vector<Vec3>().swap(state_n.subcycle_x0_last);
            std::vector<Vec3>().swap(state_n.subcycle_vbar_last);
            const Real remaining_time = target_time - time;
            Real dt_try = std::min(current_dt, remaining_time);
            int attempt = 1;
            std::vector<NonlinearHistoryEntry> hist;
            NonlinearSolveInfo ninfo;
            State accepted_state;

            for (;;) {
                Config trial_cfg = cfg;
                trial_cfg.dt = dt_try;
                hist.clear();
                ninfo = NonlinearSolveInfo{};
                try {
                    accepted_state =
                        nonlinear_step_orbit(
                            state_n, g, trial_cfg, quad, hist, &ninfo, time,
                            failed_particle_diagnostics);
                    adaptivecsv << (accepted_step + 1) << "," << std::setprecision(17)
                                << time << "," << attempt << "," << dt_try << ",accepted,"
                                << ninfo.iterations << "," << ninfo.final_rms_residual << ","
                                << ninfo.final_max_particle_residual << ","
                                << ninfo.observed_convergence_rate << ","
                                << ninfo.estimated_map_contraction << ","
                                << ninfo.backtracked_steps << ","
                                << ninfo.minimum_update_alpha << ","
                                << ninfo.local_particle_iterations_mean << ","
                                << ninfo.local_particle_iterations_max << ","
                                << ninfo.uniform_substeps_used << ","
                                << (ninfo.selective_fallback_used ? 1 : 0) << ","
                                << ninfo.direct_attempt_iterations << ","
                                << ninfo.selective_refinement_restarts << ","
                                << ninfo.selective_refined_particles << ","
                                << ninfo.selective_total_particle_substeps << ","
                                << ninfo.selective_work_ratio << ","
                                << ninfo.selective_level_counts[0] << ","
                                << ninfo.selective_level_counts[1] << ","
                                << ninfo.selective_level_counts[2] << ","
                                << ninfo.selective_level_counts[3] << ","
                                << ninfo.selective_level_counts[4] << ","
                                << ninfo.selective_level_counts[5] << ","
                                << ninfo.selective_level_counts[6] << ","
                                << wall_seconds_since(program_start) << "\n";
                    adaptivecsv.flush();
                    break;
                } catch (const NonlinearSolveFailure& e) {
                    adaptivecsv << (accepted_step + 1) << "," << std::setprecision(17)
                                << time << "," << attempt << "," << dt_try << ",rejected,"
                                << ninfo.iterations << "," << ninfo.final_rms_residual << ","
                                << ninfo.final_max_particle_residual << ","
                                << ninfo.observed_convergence_rate << ","
                                << ninfo.estimated_map_contraction << ","
                                << ninfo.backtracked_steps << ","
                                << ninfo.minimum_update_alpha << ","
                                << ninfo.local_particle_iterations_mean << ","
                                << ninfo.local_particle_iterations_max << ","
                                << ninfo.uniform_substeps_used << ","
                                << (ninfo.selective_fallback_used ? 1 : 0) << ","
                                << ninfo.direct_attempt_iterations << ","
                                << ninfo.selective_refinement_restarts << ","
                                << ninfo.selective_refined_particles << ","
                                << ninfo.selective_total_particle_substeps << ","
                                << ninfo.selective_work_ratio << ","
                                << ninfo.selective_level_counts[0] << ","
                                << ninfo.selective_level_counts[1] << ","
                                << ninfo.selective_level_counts[2] << ","
                                << ninfo.selective_level_counts[3] << ","
                                << ninfo.selective_level_counts[4] << ","
                                << ninfo.selective_level_counts[5] << ","
                                << ninfo.selective_level_counts[6] << ","
                                << wall_seconds_since(program_start) << "\n";
                    adaptivecsv.flush();

                    if (!cfg.adaptive_dt) {
                        std::ostringstream msg;
                        msg << "fixed step " << (accepted_step + 1)
                            << " failed at t=" << time << ": " << e.what();
                        throw std::runtime_error(msg.str());
                    }

                    // SG added: a failed attempt to regrow dt increases the
                    // waiting time exponentially before the next growth probe.
                    if (attempt == 1) {
                        if (growth_trial_pending) ++failed_growth_attempts;
                        const int cooldown_multiplier =
                            1 << std::min(failed_growth_attempts, 4);
                        growth_cooldown_remaining =
                            cfg.adaptive_dt_growth_interval * cooldown_multiplier;
                        consecutive_easy_steps = 0;
                        growth_trial_pending = false;
                    }
                    if (attempt >= cfg.adaptive_max_attempts) {
                        std::ostringstream msg;
                        msg << "adaptive step failed at t=" << time
                            << " after " << attempt << " attempts; last dt=" << dt_try
                            << ": " << e.what();
                        throw std::runtime_error(msg.str());
                    }

                    const Real reduced_dt = dt_try * cfg.adaptive_dt_shrink;
                    if (reduced_dt < dt_min * (1.0 - 64.0 * std::numeric_limits<Real>::epsilon())) {
                        std::ostringstream msg;
                        msg << "adaptive step failed at t=" << time
                            << ": reducing dt=" << dt_try << " by "
                            << cfg.adaptive_dt_shrink << " would go below dt_min="
                            << dt_min << "; last failure: " << e.what();
                        throw std::runtime_error(msg.str());
                    }

                    const Real next_dt = std::max(dt_min, reduced_dt);
                    std::cerr << std::scientific << std::setprecision(6)
                              << "adaptive rejection at t=" << time
                              << ": dt=" << dt_try << " failed (" << e.what()
                              << "); retrying unchanged state with dt=" << next_dt << "\n";
                    dt_try = next_dt;
                    ++attempt;
                }
            }

            const bool accepted_growth_trial =
                growth_trial_pending && attempt == 1;
            state = std::move(accepted_state);
            time += dt_try;
            if (std::abs(time - target_time) <= time_epsilon) time = target_time;
            current_dt = dt_try;
            growth_trial_pending = false;
            if (accepted_growth_trial &&
                current_dt >= cfg.dt * (1.0 - 64.0 * std::numeric_limits<Real>::epsilon())) {
                // SG added: a successful return to nominal dt clears growth
                // failure history and restores the shortest cooldown.
                failed_growth_attempts = 0;
            }
            ++accepted_step;
            save_due_particle_snapshots(accepted_step);

            const StepDiagnostics sd = compute_step_diagnostics(state_n, state, g, dt_try,
                                                                 cfg.spline_order, quad, cfg.sigma1,
                                                                 cfg.sigma2, cfg.kappa,
                                                                 cfg.split_orbit_at_knots);
            PathStats step_path_stats;
            step_path_stats.n_particles = state_n.x.size();
            step_path_stats.crossing_particles = sd.crossing_particles;
            // For a mixed subcycle layout, total_crossings counts only mesh-knot
            // crossings and therefore does not include the extra substep
            // boundaries.  Recover the already computed total segment count
            // from the subcycle-aware average instead of assuming one path per
            // particle.
            step_path_stats.total_segments = static_cast<std::size_t>(std::llround(
                sd.average_path_segments * static_cast<Real>(state_n.x.size())));
            step_path_stats.max_segments = sd.max_path_segments;
            const Diagnostics step_diag = compute_diagnostics(state, g, cfg.sigma1, cfg.sigma2,
                                                               cfg.kappa, cfg.perturbation_mode);
            write_diag(accepted_step, time, ninfo, step_path_stats, step_diag, dt_try);
            latest_diag = step_diag;
            stepcsv << accepted_step << "," << std::setprecision(17) << time << ","
                    << ninfo.iterations << "," << ninfo.final_rms_residual << ","
                    << ninfo.final_max_particle_residual << "," << ninfo.anderson_restarts << ","
                    << ninfo.convergence_method << ","
                    << ninfo.observed_convergence_rate << ","
                    << ninfo.estimated_map_contraction << ","
                    << ninfo.backtracked_steps << "," << ninfo.minimum_update_alpha << ","
                    << ninfo.local_particle_iterations_mean << ","
                    << ninfo.local_particle_iterations_max << ","
                    << ninfo.uniform_substeps_used << ","
                    << (ninfo.selective_fallback_used ? 1 : 0) << ","
                    << ninfo.direct_attempt_iterations << ","
                    << ninfo.selective_refinement_restarts << ","
                    << ninfo.selective_refined_particles << ","
                    << ninfo.selective_total_particle_substeps << ","
                    << ninfo.selective_work_ratio << ","
                    << ninfo.selective_level_counts[0] << ","
                    << ninfo.selective_level_counts[1] << ","
                    << ninfo.selective_level_counts[2] << ","
                    << ninfo.selective_level_counts[3] << ","
                    << ninfo.selective_level_counts[4] << ","
                    << ninfo.selective_level_counts[5] << ","
                    << ninfo.selective_level_counts[6] << ","
                    << sd.A_chain_abs_rms << ","
                    << sd.A_chain_rel_rms << "," << sd.A_chain_max_norm << ","
                    << sd.A_chain_mean_norm << "," << sd.A_chain_work_defect << ","
                    << sd.delta_K << "," << sd.delta_W << "," << sd.delta_total_energy << ","
                    << sd.particle_work << "," << sd.grid_work << ","
                    << sd.deposit_gather_work_residual << "," << sd.particle_energy_residual << ","
                    << sd.field_energy_residual << "," << sd.crossing_particles << ","
                    << sd.total_crossings << "," << sd.average_path_segments << ","
                    << sd.max_path_segments << "," << dt_try << ","
                    << wall_seconds_since(program_start) << "\n";
            diag.flush();
            stepcsv.flush();

            const bool final_field_state =
                time + time_epsilon >= target_time;
            const bool interval_field_state =
                cfg.field_diagnostics_interval > 0.0 &&
                time + time_epsilon >= next_field_diagnostic_time;
            if (cfg.field_diagnostics &&
                (final_field_state || interval_field_state) &&
                std::abs(time - last_field_diagnostic_time) > time_epsilon) {
                write_field_line_and_knot_snapshot(
                    field_line_csv, field_jump_csv, state, g, cfg,
                    accepted_step, time);
                if (cfg.field_grid_snapshots) {
                    write_field_grid_snapshot(
                        field_grid_csv, state, g, cfg, accepted_step, time);
                }
                field_line_csv.flush();
                field_jump_csv.flush();
                if (cfg.field_grid_snapshots) field_grid_csv.flush();
                last_field_diagnostic_time = time;
                while (next_field_diagnostic_time <= time + time_epsilon) {
                    next_field_diagnostic_time += cfg.field_diagnostics_interval;
                }
            }

            // SG added: solver-work controller. These thresholds use nonlinear
            // convergence behavior, not a time-discretization error estimate.
            constexpr Real easy_iteration_fraction = 0.25;
            constexpr Real stiff_iteration_fraction = 0.75;
            constexpr Real easy_rate_limit = 0.50;
            constexpr Real stiff_rate_limit = 0.90;
            constexpr Real strong_backtracking_alpha = 0.125;

            const Real iteration_fraction =
                static_cast<Real>(ninfo.iterations) /
                static_cast<Real>(cfg.nonlinear_max_iter);
            const bool finite_rate =
                std::isfinite(ninfo.observed_convergence_rate);
            const bool easy_solve =
                iteration_fraction <= easy_iteration_fraction &&
                ninfo.anderson_restarts <= 1 &&
                ninfo.backtracked_steps == 0 &&
                finite_rate &&
                ninfo.observed_convergence_rate <= easy_rate_limit;
            const bool stiff_solve =
                iteration_fraction >= stiff_iteration_fraction ||
                (finite_rate &&
                 ninfo.observed_convergence_rate >= stiff_rate_limit) ||
                (ninfo.backtracked_steps > 0 &&
                 ninfo.minimum_update_alpha <= strong_backtracking_alpha);
            // Lsec is reported as a contractivity diagnostic but is not a hard
            // controller condition: Anderson can solve some maps with Lsec>1.

            if (growth_cooldown_remaining > 0) {
                --growth_cooldown_remaining;
            }
            consecutive_easy_steps =
                easy_solve ? (consecutive_easy_steps + 1) : 0;

            if (cfg.adaptive_dt && time + time_epsilon < target_time &&
                stiff_solve) {
                const Real reduced_next_dt =
                    std::max(dt_min, current_dt * cfg.adaptive_dt_shrink);
                if (reduced_next_dt <
                    current_dt * (1.0 - 64.0 * std::numeric_limits<Real>::epsilon())) {
                    std::cout << std::scientific << std::setprecision(6)
                              << "adaptive proactive reduction after t=" << time
                              << ": dt " << current_dt << " -> " << reduced_next_dt
                              << " (iteration_fraction=" << iteration_fraction
                              << ", rho=" << ninfo.observed_convergence_rate
                              << ", Lsec=" << ninfo.estimated_map_contraction
                              << ", min_alpha=" << ninfo.minimum_update_alpha << ")\n";
                    current_dt = reduced_next_dt;
                    consecutive_easy_steps = 0;
                    growth_cooldown_remaining =
                        std::max(growth_cooldown_remaining,
                                 cfg.adaptive_dt_growth_interval);
                    growth_trial_pending = false;
                }
            } else if (cfg.adaptive_dt &&
                       time + time_epsilon < target_time &&
                       current_dt <
                           cfg.dt * (1.0 - 64.0 * std::numeric_limits<Real>::epsilon()) &&
                       consecutive_easy_steps >= cfg.adaptive_dt_growth_interval &&
                       growth_cooldown_remaining == 0) {
                const Real grown_dt =
                    std::min(cfg.dt, current_dt * cfg.adaptive_dt_growth);
                if (grown_dt >
                    current_dt * (1.0 + 64.0 * std::numeric_limits<Real>::epsilon())) {
                    std::cout << std::scientific << std::setprecision(6)
                              << "adaptive growth after t=" << time
                              << ": dt " << current_dt << " -> " << grown_dt
                              << " after " << consecutive_easy_steps
                              << " easy accepted steps"
                              << " (rho=" << ninfo.observed_convergence_rate
                              << ", Lsec=" << ninfo.estimated_map_contraction << ")\n";
                    current_dt = grown_dt;
                    consecutive_easy_steps = 0;
                    growth_trial_pending = true;
                }
            }

            const bool reached_progress =
                progress_interval > 0.0 &&
                (time + time_epsilon >= next_progress_time ||
                 time + time_epsilon >= target_time);
            if (reached_progress) {
                std::cout << std::scientific << std::setprecision(3)
                          << "accepted_step " << std::setw(5) << accepted_step
                          << " t=" << time << "/" << target_time << " dt=" << dt_try
                          << " Gauss=" << step_diag.gauss_rms
                          << " R_A=" << sd.A_chain_abs_rms
                          << " R_work=" << sd.deposit_gather_work_residual
                          << " dE=" << sd.delta_total_energy
                          << " Rnl=" << ninfo.final_rms_residual
                          << " Rnl_max=" << ninfo.final_max_particle_residual
                          << " restarts=" << ninfo.anderson_restarts
                          << " method=" << ninfo.convergence_method
                          << " rho=" << ninfo.observed_convergence_rate
                          << " Lsec=" << ninfo.estimated_map_contraction
                          << " bt=" << ninfo.backtracked_steps
                              << " crossings=" << sd.total_crossings
                              << " nonlinear_iters=" << ninfo.iterations
                              << " wall="
                              << format_wall_duration(
                                     wall_seconds_since(program_start));
                if (cfg.uniform_particle_subcycling ||
                    ninfo.selective_fallback_used) {
                    std::cout << " local_iters_mean="
                              << ninfo.local_particle_iterations_mean
                              << " local_iters_max="
                              << ninfo.local_particle_iterations_max
                              << " substeps=" << ninfo.uniform_substeps_used;
                }
                if (ninfo.selective_fallback_used) {
                    std::cout << " selective_refined="
                              << ninfo.selective_refined_particles
                              << " selective_work="
                              << ninfo.selective_work_ratio
                              << " layout_restarts="
                              << ninfo.selective_refinement_restarts;
                }
                std::cout << "\n";
                while (next_progress_time <= time + time_epsilon &&
                       next_progress_time < target_time) {
                    next_progress_time += progress_interval;
                }
            }
        }

        write_particle_sample(cfg.output_prefix + "_particle_sample.csv", state, 20000);
        const Diagnostics& fd = latest_diag;
        const Real total_wall_seconds = wall_seconds_since(program_start);
        std::cout << std::scientific << std::setprecision(6)
                  << "Final Gauss RMS = " << fd.gauss_rms << "\n"
                  << "Final relative Gauss residual = " << fd.gauss_relative << "\n"
                  << "Final gauge RMS = " << fd.gauge_rms << "\n"
                  << "Final relative gauge residual = " << fd.gauge_relative << "\n"
                  << "Final total energy = " << fd.total_energy << "\n"
                  << "Total wall time = "
                  << format_wall_duration(total_wall_seconds)
                  << " (" << total_wall_seconds << " s)\n";
        return 0;
    } catch (const std::exception& e) {
        const Real failed_wall_seconds = wall_seconds_since(program_start);
        std::cerr << "error: " << e.what() << "\n"
                  << "Elapsed wall time before failure = "
                  << format_wall_duration(failed_wall_seconds)
                  << " (" << failed_wall_seconds << " s)\n";
        return 1;
    }
}
