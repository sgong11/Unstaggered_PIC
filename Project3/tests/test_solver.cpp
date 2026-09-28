// Standalone regressions for manuscript-specific dispatch and projection.
// Build through validate.py; the included program main is renamed here.
#define main pic_program_main
#include "../ec_pic_nyquist_projected.cpp"
#undef main

// ---------------------------------------------------------------------------
// Check the projector's algebra and its off switch on mixed Fourier modes
// Inputs: none (deterministic 4^3 grid and nonzero Nyquist content)
// Output: bool, identity/off, idempotence, self-adjointness, and removal pass
// Dependencies: make_grid, project_nyquist; standard C++ math
// ---------------------------------------------------------------------------
static bool check_projection() {
    Config cfg;
    cfg.nx = cfg.ny = cfg.nz = 4;
    Grid g = make_grid(cfg);
    Field a(g.N), b(g.N);
    for (int i=0; i<g.nx; ++i)
        for (int j=0; j<g.ny; ++j)
            for (int k=0; k<g.nz; ++k) {
                int id=g.index(i,j,k);
                a[id]=std::sin(2*PI*i/g.nx)+0.7*(i%2 ? -1:1)+0.4*(j%2 ? -1:1);
                b[id]=std::cos(2*PI*j/g.ny)+0.3*(k%2 ? -1:1)+0.2*a[id];
            }
    Field off=a; project_nyquist(off,g);
    if (off != a) return false;
    g.nyquist_projection=true;
    Field pa=a, pb=b; project_nyquist(pa,g); project_nyquist(pb,g);
    Field twice=pa; project_nyquist(twice,g);
    Real left=0,right=0;
    for (int id=0;id<g.N;++id) {
        if (std::abs(twice[id]-pa[id])>1e-14) return false;
        left+=pa[id]*b[id]; right+=a[id]*pb[id];
    }
    for (int i=0;i<g.nx;++i)
        for (int j=0;j<g.ny;++j)
            for (int k=0;k<g.nz;++k)
                if (std::abs(pa[g.index(i,j,k)]-std::sin(2*PI*i/g.nx))>1e-14) return false;
    return std::abs(left-right)<1e-12;
}

// ---------------------------------------------------------------------------
// Force a local failure through the public step dispatcher
// Inputs: none; deliberately loose one-iteration controls force refinement
// Output: bool; refinement on succeeds, off fails, accepted input stays intact
// Dependencies: initialize_state, nonlinear_step_orbit, make_grid, gauss_legendre_01
// ---------------------------------------------------------------------------
static bool check_refinement_dispatch() {
    Config cfg;
    cfg.nx=cfg.ny=cfg.nz=4;
    cfg.Lx=cfg.Ly=cfg.Lz=2*PI;
    cfg.test_case="cold_relativistic_weibel";
    cfg.dt=1; cfg.particles_per_cell_pair=1; cfg.spline_order=1;
    cfg.orbit_quad_order=8; cfg.v0=0.9; cfg.weibel_B0=0.2;
    cfg.weibel_eigenmode=false;
    cfg.nonlinear_rtol=1e-9;cfg.nonlinear_atol=1e-11;cfg.nonlinear_max_iter=32;
    cfg.anderson_damping=0.5;cfg.uniform_local_rtol=2e-3;
    cfg.uniform_local_atol=1e-10;cfg.uniform_local_max_iter=1;
    cfg.output_prefix="forced_dispatch";
    Grid g=make_grid(cfg); auto quad=gauss_legendre_01(8);
    const State initial=initialize_state(cfg,g);
    const State before=initial;
    std::vector<NonlinearHistoryEntry> hist;
    bool failed=false;
    try { nonlinear_step_orbit(initial,g,cfg,quad,hist,nullptr,0); }
    catch(const NonlinearSolveFailure&) { failed=true; }
    if (!failed) return false;
    cfg.selective_particle_fallback=true;
    NonlinearSolveInfo info;
    State after=nonlinear_step_orbit(initial,g,cfg,quad,hist,&info,0);
    return info.converged && info.selective_refined_particles>0
        && info.uniform_substeps_used>1 && after.dt_last==cfg.dt
        && initial.x==before.x && initial.P==before.P && initial.rho==before.rho;
}

// ---------------------------------------------------------------------------
// Execute focused manuscript regressions
// Inputs: none
// Output: exit code zero when both regression checks pass
// Dependencies: check_projection, check_refinement_dispatch
// ---------------------------------------------------------------------------
int main() {
    if (!check_projection()) { std::cerr<<"Projector regression failed\n"; return 1; }
    if (!check_refinement_dispatch()) { std::cerr<<"Dispatch regression failed\n"; return 1; }
    std::cout<<"Projection algebra and selective-dispatch regressions passed.\n";
}
