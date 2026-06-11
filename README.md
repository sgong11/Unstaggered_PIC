# Unstaggered Particle-in-Cell Teaching Codes

This repository collects small, serial C++17 teaching implementations related to unstaggered particle-in-cell methods and generalized-momentum electromagnetic formulations. The goal is to provide one place for readable, self-contained reference codes connected to our current and related papers.

The codes are intended for study, verification, and modification by students and researchers. They are deliberately kept simple and self-contained rather than optimized for large-scale production use.

## Repository layout

```text
.
├── Project1/
│   └── README.md
├── Project2/
│   └── README.md
└── README.md
```

Each project directory contains its own README with details about the method, code organization, build instructions, and example usage.

## Project1: Current paper code

`Project1` contains a serial C++17 teaching implementation of the energy-conserving unstaggered potential particle-in-cell method described in the accompanying paper:

**An Energy-Conserving Unstaggered Electromagnetic-Potential Particle-in-Cell Method, Part I: Non-relativistic Generalized-Momentum Formulation**

The manuscript is expected to be released on arXiv soon. This project is associated with our current paper and focuses on the non-relativistic generalized-momentum formulation of the unstaggered electromagnetic-potential PIC method.

## Project2: Part III spectral generalized-momentum PIC code

`Project2` contains a small, serial C++17 teaching implementation for the spectral generalized-momentum particle-in-cell ideas used in:

Andrew J. Christlieb, William A. Sands, and Stephen R. White,  
**A Particle-in-cell Method for Plasmas with A Generalized Momentum Formulation, Part III: A family of Gauge Conserving Methods**,  
*Journal of Scientific Computing* **104**, 38, 2025.  
DOI: `10.1007/s10915-025-02953-7`

This project is related to the Part III paper by Stephen White and Bill Sands. The code is intentionally self-contained: it uses no MPI, no OpenMP, and no FFTW. In particular, `spectral_pic_part3.cpp` includes its own radix-2 FFT and is heavily commented so that the method can be read and modified by students.

## Purpose

This repository is meant to serve as a shared home for teaching and reference implementations related to unstaggered particle-in-cell methods. The emphasis is on clarity, reproducibility, and connection to the mathematical formulations in the corresponding papers.

These codes may be useful for:

- learning the structure of generalized-momentum PIC algorithms;
- comparing related unstaggered electromagnetic formulations;
- reproducing small test problems from the associated papers;
- modifying individual components for teaching, experimentation, or verification.

## Requirements

The projects are written in standard C++17. A typical build only requires a C++17-capable compiler such as `g++` or `clang++`.

Individual project directories may contain more specific build or run instructions.

## Notes

These implementations are serial teaching codes. They are not intended to replace optimized production PIC codes. Instead, they prioritize transparency of the numerical method and ease of modification.

## Citation

If you use code from `Project2`, please cite:

```bibtex
@article{ChristliebSandsWhite2025PartIII,
  author  = {Andrew J. Christlieb and William A. Sands and Stephen R. White},
  title   = {A Particle-in-cell Method for Plasmas with A Generalized Momentum Formulation, Part III: A family of Gauge Conserving Methods},
  journal = {Journal of Scientific Computing},
  volume  = {104},
  pages   = {38},
  year    = {2025},
  doi     = {10.1007/s10915-025-02953-7}
}
```

A citation for `Project1` will be added after the accompanying manuscript is available.
