#!/usr/bin/env python3
"""Hydrogenic orbital sampler with optional Dirac radial corrections.

The Dirac mode here uses the analytical Dirac-Coulomb bound-state scaling
(nu and gamma) for the radial large component. Angular structure is kept as
spherical harmonics so generated JSON stays compatible with the existing viewer.
"""

import argparse
import json
import os
from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import scipy.special as sp

A0 = 1.0
ALPHA = 1.0 / 137.035999084
ME_C2_EV = 510998.95


@dataclass
class OrbitalConfig:
    mode: Literal["schrodinger", "dirac"]
    n: int
    l: int
    m: int
    z: int
    j_branch: Literal["plus", "minus"] = "plus"


def spherical_harmonic(l: int, m: int, theta, phi):
    """Return Y_l^m(theta, phi) with theta=polar, phi=azimuth across SciPy APIs."""
    if hasattr(sp, "sph_harm_y"):
        return sp.sph_harm_y(l, m, theta, phi)

    # Legacy SciPy API: sph_harm(m, n, theta_azimuth, phi_polar)
    return sp.sph_harm(m, l, phi, theta)


def _dirac_quantum_numbers(n: int, l: int, z: int, j_branch: str):
    if j_branch == "plus":
        kappa = -(l + 1)
        j = l + 0.5
    else:
        if l == 0:
            raise ValueError("j= l-1/2 branch requires l >= 1.")
        kappa = l
        j = l - 0.5

    n_r = n - abs(kappa)
    if n_r < 0:
        raise ValueError(
            f"Invalid Dirac state: n={n} is too small for l={l}, j_branch={j_branch}."
        )

    z_alpha = z * ALPHA
    if z_alpha >= abs(kappa):
        raise ValueError(
            "Requested state is supercritical for this kappa (point nucleus model)."
        )

    gamma = np.sqrt(kappa * kappa - z_alpha * z_alpha)
    nu = n_r + gamma
    return kappa, n_r, gamma, nu, j


def dirac_binding_energy_ev(cfg: OrbitalConfig):
    _, _, _, nu, _ = _dirac_quantum_numbers(cfg.n, cfg.l, cfg.z, cfg.j_branch)
    z_alpha = cfg.z * ALPHA
    e_over_mc2 = 1.0 / np.sqrt(1.0 + (z_alpha / nu) ** 2)
    return (e_over_mc2 - 1.0) * ME_C2_EV


def radial_schrodinger(n: int, l: int, z: int, r):
    rho = 2.0 * z * r / (n * A0)

    log_norm = 0.5 * (
        3.0 * np.log(2.0 * z / (n * A0))
        + sp.gammaln(n - l)
        - np.log(2.0 * n)
        - sp.gammaln(n + l + 1)
    )
    norm = np.exp(log_norm)

    laguerre = sp.eval_genlaguerre(n - l - 1, 2 * l + 1, rho)
    return norm * np.exp(-rho / 2.0) * np.power(rho, l) * laguerre


def radial_dirac_large(cfg: OrbitalConfig, r):
    _, n_r, gamma, nu, _ = _dirac_quantum_numbers(cfg.n, cfg.l, cfg.z, cfg.j_branch)
    rho = 2.0 * cfg.z * r / (nu * A0)
    laguerre = sp.eval_genlaguerre(n_r, 2.0 * gamma, rho)
    rho_safe = np.maximum(rho, 1e-14)
    return np.exp(-rho / 2.0) * np.power(rho_safe, gamma - 1.0) * laguerre


def radial_pdf(cfg: OrbitalConfig, r):
    if cfg.mode == "schrodinger":
        radial = radial_schrodinger(cfg.n, cfg.l, cfg.z, r)
    else:
        radial = radial_dirac_large(cfg, r)

    pdf = np.square(np.abs(radial)) * np.square(r)
    pdf = np.nan_to_num(pdf, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(pdf, 0.0, None)


def build_cdf(grid, pdf):
    dx = np.diff(grid)
    trapezoids = 0.5 * (pdf[:-1] + pdf[1:]) * dx
    cdf = np.concatenate(([0.0], np.cumsum(trapezoids)))

    total = cdf[-1]
    if total <= 0.0:
        raise ValueError("Failed to build CDF: PDF integrated to zero.")

    return cdf / total


def build_theta_cdf(l: int, m: int, size: int = 4096):
    theta = np.linspace(0.0, np.pi, size)
    y = np.abs(spherical_harmonic(l, m, theta, 0.0)) ** 2
    pdf = np.sin(theta) * y
    cdf = build_cdf(theta, pdf)
    return theta, cdf


def default_r_max(cfg: OrbitalConfig):
    if cfg.mode == "schrodinger":
        return max(50.0 * cfg.n * cfg.n / cfg.z, 15.0 / cfg.z)

    _, _, _, nu, _ = _dirac_quantum_numbers(cfg.n, cfg.l, cfg.z, cfg.j_branch)
    return max(60.0 * nu * nu / cfg.z, 20.0 * nu / cfg.z)


def wavefunction(cfg: OrbitalConfig, r, theta, phi):
    if cfg.mode == "schrodinger":
        radial = radial_schrodinger(cfg.n, cfg.l, cfg.z, r)
    else:
        radial = radial_dirac_large(cfg, r)

    y_lm = spherical_harmonic(cfg.l, cfg.m, theta, phi)
    return radial * y_lm


def sample_points(cfg: OrbitalConfig, sample_count: int, seed: Optional[int] = None):
    rng = np.random.default_rng(seed)

    r_grid = np.linspace(1e-6, default_r_max(cfg), 20000)
    r_cdf = build_cdf(r_grid, radial_pdf(cfg, r_grid))

    theta_grid, theta_cdf = build_theta_cdf(cfg.l, cfg.m)

    r = np.interp(rng.random(sample_count), r_cdf, r_grid)
    theta = np.interp(rng.random(sample_count), theta_cdf, theta_grid)
    phi = rng.uniform(0.0, 2.0 * np.pi, size=sample_count)

    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)

    psi = wavefunction(cfg, r, theta, phi)

    points = []
    for idx in range(sample_count):
        points.append(
            {
                "x": float(x[idx]),
                "y": float(y[idx]),
                "z": float(z[idx]),
                "psi_re": float(np.real(psi[idx])),
                "psi_im": float(np.imag(psi[idx])),
            }
        )

    return points


def parse_mode(raw: str):
    value = raw.strip().lower()
    if value in {"", "d", "dirac"}:
        return "dirac"
    if value in {"s", "sch", "schrodinger"}:
        return "schrodinger"
    raise ValueError("Mode must be one of: dirac, schrodinger.")


def prompt_int(
    prompt_text: str,
    *,
    default: Optional[int] = None,
    min_value: Optional[int] = None,
    max_value: Optional[int] = None,
):
    while True:
        raw = input(prompt_text).strip()

        if raw == "":
            if default is None:
                print("Please enter a value.")
                continue
            value = default
        else:
            try:
                value = int(raw)
            except ValueError:
                print("Please enter a whole number.")
                continue

        if min_value is not None and value < min_value:
            print(f"Value must be >= {min_value}.")
            continue
        if max_value is not None and value > max_value:
            print(f"Value must be <= {max_value}.")
            continue

        return value


def validate_config(cfg: OrbitalConfig):
    if cfg.n < 1:
        raise ValueError("n must be >= 1.")
    if cfg.l < 0 or cfg.l > cfg.n - 1:
        raise ValueError(f"l must be in [0, {cfg.n - 1}].")
    if abs(cfg.m) > cfg.l:
        raise ValueError(f"m must be in [-{cfg.l}, {cfg.l}].")
    if cfg.z < 1:
        raise ValueError("z (nuclear charge) must be >= 1.")

    if cfg.mode == "dirac":
        _dirac_quantum_numbers(cfg.n, cfg.l, cfg.z, cfg.j_branch)


def prompt_config():
    print("\n=== Orbital Generator (Schrodinger + Dirac Radial Mode) ===\n")

    mode = parse_mode(input("Mode [dirac/schrodinger] (default: dirac): "))
    z = prompt_int(
        "Nuclear charge Z (e.g. 1=H, 6=C, 26=Fe) (default: 1): ",
        default=1,
        min_value=1,
    )
    n = prompt_int("Enter n (1..) (default: 1): ", default=1, min_value=1)
    l = prompt_int(
        f"Enter l (0..{n - 1}) (default: 0): ", default=0, min_value=0, max_value=n - 1
    )
    m = prompt_int(
        f"Enter m (-{l}..{l}) (default: 0): ", default=0, min_value=-l, max_value=l
    )

    j_branch = "plus"
    if mode == "dirac":
        raw = input("Dirac j branch [plus/minus] (default: plus): ").strip().lower()
        if raw:
            if raw not in {"plus", "minus"}:
                raise ValueError("j branch must be plus or minus.")
            j_branch = raw

    sample_count = prompt_int(
        "How many particle samples? (default: 50000): ", default=50000, min_value=1
    )

    seed_raw = input("Random seed (blank for random): ").strip()
    seed = int(seed_raw) if seed_raw else None

    return OrbitalConfig(mode=mode, n=n, l=l, m=m, z=z, j_branch=j_branch), sample_count, seed


def default_filename(cfg: OrbitalConfig):
    base = f"orbital_{cfg.mode}_Z{cfg.z}_n{cfg.n}_l{cfg.l}_m{cfg.m}"
    if cfg.mode == "dirac":
        base += f"_j{cfg.j_branch}"
    return os.path.join("orbitals", f"{base}.json")


def build_parser():
    parser = argparse.ArgumentParser(description="Generate hydrogenic orbital samples.")
    parser.add_argument("--mode", choices=["schrodinger", "dirac"])
    parser.add_argument("--z", type=int, help="Nuclear charge Z")
    parser.add_argument("--n", type=int)
    parser.add_argument("--l", type=int)
    parser.add_argument("--m", type=int)
    parser.add_argument("--j-branch", choices=["plus", "minus"], default="plus")
    parser.add_argument("--samples", type=int, default=50000)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output", type=str)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.mode:
        required = {"z": args.z, "n": args.n, "l": args.l, "m": args.m}
        missing = [name for name, value in required.items() if value is None]
        if missing:
            parser.error(f"Missing required args with --mode: {', '.join(missing)}")

        cfg = OrbitalConfig(
            mode=args.mode,
            z=args.z,
            n=args.n,
            l=args.l,
            m=args.m,
            j_branch=args.j_branch,
        )
        sample_count = args.samples
        seed = args.seed
    else:
        cfg, sample_count, seed = prompt_config()

    validate_config(cfg)

    print("\nGenerating samples... please wait...")
    points = sample_points(cfg, sample_count=sample_count, seed=seed)

    filename = args.output if args.output else default_filename(cfg)
    out_dir = os.path.dirname(filename)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    payload = {
        "mode": cfg.mode,
        "z": cfg.z,
        "n": cfg.n,
        "l": cfg.l,
        "m": cfg.m,
        "points": points,
    }

    if cfg.mode == "dirac":
        kappa, n_r, gamma, nu, j = _dirac_quantum_numbers(
            cfg.n, cfg.l, cfg.z, cfg.j_branch
        )
        payload["j_branch"] = cfg.j_branch
        payload["j"] = j
        payload["kappa"] = kappa
        payload["n_r"] = n_r
        payload["gamma"] = float(gamma)
        payload["nu"] = float(nu)
        payload["binding_energy_ev"] = float(dirac_binding_energy_ev(cfg))
        payload["notes"] = (
            "Dirac-Coulomb large-component radial scaling with spherical-harmonic angular part."
        )

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(payload, f)

    print(f"\nSaved {len(points)} samples to: {filename}")
    print("\nLoad this JSON in your C++ viewer to visualize.\n")


if __name__ == "__main__":
    main()
