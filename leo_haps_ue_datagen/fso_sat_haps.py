"""Satellite->HAPS FSO ergodic capacity (model taken from the original SAGIN project notebook).

Model (unchanged physics; group report sec. 3.2.1, eqs. (1)-(4), (11)):
  h_a = exp(-alpha * H_st * sec(zeta))                    Beer-Lambert stratospheric loss [16],[17]
  h_p ~ f(h) = phi^2 / A^phi^2 * h^(phi^2-1), 0<=h<=A        pointing error, Le et al. TAES 2023 [18] eq.(8)
  gamma = (R * P * h_a * h_p / sigma_n)^2,  gamma_max = (R P h_a A / sigma_n)^2
  f_gamma(g) = phi^2/(2g) * (g/gamma_max)^(phi^2/2)        (report eq. (4))
  C = B * int_0^gamma_max log2(1+g) f_gamma(g) dg           [bit/s]
Numerics changed (same math): substitute u = (g/gamma_max)^(phi^2/2)  =>  f_gamma dg = du,
  C = B * int_0^1 log2(1 + gamma_max * u^(2/phi^2)) du
which avoids the underflow (A*Ha)^(phi^2) -> 0 that makes the notebook return NaN.
"""
import math

import numpy as np
from scipy.integrate import quad

# B follows the project report (Table I); the original notebook used 1200 MHz (LEGACY_NOTEBOOK_PARAMS).
FSO_PARAMS = dict(
    alpha_st=2e-2,      # 1/km stratospheric attenuation coefficient (report Table I)
    H_st_km=28.0,       # km (report Table I)
    lam=1550e-9,        # m wavelength
    theta_div=20e-6,    # rad, full divergence angle (report Table I)
    Dr=0.08,            # m receiver aperture
    theta_jit=0.3e-6,   # rad satellite jitter (report Table I)
    sigma_Rd=1.0,       # m receiver-side displacement std -- NOT in report, origin unknown
    mu_R=0.0,           # m boresight offset
    P_tx_dBm=15.0,      # dBm (report Table I)
    responsivity=0.9,   # A/W -- not in report table
    sigma_n=1e-7,       # A  noise std (report: 1e-7 "A/Hz")
    B=1550e6,           # Hz (report Table I)
)
LEGACY_NOTEBOOK_PARAMS = dict(FSO_PARAMS, B=1200e6)


def stratos_atten(zenith_deg, p=FSO_PARAMS):
    return np.exp(-p["alpha_st"] * p["H_st_km"] / np.cos(np.radians(zenith_deg)))


def pointing_error(L, p=FSO_PARAMS):
    """(A_mod, phi_mod) at link length L [m]; identical to notebook pointing_error()."""
    k = 2 * np.pi / p["lam"]
    w0 = 2 * p["lam"] / (np.pi * p["theta_div"])
    Lambda0 = 2 * L / (k * w0**2)
    wL = w0 * np.sqrt(1 + Lambda0**2)
    v = np.sqrt(np.pi) * (p["Dr"] / 2) / (np.sqrt(2) * wL)
    A0 = math.erf(v) ** 2
    wLeq = wL * np.sqrt(np.sqrt(np.pi) * math.erf(v) / (2 * v * np.exp(-v**2)))
    sx = np.sqrt(p["sigma_Rd"] ** 2 + (p["theta_jit"] * L) ** 2)  # sigma_Rx = sigma_Ry
    mu = p["mu_R"]
    s2mod = ((3 * mu**2 * sx**4 + 3 * mu**2 * sx**4 + 2 * sx**6) / 2) ** (1 / 3)
    phi_mod = wLeq / (2 * np.sqrt(s2mod))
    phi_x = wLeq / (2 * sx)
    A_mod = A0 * np.exp(1 / phi_mod**2 - 1 / phi_x**2 - mu**2 / (sx**2 * phi_x**2))
    return A_mod, phi_mod


def gamma_max(L, zenith_deg, p=FSO_PARAMS):
    A, _ = pointing_error(L, p)
    Pt = 10 ** ((p["P_tx_dBm"] - 30) / 10)
    return (p["responsivity"] * Pt * stratos_atten(zenith_deg, p) * A / p["sigma_n"]) ** 2


def snr_pdf(g, L, zenith_deg, p=FSO_PARAMS):
    """f_gamma(g), evaluated in log domain (0 outside [0, gamma_max])."""
    _, phi = pointing_error(L, p)
    gm = gamma_max(L, zenith_deg, p)
    g = np.asarray(g, float)
    with np.errstate(divide="ignore"):
        logf = np.log(phi**2 / 2) - np.log(g) + (phi**2 / 2) * (np.log(g) - np.log(gm))
    return np.where((g > 0) & (g <= gm), np.exp(logf), 0.0)


def link_length(slant_m, zenith_deg, path_mode):
    """'legacy': notebook behaviour L = slant/cos(zenith) (correct only if input is the vertical
    separation); 'slant': input already is the true slant range, L = slant."""
    if path_mode == "legacy":
        return slant_m / np.cos(np.radians(zenith_deg))
    if path_mode == "slant":
        return slant_m
    raise ValueError(path_mode)


def ergodic_capacity(slant_m, zenith_deg, p=FSO_PARAMS, path_mode="slant"):
    """Ergodic FSO capacity [bit/s]. default uses the true slant range; 'legacy' reproduces the notebook."""
    L = link_length(slant_m, zenith_deg, path_mode)
    _, phi = pointing_error(L, p)
    gm = gamma_max(L, zenith_deg, p)
    e = 2 / phi**2
    val, _ = quad(lambda u: np.log2(1 + gm * u**e), 0.0, 1.0, limit=200)
    return p["B"] * val


_NODES = {}


def _gauss_nodes(n):
    if n not in _NODES:  # leggauss is costly, cache the nodes
        x, w = np.polynomial.legendre.leggauss(n)
        _NODES[n] = ((x + 1) / 2, w / 2)
    return _NODES[n]


def ergodic_capacity_vec(slant_m, zenith_deg, p=FSO_PARAMS, path_mode="slant", n=256):
    """Vectorised version: Gauss-Legendre on u in [0,1]; matches ergodic_capacity to <1e-6 rel."""
    slant_m, zenith_deg = np.broadcast_arrays(np.asarray(slant_m, float), np.asarray(zenith_deg, float))
    out = np.empty(slant_m.shape)
    u, w = _gauss_nodes(n)
    for idx in np.ndindex(slant_m.shape):
        L = link_length(slant_m[idx], zenith_deg[idx], path_mode)
        _, phi = pointing_error(L, p)
        gm = gamma_max(L, zenith_deg[idx], p)
        out[idx] = p["B"] * np.sum(w * np.log2(1 + gm * u ** (2 / phi**2)))
    return out


def mean_snr(slant_m, zenith_deg, p=FSO_PARAMS, path_mode="slant"):
    """E[gamma] = gamma_max * phi^2 / (phi^2 + 2)  (first moment of the eq.(4) pdf). Linear."""
    L = link_length(slant_m, zenith_deg, path_mode)
    _, phi = pointing_error(L, p)
    return gamma_max(L, zenith_deg, p) * phi**2 / (phi**2 + 2)
