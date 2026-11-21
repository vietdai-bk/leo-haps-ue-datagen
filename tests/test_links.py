import os
import sys

import numpy as np
import pytest
from scipy.integrate import quad

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from leo_haps_ue_datagen import fso_sat_haps as F  # noqa: E402
from leo_haps_ue_datagen import rf_link as R  # noqa: E402


# Reference values produced once by the original notebook (pointing_error, C_FSO with B=1200 MHz,
# Dr=8 cm, legacy slant/cos path). The notebook returned NaN at (960 km, zenith 60 deg).
NB_POINTING = {300e3: (0.00035539326517870947, 1.4943027359806633),
               530e3: (0.00011390287943645536, 2.617316165460676),
               1e6: (3.199868492894528e-05, 4.789229836342433)}
NB_CFSO = {(300e3, 1.0): 12497438699.677868, (530e3, 0.0): 9607550705.734255, (604e3, 30.0): 7629396224.368599}


def test_fspl_known_value():
    # 1 km @ 2 GHz: 32.45 + 20log10(2000 MHz) + 20log10(1 km) = 98.47 dB
    assert np.isclose(R.fspl_db(1e3, 2e9), 32.45 + 20 * np.log10(2000), atol=0.01)
    assert np.isclose(R.fspl_db(2e3, 2e9) - R.fspl_db(1e3, 2e9), 20 * np.log10(2))


def test_noise_and_budget_by_hand():
    assert np.isclose(R.noise_dbm(1e6, 0), -114.0, atol=0.01)
    p = dict(f_hz=2e9, B_hz=1e6, P_tx_dBm=30, G_tx_dBi=10, G_rx_dBi=3, NF_dB=5, A_zenith_dB=0.0)
    lb = R.link_budget(1e3, 90, p)
    expect = 30 + 10 + 3 - 98.4684 - (-174 + 60 + 5)
    assert np.isclose(lb["snr_dB"], expect, atol=0.01)
    assert np.isclose(10 * np.log10(lb["channel_gain"]), 13 - (32.45 + 20 * np.log10(2000)), atol=0.01)


def test_haps_noise_matches_report():
    assert np.isclose(R.noise_dbm(R.HAPS_UE["B_hz"], R.HAPS_UE["NF_dB"]), -80.4)


def test_atm_cosecant():
    assert np.isclose(R.atm_loss_db(90, 0.5), 0.5) and np.isclose(R.atm_loss_db(30, 0.5), 1.0)


def test_snr_decreases_with_distance():
    d = np.linspace(500e3, 2000e3, 50)
    snr = R.sat_ue_link(d, 60)["snr_dB"]
    assert np.all(np.diff(snr) < 0)


def test_p_los_limits():
    assert R.p_los_al_hourani(90, 9.61, 0.16) > 0.99
    assert R.p_los_al_hourani(0, 9.61, 0.16) < 0.1


def test_haps_ue_matches_legacy_formula():
    # legacy SAGIN HAPS formula (40 dBm, N = -80.4 dBm) plus the 30 dBi HAPS array; c = 3e8 there -> <0.01 dB
    d, z = 25e3, 20e3
    el = np.degrees(np.arcsin(z / d))
    lb = R.haps_ue_link(d, el)
    pl = 1 / (1 + 9.61 * np.exp(-0.16 * (el - 9.61)))
    eta = 20 * np.log10(4 * np.pi * 2e9 * d / 3e8) + pl * 1 + (1 - pl) * 20
    assert np.isclose(lb["snr_dB"], 40 + 30 - eta + 80.4, atol=0.01)


def test_fading_and_shadowing_statistics():
    rng = np.random.default_rng(0)
    for K in (0.0, 3.0, 10.0):
        assert np.isclose(R.rician_power_gain(rng, 200_000, K).mean(), 1.0, atol=0.01)
    assert np.all(R.rician_power_gain(rng, 10, np.inf) == 1.0)
    s = R.shadowing_db(rng, 200_000, 4.0)
    assert abs(s.mean()) < 0.05 and np.isclose(s.std(), 4.0, atol=0.05)


def test_sinr_and_capacity():
    assert np.isclose(R.sinr_db(-90, -100), 10.0)
    assert np.isclose(R.sinr_db(-90, -100, [-100]), 10 - 10 * np.log10(2))
    assert np.isclose(R.capacity_bps(1e6, 0.0), 1e6)  # SNR 1 -> 1 bit/s/Hz
    assert R.availability([5, 15], [40, 40], 10, 30).tolist() == [False, True]


def test_pointing_error_matches_notebook():
    for L, (A0, phi0) in NB_POINTING.items():
        A, phi = F.pointing_error(L)
        assert np.isclose(A, A0, rtol=1e-12) and np.isclose(phi, phi0, rtol=1e-12)


@pytest.mark.parametrize("L,z", [(530e3, 0.0), (700e3, 30.0), (1e6, 50.0)])
def test_snr_pdf_normalised(L, z):
    gm = F.gamma_max(L, z)
    val, _ = quad(lambda g: F.snr_pdf(g, L, z), 0, gm, limit=500, points=[gm * 1e-6])
    assert np.isclose(val, 1.0, atol=1e-4)


def test_capacity_matches_notebook_where_notebook_is_finite():
    for (slant, z), ref in NB_CFSO.items():
        assert np.isclose(F.ergodic_capacity(slant, z, F.LEGACY_NOTEBOOK_PARAMS, "legacy"), ref, rtol=1e-4)


def test_capacity_finite_where_notebook_is_nan():
    slant, z = 960e3, 60.0  # elev 30 deg from HAPS to 550 km shell; notebook gave NaN here
    c = F.ergodic_capacity(slant, z, F.LEGACY_NOTEBOOK_PARAMS, "legacy")
    gm = F.gamma_max(F.link_length(slant, z, "legacy"), z)
    assert 0 < c < 1200e6 * np.log2(1 + gm)  # Jensen upper bound


def test_capacity_bounds_and_monotonic():
    L = np.linspace(500e3, 1500e3, 6)
    c = [F.ergodic_capacity(x, 0.0, path_mode="slant") for x in L]
    assert np.all(np.diff(c) < 0)
    for x, cc in zip(L, c):
        assert cc <= F.FSO_PARAMS["B"] * np.log2(1 + F.gamma_max(x, 0.0))


def test_path_modes():
    assert F.link_length(600e3, 60, "slant") == 600e3
    assert np.isclose(F.link_length(600e3, 60, "legacy"), 1200e3)
    with pytest.raises(ValueError):
        F.link_length(1, 0, "bad")


def test_vectorised_matches_quad():
    s = np.array([530e3, 700e3, 960e3])
    z = np.array([0.0, 30.0, 60.0])
    v = F.ergodic_capacity_vec(s, z, path_mode="slant")
    q = [F.ergodic_capacity(a, b, path_mode="slant") for a, b in zip(s, z)]
    assert np.allclose(v, q, rtol=1e-5)


def test_no_link_below_horizon():
    assert np.isnan(R.atm_loss_db(-0.001, 0.1)) and np.isnan(R.atm_loss_db(0.0, 0.1))
    lb = R.sat_ue_link(np.array([1e6, 1e6]), np.array([-1e-5, 45.0]))
    assert np.isnan(lb["snr_dB"][0]) and np.isfinite(lb["snr_dB"][1])
