import os
import sys

import numpy as np
from scipy.integrate import quad

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from leo_haps_ue_datagen import fso_sat_haps as F  # noqa: E402
from leo_haps_ue_datagen import isac as I  # noqa: E402


def test_monostatic_by_hand():
    R, s = 20e3, 1.0
    lam = I.C_LIGHT / 2e9
    p = I.HAPS_MONO
    lin = (10 ** ((40 - 30 + 60) / 10) * lam**2 * s * p["N_p"] * p["T_p_s"]
           / ((4 * np.pi) ** 3 * R**4 * I.K_BOLTZ * I.T0 * 10 ** (p["NF_dB"] / 10)))
    assert np.isclose(I.snr_monostatic_db(R, s), 10 * np.log10(lin))


def test_scaling_laws():
    assert np.isclose(I.snr_monostatic_db(20e3) - I.snr_monostatic_db(40e3), 40 * np.log10(2))  # 12.04 dB
    assert np.isclose(I.snr_monostatic_db(20e3, 10.0) - I.snr_monostatic_db(20e3, 1.0), 10.0)
    assert np.isclose(I.snr_bistatic_db(600e3, 20e3) - I.snr_bistatic_db(600e3, 40e3), 20 * np.log10(2))
    # bistatic with Rt = Rr = R and same radio == monostatic
    assert np.isclose(I.snr_bistatic_db(5e3, 5e3, 1.0, I.HAPS_MONO), I.snr_monostatic_db(5e3, 1.0))


def test_sensing_snr_differs_from_comm_snr():
    from leo_haps_ue_datagen import rf_link as RF
    comm = RF.haps_ue_link(20e3, 90)["snr_dB"]
    assert abs(I.snr_monostatic_db(20e3) - comm) > 10


def test_detection_probability():
    snr = np.linspace(-10, 30, 41)
    pd1, pd0 = I.pd_swerling1(snr), I.pd_albersheim(snr)
    assert np.all(np.diff(pd1) > 0) and np.all(np.diff(pd0) >= 0)
    assert np.all((pd0 >= I.P_FA) & (pd0 <= 1))
    assert np.isclose(I.pd_swerling1(-300), I.P_FA, rtol=1e-3)
    # Albersheim reference: Pfa=1e-6, Pd=0.9, N=1 -> ~13.1 dB (Richards Table 6.x / Skolnik)
    assert np.isclose(I.pd_albersheim(13.14), 0.9, atol=0.01)


def test_crlb():
    assert np.isclose(I.range_crlb_m(0.0, 600e6),
                      I.C_LIGHT / (2 * 2 * np.pi * 600e6 / np.sqrt(12) * np.sqrt(2)))
    assert I.range_crlb_m(20, 600e6) < I.range_crlb_m(10, 600e6)


def test_fso_mean_snr_matches_pdf():
    for L, z in [(530e3, 0.0), (960e3, 60.0)]:
        gm = F.gamma_max(L, z)
        num, _ = quad(lambda g: g * F.snr_pdf(g, L, z), 0, gm, limit=500)
        assert np.isclose(F.mean_snr(L, z), num, rtol=1e-4)
        assert F.mean_snr(L, z) <= gm
