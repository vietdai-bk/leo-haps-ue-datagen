"""ISAC sensing SNR (RF radar equation). Sensing SNR != communication SNR.

Monostatic (HAPS transmits the shared ISAC waveform and receives the echo from a ground UE/target):
  SNR_s = Pt Gt Gr lambda^2 sigma Gp / ((4pi)^3 R^4 k T0 B F Ls),  Gp = Np * B * Tp (coherent)
        = Pt Gt Gr lambda^2 sigma Np Tp / ((4pi)^3 R^4 k T0 F Ls)
  (Skolnik, Radar Handbook 3rd ed., ch.1; Richards, Fundamentals of Radar Signal Processing, ch.2)
Bistatic (satellite transmits, HAPS receives the echo): R^4 -> Rt^2 Rr^2 (Willis, Bistatic Radar, ch.4).
Detection: Swerling 1, P_d = P_fa^(1/(1+SNR)) (Richards eq. 6.x);
           Swerling 0 via Albersheim's approximation (valid 1e-7<=Pfa<=1e-3, 0.1<=Pd<=0.9).
Range accuracy (CRLB): sigma_R = c / (2 * 2pi * B_rms * sqrt(2 SNR)), B_rms = B/sqrt(12) for a flat spectrum.
ALL default parameters are SIMULATION ASSUMPTIONS.
"""
import numpy as np

from leo_haps_ue_datagen.rf_link import SAT_UE

K_BOLTZ = 1.380649e-23
T0 = 290.0
C_LIGHT = 299_792_458.0

# HAPS monostatic: same radio and array as rf_link.HAPS_UE
HAPS_MONO = dict(f_hz=2e9, B_hz=600e6, P_tx_dBm=40.0, G_tx_dBi=30.0, G_rx_dBi=30.0,
                 NF_dB=-80.4 - (-174 + 10 * np.log10(600e6)), L_s_dB=0.0, N_p=64, T_p_s=10e-6)
# Bistatic: satellite transmitter of rf_link.SAT_UE, HAPS array receiver
SAT_HAPS_BI = dict(f_hz=2e9, B_hz=30e6, P_tx_dBm=SAT_UE["P_tx_dBm"], G_tx_dBi=SAT_UE["G_tx_dBi"], G_rx_dBi=30.0,
                   NF_dB=HAPS_MONO["NF_dB"], L_s_dB=0.0, N_p=64, T_p_s=10e-6)
RCS_M2 = 1.0
P_FA = 1e-6
SNR_MIN_DB = 13.0


def _db(x):
    return 10 * np.log10(x)


def _common_db(p, rcs_m2):
    lam = C_LIGHT / p["f_hz"]
    return (p["P_tx_dBm"] - 30 + p["G_tx_dBi"] + p["G_rx_dBi"] + _db(lam**2) + _db(rcs_m2)
            + _db(p["N_p"] * p["T_p_s"]) - _db((4 * np.pi) ** 3) - _db(K_BOLTZ * T0)
            - p["NF_dB"] - p["L_s_dB"])


def snr_monostatic_db(R_m, rcs_m2=RCS_M2, p=HAPS_MONO):
    return _common_db(p, np.asarray(rcs_m2, float)) - 40 * np.log10(np.asarray(R_m, float))


def snr_bistatic_db(R_t_m, R_r_m, rcs_m2=RCS_M2, p=SAT_HAPS_BI):
    return (_common_db(p, np.asarray(rcs_m2, float)) - 20 * np.log10(np.asarray(R_t_m, float))
            - 20 * np.log10(np.asarray(R_r_m, float)))


def pd_swerling1(snr_db, p_fa=P_FA):
    return p_fa ** (1.0 / (1.0 + 10 ** (np.asarray(snr_db) / 10)))


def pd_albersheim(snr_db, p_fa=P_FA):
    """Swerling 0, single (already integrated) look: invert SNR = A + 0.12AB + 1.7B (N = 1)."""
    A = np.log(0.62 / p_fa)
    z = np.asarray(snr_db) / (6.2 + 4.54 / np.sqrt(1.44))
    B = (10**z - A) / (0.12 * A + 1.7)
    # Albersheim is only valid for 0.1 <= Pd <= 0.9; clip so it stays a probability >= Pfa
    return np.clip(1.0 / (1.0 + np.exp(-B)), p_fa, 1.0)


def range_crlb_m(snr_db, B_hz):
    b_rms = B_hz / np.sqrt(12.0)
    return C_LIGHT / (2 * 2 * np.pi * b_rms * np.sqrt(2 * 10 ** (np.asarray(snr_db) / 10)))


def sensing_metrics(snr_db, B_hz, p_fa=P_FA, snr_min_db=SNR_MIN_DB):
    return dict(isac_snr_dB=np.asarray(snr_db), pd_swerling1=pd_swerling1(snr_db, p_fa),
                range_crlb_m=range_crlb_m(snr_db, B_hz), sensing_feasible=np.asarray(snr_db) >= snr_min_db)
