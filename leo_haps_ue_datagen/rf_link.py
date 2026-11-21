"""RF link budget for Satellite-UE, HAPS-UE and (optional) Satellite-HAPS.

  SNR_dB = P_tx_dBm + G_tx_dBi + G_rx_dBi - L_total_dB - N_dBm
  N_dBm  = -174 + 10 log10(B_Hz) + NF_dB                     (kT0, T0 = 290 K)
  L_FSPL = 20 log10(4 pi d f / c)                             (Friis; ITU-R P.525)
  L_total = L_FSPL + L_atm + L_shadow (+ excess loss)        [dB]; small-scale fading as |h|^2 (linear)
  L_atm  = A_zenith / sin(elev)  (cosecant law, ITU-R P.618 sec. 2.2, valid for elev >= ~10 deg)
  SINR   = S / (N + sum I)  (linear);  C = B log2(1 + SINR)  [bit/s] (Shannon bound)

Presets are SIMULATION ASSUMPTIONS, sources stated per preset; verify before publishing.
"""
import numpy as np

C_LIGHT = 299_792_458.0

# HAPS-UE: group report Table II (Ptx 40 dBm, 2 GHz, 600 MHz, N = -80.4 dBm, a,b = 9.61,0.16,
#          xi = 1/20 dB). NF chosen so that -174+10log10(600e6)+NF = -80.4 dBm.
#          HAPS array gain 30 dBi (~256-element array; the report used 0 dBi), UE 0 dBi.
HAPS_UE = dict(f_hz=2e9, B_hz=600e6, P_tx_dBm=40.0, G_tx_dBi=30.0, G_rx_dBi=0.0,
               NF_dB=-80.4 - (-174 + 10 * np.log10(600e6)), A_zenith_dB=0.0,
               a=9.61, b=0.16, xi_los_dB=1.0, xi_nlos_dB=20.0)
# Satellite-UE: 3GPP TR 38.821 Table 6.1.1.1-1, LEO-600 S-band set-1 (EIRP density 34 dBW/MHz,
#   Tx gain 30 dBi, BW 30 MHz) and handheld UE (0 dBi, NF 7 dB). TO VERIFY against the TR.
#   Not Starlink parameters. UE assumed at beam centre (no beam pattern).
SAT_UE = dict(f_hz=2e9, B_hz=30e6, G_tx_dBi=30.0, G_rx_dBi=0.0, NF_dB=7.0,
              # P_tx = EIRP - G_tx, EIRP = 34 dBW/MHz over 30 MHz = 78.77 dBm
              P_tx_dBm=(34 + 10 * np.log10(30) + 30) - 30.0, A_zenith_dB=0.1)
# Satellite-HAPS RF alternative to FSO: Ka-band, all values ASSUMED (HAPS above ~99% of the
# troposphere's water vapour -> A_zenith ~ 0).
SAT_HAPS_RF = dict(f_hz=20e9, B_hz=400e6, P_tx_dBm=40.0, G_tx_dBi=38.5, G_rx_dBi=40.0,
                   NF_dB=2.0, A_zenith_dB=0.0)


def fspl_db(d_m, f_hz):
    return 20 * np.log10(4 * np.pi * np.asarray(d_m) * f_hz / C_LIGHT)


def noise_dbm(B_hz, NF_dB):
    return -174.0 + 10 * np.log10(B_hz) + NF_dB


def atm_loss_db(elev_deg, A_zenith_dB):
    """Cosecant law; NaN for elev <= 0 (no line of sight through the Earth) -> SNR NaN, link unavailable."""
    el = np.asarray(elev_deg, float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(el > 0, A_zenith_dB / np.sin(np.radians(el)), np.nan)


def p_los_al_hourani(elev_deg, a, b):
    """Al-Hourani et al., IEEE WCL 2014: P_LoS = 1 / (1 + a exp(-b (theta - a))), theta in deg."""
    return 1.0 / (1.0 + a * np.exp(-b * (np.asarray(elev_deg, float) - a)))


def haps_ue_pathloss_db(d_m, elev_deg, p=HAPS_UE):
    """Mean path loss = FSPL + P_LoS*xi_LoS + (1-P_LoS)*xi_NLoS (project report sec. 3.4)."""
    pl = p_los_al_hourani(elev_deg, p["a"], p["b"])
    return fspl_db(d_m, p["f_hz"]) + pl * p["xi_los_dB"] + (1 - pl) * p["xi_nlos_dB"]


def shadowing_db(rng, shape, sigma_dB):
    """Log-normal shadowing, zero-mean Gaussian in dB."""
    return rng.normal(0.0, sigma_dB, shape) if sigma_dB > 0 else np.zeros(shape)


def rician_power_gain(rng, shape, K):
    """|h|^2 with E|h|^2 = 1; K = inf -> 1 (pure LoS), K = 0 -> Rayleigh."""
    if np.isinf(K):
        return np.ones(shape)
    los = np.sqrt(K / (K + 1))
    s = np.sqrt(1 / (2 * (K + 1)))
    h = los + s * (rng.standard_normal(shape) + 1j * rng.standard_normal(shape))
    return np.abs(h) ** 2


def link_budget(d_m, elev_deg, p, extra_loss_dB=0.0):
    """Mean (no shadowing/fading) budget. Returns dict of dB quantities and linear channel gain."""
    L_fspl = fspl_db(d_m, p["f_hz"])
    L_atm = atm_loss_db(elev_deg, p.get("A_zenith_dB", 0.0))
    L_tot = L_fspl + L_atm + extra_loss_dB
    N = noise_dbm(p["B_hz"], p["NF_dB"])
    snr = p["P_tx_dBm"] + p["G_tx_dBi"] + p["G_rx_dBi"] - L_tot - N
    gain_db = p["G_tx_dBi"] + p["G_rx_dBi"] - L_tot
    return dict(fspl_dB=L_fspl, atm_dB=L_atm, path_loss_dB=L_tot, noise_dBm=N,
                channel_gain=10 ** (gain_db / 10), rx_power_dBm=p["P_tx_dBm"] + gain_db, snr_dB=snr)


def sinr_db(rx_power_dBm, noise_dBm, interference_dBm=()):
    tot = 10 ** (np.asarray(noise_dBm) / 10) + sum(10 ** (np.asarray(i) / 10) for i in interference_dBm)
    return np.asarray(rx_power_dBm) - 10 * np.log10(tot)


def capacity_bps(B_hz, snr_dB):
    return B_hz * np.log2(1 + 10 ** (np.asarray(snr_dB) / 10))


def availability(snr_dB, elev_deg, snr_min_dB, elev_min_deg):
    return (np.asarray(snr_dB) >= snr_min_dB) & (np.asarray(elev_deg) >= elev_min_deg)


def sat_ue_link(d_m, elev_deg, p=SAT_UE):
    return link_budget(d_m, elev_deg, p)


def haps_ue_link(d_m, elev_deg, p=HAPS_UE):
    excess = haps_ue_pathloss_db(d_m, elev_deg, p) - fspl_db(d_m, p["f_hz"])
    return link_budget(d_m, elev_deg, p, extra_loss_dB=excess)


def sat_haps_rf_link(d_m, elev_deg, p=SAT_HAPS_RF):
    return link_budget(d_m, elev_deg, p)
