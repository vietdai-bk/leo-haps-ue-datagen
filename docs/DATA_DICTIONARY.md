## 0. Quy ước chung

| Quy ước | Giá trị |
|---|---|
| Thời gian | `timestamp`: UTC, độ phân giải ms. `t_idx`: chỉ số bước, $t = t_0 + \texttt{t\_idx}\cdot\Delta t$ (mặc định $\Delta t = 1$ s) |
| Khoảng cách | m |
| Vận tốc | m/s |
| Góc | độ |
| Công suất | dBm |
| SNR | dB |
| Dung lượng | bit/s |
| Băng thông | Hz |
| RST | s |
| Hệ tọa độ | **ECEF** (WGS84, gốc ở tâm Trái Đất) cho vệ tinh và HAPS. **ENU cục bộ** (East–North–Up) có gốc tại điểm chiếu của HAPS trên mặt đất (độ cao 0 m) cho UE |
| Góc ngẩng / phương vị | góc ngẩng so với mặt phẳng tiếp tuyến ellipsoid WGS84 tại điểm quan sát. Phương vị tính từ hướng Bắc, theo chiều kim đồng hồ |
| Giá trị thiếu | **NaN**, không bao giờ dùng 0 |
| `sat_id` | số hiệu NORAD của vệ tinh |
| `ue_id` | $0,\ldots,N_{UE}-1$ |
| Khóa nối bảng | `t_idx` (hoặc `timestamp`) + `sat_id` và/hoặc `ue_id` |

### Hằng số và preset dùng trong các công thức

| Ký hiệu | HAPS→UE (`HAPS_UE`) | Sat→UE (`SAT_UE`) | Sat→HAPS RF (`SAT_HAPS_RF`) |
|---|---|---|---|
| $f$ | 2 GHz | 2 GHz | 20 GHz |
| $B$ | 600 MHz | 30 MHz | 400 MHz |
| $P_{tx}$ (= `p_max_dBm`) | 40 dBm | 48.77 dBm (EIRP 78.77 dBm − 30 dBi) | 40 dBm |
| $G_{tx}$ / $G_{rx}$ | 30 / 0 dBi | 30 / 0 dBi | 38.5 / 40 dBi |
| $NF$ | 5.82 dB (để $N = -80.4$ dBm) | 7 dB | 2 dB |
| $A_{zen}$ (suy hao khí quyển ở thiên đỉnh) | 0 dB | 0.1 dB | 0 dB |
| Tham số LoS $(a, b, \xi_{LoS}, \xi_{NLoS})$ | 9.61, 0.16, 1 dB, 20 dB | – | – |

$c = 299\,792\,458$ m/s, $k_B = 1.380649\times10^{-23}$ J/K, $T_0 = 290$ K.

---

## 1. Bảng `sat_haps.parquet`: vệ tinh → HAPS

**Một dòng** = một cặp (vệ tinh, thời điểm) mà góc ngẩng của vệ tinh nhìn từ HAPS $\ge 0^\circ$.

> **Tập vệ tinh trong bảng** là các vệ tinh *ứng viên*. Đó là những vệ tinh trong cửa sổ mô phỏng có lúc
> tiến lại đủ gần HAPS để có góc ngẩng khoảng $\ge 20^\circ$ (đúng bằng `ue_row_min_elev_deg`, có thêm biên an toàn
> 60 km). Vì vậy:
> - dòng có góc ngẩng $\ge 20^\circ$ là **đầy đủ**: đã kiểm tra bằng skyfield, không thiếu, không thừa;
> - dòng có góc ngẩng từ $0^\circ$ đến $20^\circ$ **không đầy đủ**: chỉ có phần đầu và phần cuối lượt bay của các vệ tinh
>   ứng viên, không có các vệ tinh chỉ bay sát đường chân trời.

| Cột | Kiểu | Đơn vị | Ý nghĩa và cách tính |
|---|---|---|---|
| `t_idx` | int | – | chỉ số bước thời gian |
| `timestamp` | timestamp | UTC | thời điểm |
| `sat_id` | int | – | NORAD ID |
| `sat_x`, `sat_y`, `sat_z` | float | m | vị trí vệ tinh trong ECEF, $\mathbf r_{ECEF} = R_3(\theta_{GMST})\,\mathbf r_{TEME}$ (mục 5.0) |
| `sat_vx`, `sat_vy`, `sat_vz` | float | m/s | vận tốc trong ECEF (so với mặt đất đang quay), $\mathbf v_{ECEF} = R_3(\theta_{GMST})\,\mathbf v_{TEME} - \boldsymbol\omega_E \times \mathbf r_{ECEF}$ |
| `haps_x`, `haps_y`, `haps_z` | float | m | vị trí HAPS trong ECEF, đổi từ (lat, lon, alt) theo WGS84. Không đổi theo thời gian vì HAPS đứng yên |
| `sat_haps_distance_m` | float | m | $d = \lVert \mathbf r_{sat} - \mathbf r_{HAPS} \rVert$ (slant range) |
| `elev_deg` | float | độ | góc ngẩng $\varepsilon$ của vệ tinh nhìn từ HAPS (mục 5.0) |
| `az_deg` | float | độ | phương vị (`pymap3d.ecef2aer`) |
| `geometric_visible` | bool | – | $\varepsilon \ge 0^\circ$. Trong bảng này luôn là True |
| `available` | bool | – | **liên lạc khả dụng**: $\varepsilon \ge 30^\circ$ (`elev_min_haps_deg`). Không có điều kiện SNR |
| `rst_sat_haps_s` | float | s | **RST**: thời gian còn lại của khoảng `available` liên tục hiện tại (mục 5.6). NaN khi `rst_status` $\neq 1$ |
| `rst_status` | int8 | – | 0 = không phục vụ, 1 = đang phục vụ và đã biết lúc rời, 2 = censored (vẫn phục vụ ở cuối cửa sổ) |
| `rst_lb_s` | float | s | cận dưới của RST. Khi `rst_status` = 1 thì bằng `rst_sat_haps_s`; khi = 2 thì là $t_{last} + \Delta t - t$; khi = 0 thì là NaN |
| `fso_capacity_bps` | float | bit/s | dung lượng ergodic $C_{FSO}$ của liên kết FSO (mục 5.3). **NaN khi `available` = False** |
| `fso_mean_snr_dB` | float | dB | $10\log_{10}\mathbb E[\gamma]$, với $\mathbb E[\gamma] = \gamma_{max}\,\phi^2/(\phi^2+2)$. NaN khi không khả dụng |
| `fso_gamma_max_dB` | float | dB | $10\log_{10}\gamma_{max}$, SNR tức thời cực đại của FSO (mục 5.3). NaN khi không khả dụng |
| `rf_snr_dB` | float | dB | SNR nếu dùng **RF Ka-band** thay cho FSO: link budget (mục 5.1) với preset `SAT_HAPS_RF`, ở công suất `rf_p_max_dBm` |
| `rf_capacity_bps` | float | bit/s | $B\log_2\!\left(1 + 10^{\mathrm{SNR_{dB}}/10}\right)$ |
| `rf_channel_gain` | float | tuyến tính | $g = G_{tx}G_{rx}/L_{total}$ (mục 5.1) |
| `rf_noise_power_dBm` | float | dBm | $-174 + 10\log_{10}B + NF = -85.98$ dBm |
| `rf_bandwidth_hz` | float | Hz | $4\times10^{8}$ |
| `rf_p_max_dBm` | float | dBm | 40 |

Các cột `rf_*` có cho **mọi dòng**, kể cả khi `available` = False, để người dùng tự chọn ngưỡng.

---

## 2. Bảng `sat_ue.parquet`: vệ tinh → UE (truy nhập trực tiếp)

**Một dòng** = một bộ (vệ tinh, UE, thời điểm) mà góc ngẩng của vệ tinh nhìn từ UE $\ge 20^\circ$
(`ue_row_min_elev_deg`). Tập vệ tinh giống bảng `sat_haps`.

| Cột | Kiểu | Đơn vị | Ý nghĩa và cách tính |
|---|---|---|---|
| `t_idx`, `timestamp`, `sat_id`, `ue_id` | | | khóa |
| `user_type` | str | – | `critical` hoặc `normal`, lấy từ bảng `ue` |
| `sat_ue_distance_m` | float | m | $\lVert \mathbf r_{sat}(t) - \mathbf r_{UE}(t) \rVert$. **UE di chuyển**: $\mathbf r_{UE}$ lấy tại đúng $t$, với UE ở độ cao 0 m trên WGS84 |
| `elev_deg` | float | độ | góc ngẩng $\varepsilon$ của vệ tinh nhìn từ UE, dùng pháp tuyến WGS84 tại vị trí của UE ở thời điểm $t$ |
| `snr_sat_ue_dB` | float | dB | link budget (mục 5.1) với preset `SAT_UE`, ở công suất `p_max_dBm`. Không tính fading/shadowing, không tính nhiễu từ nút khác |
| `channel_gain` | float | tuyến tính | $g = G_{tx}G_{rx}/L_{total}$, với $L_{total} = L_{FSPL} + A_{zen}/\sin\varepsilon$ (tính theo dB) |
| `capacity_bps` | float | bit/s | $B\log_2(1+\mathrm{SNR})$: UE dùng **toàn bộ** băng, chưa chia cho các UE khác |
| `noise_power_dBm` | float | dBm | $-174 + 10\log_{10}(3\times10^{7}) + 7 = -92.23$ dBm |
| `bandwidth_hz` | float | Hz | $3\times10^{7}$ |
| `p_max_dBm` | float | dBm | 48.77 |
| `geometric_visible` | bool | – | $\varepsilon \ge 0^\circ$. Trong bảng này luôn là True |
| `available` | bool | – | $\varepsilon \ge 30^\circ$ **và** $\mathrm{SNR} \ge 0$ dB (`snr_min_comm_dB`) |
| `rst_sat_ue_s`, `rst_status`, `rst_lb_s` | | s | RST của liên kết vệ tinh → **UE cụ thể này**, tính theo điều kiện `available` ở trên, có tính chuyển động của UE (mục 5.6) |
| `isac_snr_dB` | float | dB | SNR sensing **bistatic**: vệ tinh phát, tín hiệu phản xạ từ UE, HAPS thu (mục 5.4). $R_t$ = `sat_ue_distance_m`, $R_r$ = khoảng cách HAPS–UE ở cùng $t$ |
| `isac_pd` | float | 0–1 | xác suất phát hiện $P_d$, mô hình Swerling 1, $P_{fa} = 10^{-6}$ |
| `isac_range_crlb_m` | float | m | cận dưới Cramér–Rao $\sigma_R$ của sai số đo khoảng cách, $B = 30$ MHz |
| `sensing_feasible` | bool | – | $\mathrm{SNR}_s \ge 13$ dB |

---

## 3. Bảng `haps_ue.parquet`: HAPS → UE

**Một dòng** = một cặp (UE, thời điểm). Có mọi UE ở mọi bước, vì HAPS luôn nằm cao trên đầu UE.

| Cột | Kiểu | Đơn vị | Ý nghĩa và cách tính |
|---|---|---|---|
| `t_idx`, `timestamp`, `ue_id`, `user_type` | | | khóa và loại UE |
| `haps_ue_distance_m` | float | m | $\lVert \mathbf r_{HAPS} - \mathbf r_{UE}(t) \rVert$ |
| `elev_deg` | float | độ | góc ngẩng $\theta$ của HAPS nhìn từ UE |
| `snr_haps_ue_dB` | float | dB | link budget (mục 5.1), có thêm suy hao dư theo xác suất LoS (mục 5.2), preset `HAPS_UE`, ở công suất `p_max_dBm` |
| `channel_gain` | float | tuyến tính | $g = G_{tx}G_{rx}/L_{total}$, với $L_{total} = L_{FSPL} + P_{LoS}\xi_{LoS} + (1-P_{LoS})\xi_{NLoS}$ (tính theo dB) |
| `noise_power_dBm` | float | dBm | $-80.4$ dBm |
| `bandwidth_hz` | float | Hz | $6\times10^{8}$ |
| `p_max_dBm` | float | dBm | 40 |
| `capacity_bps` | float | bit/s | $B\log_2(1+\mathrm{SNR})$ (dùng toàn bộ băng) |
| `available` | bool | – | $\mathrm{SNR} \ge 0$ dB |
| `isac_snr_dB` | float | dB | SNR sensing **monostatic**: HAPS phát và thu tiếng vọng từ UE (mục 5.4), $R$ = `haps_ue_distance_m` |
| `isac_pd`, `isac_range_crlb_m`, `sensing_feasible` | | | như bảng `sat_ue`, nhưng $B = 600$ MHz |

Bảng này **không có RST**: HAPS đứng yên và luôn phục vụ UE trong vùng, nên RST không có ý nghĩa.

---

## 4. Bảng `ue.parquet`: người dùng

**Một dòng** = một cặp (UE, thời điểm).

| Cột | Kiểu | Đơn vị | Ý nghĩa và cách tính |
|---|---|---|---|
| `ue_id`, `t_idx`, `timestamp` | | | khóa |
| `user_type` | str | – | `critical` (20%) hoặc `normal`. Gán ngẫu nhiên theo seed, không đổi trong suốt mô phỏng |
| `state` | str | – | trạng thái di chuyển `stationary` / `pedestrian` / `vehicle`, theo chuỗi Markov (mục 5.5) |
| `e`, `n` | float | m | vị trí $\mathbf p = (e, n)$ trong mặt phẳng ENU, gốc tại điểm chiếu của HAPS. Luôn có $\lVert\mathbf p\rVert \le R_{area} = 10$ km |
| `lat`, `lon` | float | độ | `pymap3d.enu2geodetic(e, n, 0)`. Độ cao UE = 0 m |
| `vx`, `vy` | float | m/s | vận tốc $\mathbf v = (v_x, v_y)$ theo hướng Đông và Bắc tại bước $t$. Vị trí bước sau: $\mathbf p(t+1) = \mathbf p(t) + \mathbf v(t)\Delta t$, trừ khi chạm biên |
| `speed` | float | m/s | $\sqrt{v_x^2 + v_y^2}$ |
| `max_latency_ms` | float | ms | Critical 10, Normal 100 |
| `min_rate_bps` | float | bit/s | Critical $10^6$, Normal $5\times10^6$ |
| `reliability` | float | – | Critical 0.99999, Normal 0.99 |
| `priority` | int | – | Critical 1 (cao nhất), Normal 5 |

---

## 5. Công thức chi tiết

### 5.0 Quỹ đạo và hình học

SGP4 cho vị trí và vận tốc trong hệ TEME. Đổi sang ECEF bằng phép quay quanh trục $z$ một góc GMST
(Vallado et al., 2006). Bỏ qua chuyển động cực và chênh lệch UT1−UTC:

```math
R_3(\theta) = \begin{pmatrix} \cos\theta & \sin\theta & 0 \\ -\sin\theta & \cos\theta & 0 \\ 0 & 0 & 1 \end{pmatrix},
\qquad
\mathbf r_{ECEF} = R_3(\theta_{GMST})\,\mathbf r_{TEME},
\qquad
\mathbf v_{ECEF} = R_3(\theta_{GMST})\,\mathbf v_{TEME} - \boldsymbol\omega_E \times \mathbf r_{ECEF}
```

với $\boldsymbol\omega_E = (0, 0, 7.2921158553\times10^{-5})$ rad/s. GMST tính theo Meeus (1998):

```math
\theta_{GMST} = 280.46061837^\circ + 360.98564736629^\circ\,(JD - 2451545.0) + 0.000387933^\circ\,T^2 - \frac{T^3}{38\,710\,000}^\circ,
\qquad T = \frac{JD - 2451545.0}{36525}
```

Khoảng cách và góc ngẩng từ một điểm quan sát $\mathbf r_o$ có pháp tuyến WGS84
$\hat{\mathbf n} = (\cos\varphi\cos\lambda,\ \cos\varphi\sin\lambda,\ \sin\varphi)$:

```math
d = \lVert \mathbf r_{sat} - \mathbf r_o \rVert,
\qquad
\varepsilon = \arcsin\!\left(\frac{(\mathbf r_{sat} - \mathbf r_o)\cdot\hat{\mathbf n}}{d}\right)
```

### 5.1 Link budget RF

```math
L_{FSPL}\,[\mathrm{dB}] = 20\log_{10}\!\left(\frac{4\pi d f}{c}\right)
\qquad\text{(ITU-R P.525)}
```

```math
L_{atm}\,[\mathrm{dB}] = \frac{A_{zen}}{\sin\varepsilon}
\qquad\text{(cosecant, ITU-R P.618; dùng khi } \varepsilon \gtrsim 10^\circ\text{)}
```

Khi $\varepsilon \le 0^\circ$ (không có đường truyền thẳng), $L_{atm}$ và do đó SNR, `channel_gain` được đặt là NaN, nên liên kết không khả dụng.

```math
L_{total} = L_{FSPL} + L_{atm} + L_{excess}
\qquad
N\,[\mathrm{dBm}] = -174 + 10\log_{10} B + NF
```

```math
\mathrm{SNR}\,[\mathrm{dB}] = P_{tx} + G_{tx} + G_{rx} - L_{total} - N
```

```math
g = 10^{(G_{tx} + G_{rx} - L_{total})/10}
\qquad\text{(tuyến tính, cột } \texttt{channel\_gain}\text{)}
```

```math
C = B\log_2\!\left(1 + 10^{\mathrm{SNR}/10}\right)
\qquad\text{(cận Shannon)}
```

### 5.2 Suy hao dư HAPS–UE

```math
P_{LoS}(\theta) = \frac{1}{1 + a\,\exp\!\big(-b(\theta - a)\big)},
\qquad
L_{excess} = P_{LoS}\,\xi_{LoS} + (1 - P_{LoS})\,\xi_{NLoS}
```

với $\theta$ là `elev_deg` (độ), $a = 9.61$, $b = 0.16$, $\xi_{LoS} = 1$ dB, $\xi_{NLoS} = 20$ dB.

### 5.3 FSO vệ tinh → HAPS

**Tham số**

| Tham số | Giá trị |
|---|---|
| Bước sóng | $\lambda = 1550$ nm |
| Góc phân kỳ | $\theta_{div} = 20$ µrad |
| Khẩu độ thu | $D_r = 8$ cm |
| Jitter vệ tinh | $\theta_j = 0.3$ µrad |
| Lệch vị trí phía thu | $\sigma_{Rd} = 1$ m |
| Công suất phát | $P_t = 15$ dBm |
| Hệ số đáp ứng | $\mathcal R = 0.9$ A/W |
| Độ lệch chuẩn nhiễu | $\sigma_n = 10^{-7}$ A |
| Hệ số suy hao tầng bình lưu | $\alpha = 0.02$ km⁻¹ |
| Độ dày tầng bình lưu | $H_{st} = 28$ km |
| Băng thông | $B = 1550$ MHz |

Độ dài liên kết $L$ = `sat_haps_distance_m` (slant thật). Góc thiên đỉnh $\zeta = 90^\circ - \varepsilon$.

**Suy hao tầng bình lưu** (Beer–Lambert):

```math
h_a = \exp\!\left(-\alpha H_{st}\sec\zeta\right)
```

**Sai lệch hướng (pointing error)**, với $k = 2\pi/\lambda$:

```math
w_0 = \frac{2\lambda}{\pi\theta_{div}},
\qquad
w_L = w_0\sqrt{1 + \left(\frac{2L}{k w_0^2}\right)^2}
```

```math
v = \frac{\sqrt{\pi}\,D_r}{2\sqrt2\,w_L},
\qquad
A_0 = \operatorname{erf}^2(v),
\qquad
w_{L,eq}^2 = w_L^2\,\frac{\sqrt\pi\,\operatorname{erf}(v)}{2v\,e^{-v^2}}
```

```math
\sigma_s = \sqrt{\sigma_{Rd}^2 + (\theta_j L)^2},
\qquad
\phi = \frac{w_{L,eq}}{2\sigma_s},
\qquad
A = A_0 \quad (\mu = 0,\ \text{jitter đối xứng})
```

**SNR cực đại, PDF và SNR trung bình:**

```math
\gamma_{max} = \left(\frac{\mathcal R\,P_t\,h_a\,A}{\sigma_n}\right)^2
\qquad\rightarrow\ \texttt{fso\_gamma\_max\_dB}
```

```math
f_\gamma(\gamma) = \frac{\phi^2}{2\gamma}\left(\frac{\gamma}{\gamma_{max}}\right)^{\phi^2/2},
\qquad 0 < \gamma \le \gamma_{max}
```

```math
\mathbb E[\gamma] = \int_0^{\gamma_{max}} \gamma\,f_\gamma(\gamma)\,d\gamma = \gamma_{max}\,\frac{\phi^2}{\phi^2 + 2}
\qquad\rightarrow\ \texttt{fso\_mean\_snr\_dB}
```

**Dung lượng ergodic.** Đổi biến $u = (\gamma/\gamma_{max})^{\phi^2/2}$, khi đó $f_\gamma\,d\gamma = du$:

```math
C_{FSO} = B\int_0^{\gamma_{max}} \log_2(1+\gamma)\,f_\gamma(\gamma)\,d\gamma
       = B\int_0^{1} \log_2\!\left(1 + \gamma_{max}\,u^{2/\phi^2}\right)du
\qquad\rightarrow\ \texttt{fso\_capacity\_bps}
```

Tích phân sau khi đổi biến được tính bằng cầu phương Gauss–Legendre 256 điểm. Đã đối chiếu với tích phân trực
tiếp theo dạng ban đầu: lệch tương đối $< 10^{-6}$.

### 5.4 ISAC (phương trình radar; Skolnik, Richards, Willis)

Tham số: RCS $\sigma = 1$ m², $N_p = 64$ xung, $T_p = 10$ µs, $P_{fa} = 10^{-6}$, ngưỡng khả thi 13 dB,
$F = 10^{NF/10}$.

**Monostatic** (HAPS phát và thu, $G_t = G_r = 30$ dBi):

```math
\mathrm{SNR}_s = \frac{P_t\,G_t\,G_r\,\lambda^2\,\sigma\,N_p\,T_p}{(4\pi)^3\,R^4\,k_B\,T_0\,F}
```

**Bistatic** (vệ tinh phát với $G_t = 30$ dBi; HAPS thu với $G_r = 30$ dBi):

```math
\mathrm{SNR}_s = \frac{P_t\,G_t\,G_r\,\lambda^2\,\sigma\,N_p\,T_p}{(4\pi)^3\,R_t^2\,R_r^2\,k_B\,T_0\,F}
```

Độ lợi xử lý $N_p B T_p$ triệt tiêu với nhiễu $k_B T_0 B F$, nên $B$ không có mặt trong công thức.

**Xác suất phát hiện** (Swerling 1) và **CRLB khoảng cách**:

```math
P_d = P_{fa}^{\,1/(1 + \mathrm{SNR}_s)},
\qquad
\sigma_R = \frac{c}{4\pi\,B_{rms}\sqrt{2\,\mathrm{SNR}_s}},
\qquad
B_{rms} = \frac{B}{\sqrt{12}}
```

Không tính clutter từ mặt đất, nên $\mathrm{SNR}_s$ thực tế sẽ thấp hơn.

### 5.5 Chuyển động UE (chuỗi Markov rời rạc, $\Delta t = 1$ s)

Trạng thái $s \in \{\text{stationary}, \text{pedestrian}, \text{vehicle}\}$:

```math
\Pr\{s(t+1) = j \mid s(t) = i\} = P_{ij},
\qquad
P = \begin{pmatrix} 0.980 & 0.015 & 0.005 \\ 0.020 & 0.970 & 0.010 \\ 0.010 & 0.020 & 0.970 \end{pmatrix}
```

Trạng thái ban đầu lấy theo phân bố dừng $\boldsymbol\pi P = \boldsymbol\pi$, nên $\boldsymbol\pi = (0.452,\ 0.355,\ 0.194)$.
Mỗi khi đổi trạng thái, UE nhận tốc độ mới $\lVert\mathbf v\rVert \sim \mathcal U[0,0],\ \mathcal U[1,2],\ \mathcal U[10,20]$ m/s.

```math
\psi(t+1) = \psi(t) + \mathcal N\!\left(0,\ (10^\circ)^2\right),
\qquad
\mathbf p(t+1) = \mathbf p(t) + \mathbf v(t)\,\Delta t
```

Phản xạ gương khi $\lVert\mathbf p\rVert > R_{area}$, với $\hat{\mathbf n} = \mathbf p/\lVert\mathbf p\rVert$:

```math
\mathbf p \leftarrow \hat{\mathbf n}\,\big(2R_{area} - \lVert\mathbf p\rVert\big),
\qquad
\hat{\mathbf d} \leftarrow \hat{\mathbf d} - 2(\hat{\mathbf d}\cdot\hat{\mathbf n})\,\hat{\mathbf n}
```

### 5.6 RST (Remaining Service Time)

```math
\mathrm{RST}(t_k) = t_{exit} - t_k,
\qquad
t_{exit} = \min\{\, t_m > t_k : \text{liên kết không còn available tại } t_m \,\}
```

- Thời điểm rời thật $t^\star$ nằm trong $(t_{exit} - \Delta t,\ t_{exit}]$, nên
  $0 \le \mathrm{RST} - (t^\star - t_k) < \Delta t$. Đã kiểm tra bằng skyfield với bước 0.1 s: lệch từ 0 tới 0.9 s.
- Nếu khoảng phục vụ còn kéo dài tới mẫu cuối cùng $t_{last}$ thì không biết $t_{exit}$:

```math
\mathrm{RST} = \text{NaN},\qquad \texttt{rst\_status} = 2,\qquad \texttt{rst\_lb\_s} = t_{last} + \Delta t - t_k \;\le\; \mathrm{RST}_{true}
```

- Với Sat–UE, điều kiện "available" được tính lại ở mỗi bước theo vị trí mới của UE.

---

## 6. Các đại lượng không có sẵn: cách suy ra từ dữ liệu

Ký hiệu: $g$ = `channel_gain` (tuyến tính), $N$ = `noise_power_dBm`, $B$ = `bandwidth_hz`,
$p$ là công suất phát (dBm khi viết $p_{\mathrm{dBm}}$, W khi tuyến tính).

| Cần | Cách tính từ các cột hiện có |
|---|---|
| **SNR ở công suất $p$ khác $P_{max}$** | $\mathrm{SNR_{dB}}(p) = p_{\mathrm{dBm}} + 10\log_{10} g - N$ (bảng `sat_haps` dùng các cột `rf_*` tương ứng) |
| **Công suất nhận** | $P_{rx}\,[\mathrm{dBm}] = p_{\mathrm{dBm}} + 10\log_{10} g$ |
| **SINR khi nhiều nút dùng chung băng** | UE $i$ được nút $s$ phục vụ: $\mathrm{SINR}_i = \dfrac{p_s\,g_{s,i}}{N + \sum_{j\neq s} p_j\,g_{j,i}}$ (mọi đại lượng tuyến tính). Gain chéo $g_{j,i}$ chính là `channel_gain` ở các dòng khác của cùng (`ue_id`, `t_idx`): các vệ tinh khác trong `sat_ue` và HAPS trong `haps_ue` |
| **Tốc độ khi chia băng** | $R_i = B_i\log_2\!\left(1 + \dfrac{p_i\,g_i}{N_0 B_i}\right)$, với $N_0\,[\mathrm{dBm/Hz}] = N - 10\log_{10}B$ |
| **Công suất tối thiểu để đạt `min_rate_bps`** | $p_{i,min} = \left(2^{R_{min}/B_i} - 1\right)\dfrac{N_0 B_i}{g_i}$ |
| **Dung lượng đầu-cuối qua HAPS** | $\min\!\big(C_{backhaul}(t),\ \sum_i R_i(t)\big)$, với $C_{backhaul}$ = `fso_capacity_bps` hoặc `rf_capacity_bps` của vệ tinh mà HAPS chọn tại $t$ |
| **Trễ lan truyền** (để so với `max_latency_ms`) | $\tau = d/c$. Qua HAPS: $\tau = (d_{Sat\text{–}HAPS} + d_{HAPS\text{–}UE})/c$. Trực tiếp: $\tau = d_{Sat\text{–}UE}/c$. Không tính trễ xử lý hay hàng đợi |
| **Dịch Doppler** | nối với `sat_haps` theo (`t_idx`, `sat_id`) để lấy $\mathbf v_{sat}$: $f_D = -\dfrac{\mathbf v_{sat}\cdot\hat{\mathbf u}}{\lambda}$, $\hat{\mathbf u} = \dfrac{\mathbf r_{sat} - \mathbf r_{rx}}{\lVert\mathbf r_{sat} - \mathbf r_{rx}\rVert}$. UE đi chậm ($\le 20$ m/s) nên có thể bỏ qua vận tốc UE |
| **Vị trí ECEF của UE** | `pymap3d.geodetic2ecef(lat, lon, 0)` từ bảng `ue` |
| **Vị trí vệ tinh trong `sat_ue`** | nối với `sat_haps` theo (`t_idx`, `sat_id`). Vệ tinh có $\varepsilon \ge 20^\circ$ nhìn từ UE (cách HAPS $\le 10$ km) thì luôn có trong `sat_haps` |
| **Góc thiên đỉnh** | $\zeta = 90^\circ - \varepsilon$ |
| **Fading tức thời** | $g_{inst} = g\,\lvert h\rvert^2$, với $h = \sqrt{\tfrac{K}{K+1}} + \sqrt{\tfrac{1}{K+1}}\,\mathcal{CN}(0,1)$, nên $\mathbb E\lvert h\rvert^2 = 1$ (`rf_link.rician_power_gain`) |
| **Shadowing** | $\mathrm{SNR_{dB}} + X$, với $X \sim \mathcal N(0, \sigma_{sh}^2)$ [dB] (`rf_link.shadowing_db`) |
| **SNR tức thời của FSO** | $\gamma = \gamma_{max}\,U^{2/\phi^2}$, $U \sim \mathcal U(0,1)$. $\phi$ lấy từ `fso_sat_haps.pointing_error(L)` |
| **$P_d$ với mục tiêu Swerling 0** | xấp xỉ Albersheim: `isac.pd_albersheim(isac_snr_dB)` |
| **Thời điểm handover** | $t_{HO} = t + \mathrm{RST}(t)$ của vệ tinh đang dùng. Vệ tinh thay thế: chọn trong các vệ tinh có `available` = True tại $t_{HO}$ (ví dụ RST lớn nhất) |
| **RST khi bị censored** | chỉ biết $\mathrm{RST} \ge$ `rst_lb_s`. Không nên thay bằng `rst_lb_s` như giá trị thật; nên xử lý như dữ liệu censored trong phân tích sống còn (survival analysis), hoặc chạy mô phỏng dài hơn |
| **Số vệ tinh khả dụng tại $t$** | $\lvert\{\texttt{sat\_id} : \texttt{available}\}\rvert$ theo `t_idx` (bảng `sat_haps`, hoặc `sat_ue` cho từng UE) |

---

## 7. `metadata.json`

| Khóa | Nội dung |
|---|---|
| `config` | toàn bộ tham số của lần chạy: file quỹ đạo, thời điểm bắt đầu, thời lượng, dt, số UE, seed, vị trí HAPS, các ngưỡng |
| `orbit_snapshot` | đường dẫn và SHA-256 của file quỹ đạo, số vệ tinh nạp vào / giữ lại, `rejected` (ID: lý do bị loại) |
| `ue_config` | tham số mobility (ma trận $P$, tốc độ, QoS) |
| `presets` | tham số vô tuyến, FSO và ISAC đã dùng |
| `units`, `rst_status`, `assumptions` | quy ước |
| `row_counts` | số dòng của từng bảng |
| `validation` | kết quả kiểm tra tự động: NaN/Inf, tính liên tục quỹ đạo, SNR theo khoảng cách, RST, khả dụng ⊆ nhìn thấy, UE trong vùng |
