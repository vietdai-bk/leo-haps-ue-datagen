# leo-haps-ue-datagen

Bộ sinh dữ liệu chuỗi thời gian (time-series) cho mạng tích hợp **vệ tinh LEO Starlink – HAPS – người dùng (UE)**.
Dữ liệu phục vụ nghiên cứu cấp phát tài nguyên (resource allocation, power allocation), lập lịch theo QoS
và handover trong mạng không gian – không trung – mặt đất (SAGIN).

> **Chỉ quỹ đạo vệ tinh là dữ liệu thật** (TLE/OMM của CelesTrak, tính bằng SGP4).
> Mọi thông số khác (vô tuyến, anten, QoS, chuyển động UE, ISAC) là **giả định mô phỏng**.
> Các giả định được liệt kê bên dưới và ghi vào `metadata.json` ở mỗi lần chạy.

## Repo sinh ra dữ liệu gì

Với mỗi bước thời gian (mặc định 1 s), repo tính:

| Nhóm | Nội dung |
|---|---|
| **Quỹ đạo vệ tinh** | vị trí, vận tốc (ECEF), khoảng cách, góc ngẩng, góc phương vị của từng vệ tinh Starlink so với HAPS và so với từng UE |
| **Người dùng (UE)** | quỹ đạo di chuyển theo **chuỗi Markov** (đứng yên / đi bộ / xe), nhãn **Critical / Normal** và hồ sơ QoS (độ trễ, tốc độ tối thiểu, độ tin cậy, mức ưu tiên) |
| **Kênh truyền** | SNR, độ lợi kênh, công suất nhiễu, băng thông, công suất phát tối đa và dung lượng của 3 loại liên kết: vệ tinh → HAPS (lưu **cả FSO lẫn RF**), vệ tinh → UE và HAPS → UE |
| **ISAC** | SNR sensing, xác suất phát hiện, sai số đo khoảng cách (CRLB): HAPS monostatic và vệ tinh → HAPS bistatic |
| **Remaining Service Time (RST)** | thời gian còn lại mà vệ tinh còn phục vụ được HAPS / từng UE, có đánh dấu trường hợp bị cắt ở cuối cửa sổ mô phỏng (censored) |
| **Khả dụng** | phân biệt *nhìn thấy về hình học* (góc ngẩng ≥ 0°) với *liên lạc được* (góc ngẩng ≥ 30°, SNR ≥ ngưỡng) |

Dữ liệu chứa đủ đầu vào để tự tính lại kết quả của một thuật toán cấp phát công suất / băng thông
(SNR ở công suất bất kỳ, SINR khi có nhiễu, tốc độ, ràng buộc QoS và backhaul). Xem mục 6 trong
[`docs/DATA_DICTIONARY.md`](docs/DATA_DICTIONARY.md).

## Kịch bản mô phỏng: vị trí HAPS, UE và vệ tinh

```
             Vệ tinh Starlink (~350–560 km, ~8 100 vệ tinh thật)
                ✦        ✦            ✦         ✦
                  \      |  FSO / RF  |        /
                   \     | (backhaul) |       /    RF trực tiếp
                    \    v            v      /     (Sat -> UE)
                         ■  HAPS (20 km, đứng yên)
                        /|\
              RF (HAPS -> UE) + ISAC
                      /  |  \
        ── ● ── ▲ ── ■ ──┼── ● ── ▲ ──  mặt đất, hình tròn bán kính 10 km
              UE di chuyển theo chuỗi Markov
```

| Thành phần | Vị trí / cấu hình mặc định |
|---|---|
| **HAPS** | đứng yên tại **35.6762°N, 139.6503°E** (Tokyo), độ cao **20 km**. Đổi được qua `haps=dict(lat, lon, alt_m)` |
| **UE** | **50 UE** ở mặt đất (độ cao 0 m), phân bố đều trong **hình tròn bán kính 10 km** quanh điểm chiếu của HAPS. **20 % Critical**, 80 % Normal. Di chuyển theo chuỗi Markov 3 trạng thái (0, 1–2, 10–20 m/s); chạm biên thì phản xạ lại. Có seed để tái lập kết quả |
| **Vệ tinh** | toàn bộ chòm Starlink trong file `data/STARLINK_20250810.txt` (8 111 vệ tinh, epoch 2025-08-10; 8 110 vệ tinh qua kiểm tra). Phần lớn ở độ cao 540–560 km, một phần nhỏ ở lớp khoảng 350 km. Vệ tinh có epoch lệch quá 3 ngày, lỗi SGP4 hoặc không còn ở quỹ đạo LEO đều bị loại |
| **Thời gian** | bắt đầu **2025-08-10 06:00 UTC** (vài giờ sau epoch của TLE, nên sai số SGP4 nhỏ), kéo dài **1 giờ**, bước **1 s** |
| **Ngưỡng liên lạc** | góc ngẩng **≥ 30°** cho cả HAPS và UE |

Với cấu hình mặc định, tại mỗi giây có khoảng **25–46 vệ tinh** (trung bình khoảng 36) ở góc ngẩng ≥ 30° so với HAPS.
Khoảng cách vệ tinh–HAPS từ khoảng 300 km (vệ tinh lớp thấp gần thiên đỉnh) tới khoảng 1 000 km (góc ngẩng 30°). Mỗi lượt bay khả dụng kéo dài từ vài giây tới khoảng 4 phút.

## Chạy nhanh

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt

# kiểm tra nhanh: 5 phút mô phỏng, 10 UE, vài giây
.venv/bin/python -m leo_haps_ue_datagen.build_dataset --quick --out output/quick

# kịch bản mặc định: 1 giờ, bước 1 s, 50 UE (khoảng 1 GB Parquet, 6 GB RAM, 1–2 phút)
.venv/bin/python -m leo_haps_ue_datagen.build_dataset --out output/run

# chạy test
.venv/bin/python -m pytest tests
```

**Google Colab:** mở [`notebooks/run_on_colab.ipynb`](notebooks/run_on_colab.ipynb), sửa `REPO_URL` rồi chạy tất cả các ô.
Notebook cài thư viện, chạy test, sinh dữ liệu (bản nhanh và bản đầy đủ), xem nhanh dữ liệu, hiển thị hình động
(bầu trời nhìn từ HAPS, UE di chuyển, quỹ đạo 3D, timeline khả dụng), rồi lưu kết quả vào Drive hoặc tải về dạng zip.

Cần file **CSV** (mở bằng Excel) thay vì Parquet? Dùng [`notebooks/run_on_colab_csv.ipynb`](notebooks/run_on_colab_csv.ipynb). Notebook này giống hệt bản trên, chỉ khác định dạng lưu. CSV nặng hơn khoảng 3.7 lần: bản mặc định khoảng 4.3 GB, so với 1.2 GB của Parquet.

**Tham số dòng lệnh:**

| Tham số | Ý nghĩa |
|---|---|
| `--orbit` | file TLE (`.txt`) hoặc OMM (`.json`) |
| `--start` | thời điểm bắt đầu, UTC, ISO 8601 |
| `--duration` | thời lượng [s] |
| `--n-ue` | số UE |
| `--seed` | seed ngẫu nhiên |
| `--fmt` | `parquet` hoặc `csv` |
| `--out` | thư mục kết quả |
| `--quick` | chạy bản kiểm tra nhanh |

Lệnh trả về mã thoát 1 nếu có kiểm tra validation không đạt.

**Dùng quỹ đạo mới hơn:** tải snapshot OMM rồi truyền vào `--orbit`. Chọn `--start` cách epoch không quá 3 ngày.
CelesTrak chỉ cập nhật dữ liệu 2 giờ một lần và từ chối tải lặp lại trong khoảng đó.

```bash
curl -o data/starlink_$(date -u +%Y%m%dT%H%M%SZ).json \
  "https://celestrak.org/NORAD/elements/gp.php?GROUP=starlink&FORMAT=JSON"
```

## Dữ liệu đầu ra

| File | Mỗi dòng là | Nội dung chính |
|---|---|---|
| `sat_haps.parquet` | (vệ tinh, t), góc ngẩng nhìn từ HAPS ≥ 0° | vị trí và vận tốc vệ tinh, vị trí HAPS, khoảng cách, góc ngẩng / phương vị, `geometric_visible`, `available`, RST, backhaul **FSO** (`fso_capacity_bps`, `fso_mean_snr_dB`, `fso_gamma_max_dB`) và **RF** (`rf_snr_dB`, `rf_capacity_bps`, `rf_channel_gain`, `rf_noise_power_dBm`, `rf_bandwidth_hz`, `rf_p_max_dBm`) |
| `sat_ue.parquet` | (vệ tinh, UE, t), góc ngẩng nhìn từ UE ≥ 20° | khoảng cách, góc ngẩng, `snr_sat_ue_dB`, `channel_gain`, `noise_power_dBm`, `bandwidth_hz`, `p_max_dBm`, `capacity_bps`, `available`, RST, ISAC bistatic |
| `haps_ue.parquet` | (UE, t) | khoảng cách, góc ngẩng, `snr_haps_ue_dB`, các đầu vào cho power allocation, `capacity_bps`, ISAC monostatic |
| `ue.parquet` | (UE, t) | quỹ đạo (ENU và lat/lon), vận tốc, trạng thái di chuyển, `user_type`, QoS |
| `metadata.json` | – | cấu hình, seed, file quỹ đạo và SHA-256, vệ tinh bị loại, preset, đơn vị, giả định, kết quả validation |
| `validation.png` | – | số vệ tinh khả dụng, khoảng cách vệ tinh–HAPS, SNR, RST, quỹ đạo UE, phân bố SNR theo loại UE |

- Nối các bảng theo `t_idx` / `timestamp`, `sat_id` (số NORAD) và `ue_id`.
- **Mô tả chi tiết từng cột, công thức (LaTeX) và cách suy ra các đại lượng không lưu sẵn**: [`docs/DATA_DICTIONARY.md`](docs/DATA_DICTIONARY.md).
- Đơn vị: m, m/s, độ, dB, dBm, bit/s, Hz, s. Thời gian theo UTC.
- Giá trị thiếu là **NaN, không bao giờ là 0**:
  - RST là NaN khi không phục vụ hoặc khi bị censored;
  - các cột FSO là NaN khi góc ngẩng dưới ngưỡng.
- `rst_status`:
  - `0`: không phục vụ;
  - `1`: đang phục vụ, `rst_*_s` chính xác tới 1 bước thời gian;
  - `2`: censored, vẫn phục vụ ở cuối cửa sổ; `rst_lb_s` là cận dưới.
- Các cột SNR tính ở `p_max_dBm`. Với công suất p bất kỳ: SNR = p · `channel_gain` / nhiễu.
- Các dòng của `sat_haps` có góc ngẩng 0–20° **không phải danh sách đầy đủ** mọi vệ tinh trên đường chân trời;
  chỉ gồm các vệ tinh ứng viên có lúc lên tới khoảng 20°.

## Cấu trúc code

| Module | Nội dung |
|---|---|
| `orbit.py` | đọc và kiểm tra TLE/OMM (tuổi epoch, lỗi SGP4, độ cao LEO, trùng ID); SGP4 → TEME → ECEF qua GMST (Vallado et al. 2006); vận tốc ECEF; tìm lượt bay theo hai bước (lọc thô rồi tính mịn) |
| `visibility.py` | đọc TLE, GMST, và định dạng visibility cũ `(name, elev, vt, slant, cfso, ENU)` của project gốc |
| `haps_ue.py` | HAPS; chuyển động Markov của UE, hướng đi random walk, phản xạ ở biên; hồ sơ QoS Critical / Normal |
| `rf_link.py` | link budget, FSPL, suy hao khí quyển (cosecant), xác suất LoS Al-Hourani (HAPS–UE), shadowing, fading Rician, SINR, dung lượng Shannon |
| `fso_sat_haps.py` | FSO vệ tinh → HAPS: suy hao tầng bình lưu Beer–Lambert, mô hình sai lệch hướng (Le et al., IEEE TAES 2023), dung lượng ergodic tính bằng phép đổi biến chính xác (không NaN), SNR trung bình |
| `isac.py` | phương trình radar: HAPS monostatic (R⁴) và vệ tinh → HAPS bistatic (Rt²Rr²); P_d Swerling 1 / Albersheim; CRLB khoảng cách |
| `rst.py` | RST có xử lý censored; phân biệt nhìn thấy và khả dụng; hỗ trợ UE di chuyển |
| `build_dataset.py` | ghép dữ liệu, validation, biểu đồ, dòng lệnh |
| `viz.py` | hình động bầu trời nhìn từ HAPS, UE di chuyển, quỹ đạo 3D (plotly), timeline khả dụng. `greedy_selection` là một chính sách chọn vệ tinh **chỉ để minh họa** |

## Giả định chính (mặc định)

| Hạng mục | Giá trị | Nguồn / ghi chú |
|---|---|---|
| HAPS–UE | 2 GHz, 600 MHz, 40 dBm, anten mảng HAPS 30 dBi, N = −80.4 dBm | 30 dBi tương ứng mảng khoảng 256 phần tử |
| Vệ tinh–UE | 2 GHz, 30 MHz, EIRP 78.77 dBm (30 dBi), UE 0 dBi, NF 7 dB | 3GPP TR 38.821 LEO-600 S-band; **cần đối chiếu lại với TR** |
| Vệ tinh–HAPS (RF) | 20 GHz, 400 MHz, 40 dBm, 38.5 / 40 dBi | giả định |
| Vệ tinh–HAPS (FSO) | λ = 1550 nm, θ_div = 20 µrad, P_t = 15 dBm, B = 1550 MHz, D_r = 8 cm, α = 0.02 /km, H_st = 28 km | σ_Rd = 1 m và hệ số đáp ứng 0.9 A/W **chưa rõ nguồn** |
| ISAC | RCS 1 m², N_p = 64, T_p = 10 µs, P_fa = 10⁻⁶, khả thi khi SNR ≥ 13 dB | chưa tính clutter mặt đất |
| Chuyển động UE | tốc độ 0 / 1–2 / 10–20 m/s; ma trận chuyển trạng thái `haps_ue.DEFAULT_P` | cho bước 1 s |
| QoS | Critical: 10 ms, 1 Mbps, 99.999 %, ưu tiên 1. Normal: 100 ms, 5 Mbps, 99 %, ưu tiên 5 | tham khảo 3GPP TS 22.261 / 23.501 |