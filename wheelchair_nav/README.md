# Smart Wheelchair Autonomous Navigation (RT-MonoDepth + YOLO26-nano)

Sistem navigasi *indoor* untuk kursi roda pintar (pencahayaan normal) yang dibangun
**di atas** RT-MonoDepth (model *full*, arsitektur asli repo ini, **tidak diubah**)
dan YOLO26-nano untuk bounding-box obstacle. Semua kode di direktori ini
(`wheelchair_nav/`) adalah tambahan baru; tidak ada file di `networks/`, `layers.py`,
`options.py`, dsb. yang dimodifikasi.

```
Perception:  frame -> RT-MonoDepth (depth_map, metres)
             frame -> YOLO26-nano  (bbox per obstacle, class-agnostic)
             -> Obstacle List: depth_m = median(depth_map[bbox_inner_ROI])

Navigation:  Obstacle List -> Sector-Based Free-Path Selection
             (FL0 | L1 | CTR2 | R3 | FR4, priority CTR>L>R>FL>FR>STOP,
              hysteresis N=3 majority vote)
             -> Decision: FORWARD | TURN_LEFT | TURN_RIGHT | STOP
```

Safety default: obstacle dengan jarak **< 1.0 m** (`SAFE_DISTANCE_M`, lihat
`wheelchair_nav/config.py`) membuat sektor tersebut dianggap terhalang; sistem lalu
memilih sektor bebas dengan prioritas tertinggi. Jika **tidak ada** sektor yang bebas,
atau ada obstacle darurat (< 0.5 m, `EMERGENCY_DISTANCE_M`) di sektor manapun, sistem
langsung **STOP** (bypass hysteresis, tanpa delay).

## Isi direktori

```
wheelchair_nav/
  config.py                        thresholds, sektor, ukuran input
  datasets/sunrgbd_dataset.py       PyTorch Dataset RGB + metric-depth SUN RGB-D
  scripts/
    prepare_sunrgbd.py               download-agnostic preprocessing SUN RGB-D
    tune_depth_sunrgbd.py             Optuna-TPE hyperparameter search for RT-MonoDepth
    train_depth_sunrgbd.py           training RT-MonoDepth (full) from scratch
    tune_yolo_obstacle.py             Optuna-TPE hyperparameter search for YOLO26-nano
    train_yolo_obstacle.py           fine-tuning YOLO26-nano (class-agnostic)
  perception/
    depth_estimator.py                wrapper inferensi RT-MonoDepth (full, unmodified)
    obstacle_detector.py              wrapper YOLO26-nano (bbox saja, tanpa recognition)
    obstacle_list.py                  fusi bbox + depth map -> Obstacle List
  navigation/
    sectors.py                        partisi FOV jadi 5 sektor FL0..FR4
    free_path.py                      priority selection + hysteresis N=3
    controller.py                     decision -> simulated drive command
  run_navigation.py                  entry point: uji sistem pada file video
  evaluation/
    eval_depth_metrics.py             abs_rel/sq_rel/rmse/rmse_log/a1-a3 + params + FPS
    eval_detection_metrics.py         mAP50/mAP50-95/precision/recall + params + FPS
    eval_navigation_metrics.py        FPS pipeline, missed/false-stop rate,
                                       decision accuracy, distance MAE/RMSE
  requirements.txt
```

---

## 0. Setup environment

```bash
cd RT-MonoDepth
python3 -m venv .venv && source .venv/bin/activate      # opsional tapi disarankan
pip install -r wheelchair_nav/requirements.txt
```

Semua perintah di bawah dijalankan dari **root repo** (`RT-MonoDepth/`), memakai
`python -m wheelchair_nav....` supaya import ke `layers.py` / `networks/` (di root)
dan ke package `wheelchair_nav` sama-sama beres.

---

## 1. Download & preprocessing dataset depth indoor (SUN RGB-D)

1. Download SUN RGB-D dari https://rgbd.cs.princeton.edu/ (`SUNRGBD.zip`, ~7GB) lalu
   ekstrak. Hasil ekstraksi harus punya struktur:

   ```
   SUNRGBD/
     kv1/...       (Kinect v1 scenes)
     kv2/...       (Kinect v2 scenes)
     realsense/... (RealSense scenes)
     xtion/...     (Xtion scenes)
   ```

   Setiap folder scene berisi `image/*.jpg`, `depth_bfx/*.png` (depth hasil refine
   sensor, dipakai script ini) dan `intrinsics.txt`.

2. Jalankan preprocessing untuk pasangan RGB/metric-depth (dipakai training
   RT-MonoDepth):

   ```bash
   python -m wheelchair_nav.scripts.prepare_sunrgbd \
       --sunrgbd_root /path/to/SUNRGBD \
       --out_dir ./data/sunrgbd_processed \
       --splits_dir ./splits_sunrgbd
   ```

   Script ini men-decode depth PNG 16-bit Kinect (bit-rotate 3-bit, standar
   SUNRGBDtoolbox) jadi metric depth dalam meter (`.npy`), lalu membagi otomatis jadi
   `train.txt` / `val.txt` / `test.txt` (80/10/10, seed=42) berisi baris
   `<path_rgb> <path_depth_npy>`. Jika hasil decode kebanyakan nol/aneh untuk sensor
   tertentu (variasi rilis dataset), coba `--depth_encoding raw_mm`.

3. (Opsional, untuk fine-tuning YOLO26-nano di langkah 5) sekaligus konversi anotasi
   2D SUN RGB-D jadi label YOLO **satu kelas** (`obstacle`), class-agnostic:

   ```bash
   python -m wheelchair_nav.scripts.prepare_sunrgbd \
       --sunrgbd_root /path/to/SUNRGBD \
       --out_dir ./data/sunrgbd_processed \
       --splits_dir ./splits_sunrgbd \
       --make_yolo_labels --yolo_out_dir ./data/sunrgbd_yolo
   ```

   Format `annotation2Dfinal/index.json` sedikit berbeda antar rilis SUN RGB-D;
   scene yang tidak cocok skema akan dilewati (bukan menghentikan seluruh proses).
   Jika hasil konversi terlalu sedikit, langsung pakai opsi A di langkah 5
   (fine-tune dari bobot COCO-pretrained) tanpa label SUN RGB-D.

---

## 2. Hyperparameter tuning RT-MonoDepth (Optuna, TPE sampler)

Dijalankan **sebelum** training penuh di langkah 3. `scripts/tune_depth_sunrgbd.py`
membuat `optuna.Study` dengan `TPESampler` (Tree-structured Parzen Estimator) dan
`MedianPruner` (menghentikan trial yang jelas buruk lebih awal). Tiap trial melatih
`DepthEncoder`+`DepthDecoder` yang baru diinisialisasi acak (arsitektur sama persis
dengan langkah 3, **from scratch**, bukan pretrained) untuk beberapa epoch singkat
(`--epochs_per_trial`, proxy budget) pada train/val split yang sama, lalu diberi skor
memakai validation masked-L1 depth error (meter) -- metrik yang sama yang dicetak
`train_depth_sunrgbd.py` tiap epoch.

Hyperparameter yang dicari: `learning_rate`, `batch_size`, `smoothness_weight`,
`si_lambda` (bobot scale-invariant log loss), `weight_decay`.

```bash
python -m wheelchair_nav.scripts.tune_depth_sunrgbd \
    --splits_dir ./splits_sunrgbd \
    --height 192 --width 640 \
    --n_trials 30 --epochs_per_trial 5 \
    --out_json ./log_sunrgbd/optuna_best_depth_hparams.json \
    --device cuda
```

- `--n_trials` -- jumlah kombinasi hyperparameter yang dicoba TPE (disarankan 20-50).
- `--epochs_per_trial` -- budget proxy per trial, jauh lebih kecil dari
  `--num_epochs` training penuh supaya pencarian tetap cepat.
- `--storage sqlite:///./log_sunrgbd/optuna_depth.db` (opsional) -- simpan progres
  studi supaya bisa dilanjutkan/dijalankan paralel di beberapa proses.

Hasil: `./log_sunrgbd/optuna_best_depth_hparams.json` (nilai val-L1 terbaik +
hyperparameter terbaik) dan `..._trials.csv` (riwayat seluruh trial). Format JSON-nya
sudah cocok langsung dipakai flag `--hparams_json` di langkah 3.

---

## 3. Training RT-MonoDepth (full, from scratch) di SUN RGB-D

Arsitektur **tidak diubah** (`networks/RTMonoDepth/RTMonoDepth.py`: `DepthEncoder` +
`DepthDecoder`, versi *full*, bukan `RTMonoDepth_s`). Training di sini **supervised**
langsung ke ground-truth depth SUN RGB-D (bukan self-supervised video seperti
`trainer.py` di root, karena SUN RGB-D adalah snapshot RGB-D statis, bukan video
ber-pose), memakai loss masked-L1 + scale-invariant log (Eigen) + smoothness (reuse
`layers.get_smooth_loss`, tidak dimodifikasi). Bobot diinisialisasi acak (Kaiming
init bawaan kelasnya) -- **tidak ada checkpoint pretrained yang dimuat**.

Pakai hyperparameter hasil tuning langkah 2 langsung lewat `--hparams_json` (meng-*override*
`--learning_rate`/`--batch_size`/`--smoothness_weight`/`--si_lambda`/`--weight_decay`):

```bash
python -m wheelchair_nav.scripts.train_depth_sunrgbd \
    --splits_dir ./splits_sunrgbd \
    --log_dir ./log_sunrgbd \
    --model_name RTMonoDepth_sunrgbd \
    --height 192 --width 640 \
    --min_depth 0.1 --max_depth 10.0 \
    --num_epochs 40 \
    --hparams_json ./log_sunrgbd/optuna_best_depth_hparams.json
```

atau tanpa hasil tuning, set manual seperti biasa:

```bash
python -m wheelchair_nav.scripts.train_depth_sunrgbd \
    --splits_dir ./splits_sunrgbd \
    --log_dir ./log_sunrgbd \
    --model_name RTMonoDepth_sunrgbd \
    --height 192 --width 640 \
    --min_depth 0.1 --max_depth 10.0 \
    --batch_size 16 --num_epochs 40 --learning_rate 1e-4
```

Checkpoint tiap epoch + checkpoint terbaik (val L1 terendah) disimpan di
`./log_sunrgbd/RTMonoDepth_sunrgbd/models/{weights_N,best}/{encoder.pth,depth.pth}`
-- format sama seperti `test_simple_full.py` di root (`encoder.pth` menyimpan juga
`height`/`width`), sehingga bisa langsung dipakai `wheelchair_nav.perception.DepthEstimator`.

---

## 4. Hyperparameter tuning YOLO26-nano (Optuna, TPE sampler)

Dijalankan **sebelum** fine-tuning penuh di langkah 5. Sama pola dengan langkah 2:
`scripts/tune_yolo_obstacle.py` memakai `optuna.Study` + `TPESampler`, tiap trial
melakukan fine-tuning singkat (`--epochs_per_trial`) dari `--pretrained` pada
`--data`, lalu diberi skor dari validation mAP50-95 (dimaksimalkan) -- dibaca dengan
cara yang sama seperti `evaluation/eval_detection_metrics.py`.

Hyperparameter yang dicari: optimizer (`lr0`, `lrf`, `momentum`, `weight_decay`,
`warmup_epochs`), bobot loss (`box`, `cls`), dan augmentasi (`hsv_h`, `hsv_s`, `hsv_v`,
`translate`, `scale`, `fliplr`, `mosaic`).

```bash
python -m wheelchair_nav.scripts.tune_yolo_obstacle \
    --data ./data/sunrgbd_yolo/obstacle.yaml \
    --pretrained yolo26n.pt \
    --n_trials 30 --epochs_per_trial 10 --imgsz 640 --batch 32 --device 0 \
    --out_json ./log_yolo/optuna_best_yolo_hparams.json
```

Hasil: `./log_yolo/optuna_best_yolo_hparams.json` (mAP50-95 terbaik + hyperparameter
terbaik) dan `..._trials.csv` (riwayat seluruh trial); artefak training tiap trial
ada di `./log_yolo/optuna_tuning/trial_XXX/`. JSON-nya sudah cocok langsung dipakai
flag `--hparams_json` di langkah 5.

---

## 5. Training / fine-tuning YOLO26-nano (deteksi bbox, class-agnostic)

YOLO26-nano dipakai **hanya** untuk bbox per obstacle; identitas kelas (nama objek)
selalu dibuang di `perception/obstacle_detector.py`, jadi tidak perlu recognition
nama objek. Dua opsi:

**Opsi A (disarankan): fine-tune dari bobot COCO-pretrained, pakai hasil tuning langkah 4**

```bash
python -m wheelchair_nav.scripts.train_yolo_obstacle \
    --data ./data/sunrgbd_yolo/obstacle.yaml \
    --pretrained yolo26n.pt \
    --epochs 60 --imgsz 640 --batch 32 --device 0 \
    --hparams_json ./log_yolo/optuna_best_yolo_hparams.json
```

**Opsi B: training dari bobot acak (tanpa COCO), tanpa hasil tuning**

```bash
python -m wheelchair_nav.scripts.train_yolo_obstacle \
    --data ./data/sunrgbd_yolo/obstacle.yaml \
    --pretrained "" --epochs 200 --imgsz 640 --batch 32 --device 0
```

Bobot hasil training ada di `./log_yolo/obstacle_yolo26n/weights/best.pt`. Jika label
SUN RGB-D dari langkah 1.3 terlalu sedikit/tidak tersedia, `yolo26n.pt`
(COCO-pretrained) langsung bisa dipakai apa adanya di langkah 6 tanpa fine-tuning --
sistem navigasi tetap berjalan class-agnostic karena nama kelas dibuang.

---

## 6. Testing sistem navigasi pada file video

```bash
python -m wheelchair_nav.run_navigation \
    --video ./data/test_indoor.mp4 \
    --depth_weights ./log_sunrgbd/RTMonoDepth_sunrgbd/models/best \
    --yolo_weights ./log_yolo/obstacle_yolo26n/weights/best.pt \
    --output ./out/navigation_demo.mp4 \
    --device cuda --safe_distance 1.0
```

Output:
- `./out/navigation_demo.mp4` -- video ber-anotasi: kotak bbox + label jarak
  (`Depth: X.XXm`, merah jika < `safe_distance`), garis 5 sektor, banner
  `Decision: FORWARD/TURN_LEFT/TURN_RIGHT/STOP`, FPS, dan *picture-in-picture*
  peta depth berwarna (mirip visualisasi contoh di kiri).
- `./out/navigation_demo_log.csv` -- log per-frame (`frame, decision,
  num_obstacles, min_depth_m, fps, FL0, L1, CTR2, R3, FR4`), dipakai evaluasi
  end-to-end di langkah 7.

Tanpa `--yolo_weights` (default `yolo26n.pt`) sistem otomatis pakai bobot
COCO-pretrained Ultralytics, cukup untuk demo cepat.

---

## 7. Matriks evaluasi

### 7a. Depth: akurasi, error, jumlah parameter, FPS

```bash
python -m wheelchair_nav.evaluation.eval_depth_metrics \
    --weights_dir ./log_sunrgbd/RTMonoDepth_sunrgbd/models/best \
    --test_list ./splits_sunrgbd/test.txt \
    --device cuda
```

Mencetak `abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3` (definisi sama dengan
`evaluate_depth_full.py` di root repo), jumlah parameter encoder+decoder, dan FPS
inferensi murni (warmup + averaged cycles, mengikuti pola `compare_runtime.py`).

### 7b. Deteksi obstacle: mAP, precision, recall, parameter, FPS

```bash
python -m wheelchair_nav.evaluation.eval_detection_metrics \
    --weights ./log_yolo/obstacle_yolo26n/weights/best.pt \
    --data ./data/sunrgbd_yolo/obstacle.yaml \
    --device 0
```

### 7c. Sistem navigasi end-to-end: FPS pipeline, keamanan, akurasi keputusan, error jarak

```bash
python -m wheelchair_nav.evaluation.eval_navigation_metrics \
    --nav_log ./out/navigation_demo_log.csv \
    --safe_distance 1.0 \
    --gt_decisions ./data/gt_decisions.csv \
    --gt_distances ./data/gt_distances.json \
    --video ./data/test_indoor.mp4 \
    --depth_weights ./log_sunrgbd/RTMonoDepth_sunrgbd/models/best
```

Selalu dihitung dari log: FPS pipeline (mean/min/max), **missed-stop rate** (obstacle
< threshold tapi sistem tetap FORWARD -- risiko tabrakan) dan **false-stop rate**
(sistem STOP padahal tidak ada obstacle < threshold -- terlalu konservatif).

Opsional dengan ground truth:
- `--gt_decisions` (CSV `frame,decision`) -> akurasi keputusan + confusion matrix.
- `--gt_distances` (JSON list `{"frame", "bbox":[x1,y1,x2,y2], "distance_m"}`, jarak
  hasil ukur manual/tape-measure pada beberapa frame referensi) + `--video` +
  `--depth_weights` -> MAE/RMSE estimasi jarak RT-MonoDepth+YOLO terhadap jarak
  sebenarnya.

---

## Konfigurasi

Semua threshold ada di `wheelchair_nav/config.py`:

| Parameter | Default | Keterangan |
|---|---|---|
| `SAFE_DISTANCE_M` | 1.0 | jarak aman minimum per sektor |
| `EMERGENCY_DISTANCE_M` | 0.5 | jarak darurat, bypass hysteresis, STOP seketika |
| `MIN_DEPTH_M` / `MAX_DEPTH_M` | 0.1 / 10.0 | rentang depth metric indoor |
| `BBOX_INNER_RATIO` | 0.6 | bbox di-*shrink* ke 60% tengah sebelum median depth |
| `SECTOR_NAMES` | FL0,L1,CTR2,R3,FR4 | 5 sektor kiri->kanan |
| `HYSTERESIS_WINDOW` | 3 | N frame majority-vote |

## Catatan / batasan

- RT-MonoDepth (`networks/RTMonoDepth/RTMonoDepth.py`) tidak diubah sama sekali;
  yang baru hanyalah training objective (supervised, bukan self-supervised) dan
  wrapper inferensi.
- Tidak ada hardware kursi roda fisik dalam repo ini -- `navigation/controller.py`
  hanya menghasilkan simulated drive command (`linear_mps`, `angular_radps`) untuk
  logging/visualisasi; ganti isinya dengan driver motor sungguhan bila diintegrasikan
  ke perangkat keras.
- Parsing anotasi 2D SUN RGB-D (`annotation2Dfinal/index.json`) bersifat best-effort
  karena format sedikit berbeda antar rilis dataset; opsi fine-tuning YOLO26-nano
  dari bobot COCO-pretrained (Opsi A langkah 5) tidak bergantung pada langkah ini.
- Hyperparameter tuning (langkah 2 dan 4) memakai Optuna dengan `TPESampler`
  (Bayesian, Tree-structured Parzen Estimator) dan budget epoch yang sengaja lebih
  kecil dari training penuh -- ini proxy search, bukan pengganti training penuh;
  hasil terbaiknya tetap perlu dilatih ulang dengan `--num_epochs`/`--epochs` penuh
  di langkah 3/5.
