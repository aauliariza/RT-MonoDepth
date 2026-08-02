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
             (FL | L | CTR | R | FR, priority CTR>L>R>FL>FR>STOP,
              hysteresis N=3 majority vote)
             -> Decision: FORWARD | TURN_LEFT | TURN_RIGHT |
                          TURN_FAR_LEFT | TURN_FAR_RIGHT | STOP
```

Safety default: obstacle dengan jarak **< 1.0 m** (`SAFE_DISTANCE_M`, lihat
`wheelchair_nav/config.py`) membuat sektor tersebut dianggap terhalang; sistem lalu
memilih sektor bebas dengan prioritas tertinggi (`CTR > L > R > FL > FR`), masing-masing
sektor memetakan ke decision-nya sendiri: `CTR->FORWARD`, `L->TURN_LEFT`,
`R->TURN_RIGHT`, `FL->TURN_FAR_LEFT`, `FR->TURN_FAR_RIGHT`. Jika **tidak ada** sektor
yang bebas, atau ada obstacle darurat (< 0.5 m, `EMERGENCY_DISTANCE_M`) di sektor
manapun, sistem langsung **STOP** (bypass hysteresis, tanpa delay). Video output
`run_navigation.py` mewarnai tiap sektor sesuai statusnya: **hijau** = sektor yang
dipilih (jalur aman yang dilewati), **kuning/amber** = bebas tapi tidak dipilih
(prioritas lebih rendah), **merah** = terhalang.

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
    tune_fastdepth_sunrgbd.py         Optuna-TPE hyperparameter search for FastDepth (baseline)
    train_fastdepth_sunrgbd.py       training FastDepth (baseline) from scratch
    tune_yolo_depth.py                Optuna-TPE hyperparameter search for YOLO26{n,s}-depth (baseline)
    train_yolo_depth.py              training YOLO26n-depth / YOLO26s-depth (baseline)
  perception/
    depth_estimator.py                wrapper inferensi RT-MonoDepth (full, unmodified)
    obstacle_detector.py              wrapper YOLO26-nano (bbox saja, tanpa recognition)
    obstacle_list.py                  fusi bbox + depth map -> Obstacle List
  baselines/                        model depth pembanding (apples-to-apples), lihat langkah 8
    fastdepth_estimator.py            wrapper inferensi FastDepth (unmodified)
    yolo_depth_estimator.py           wrapper inferensi YOLO26{n,s}-depth (unmodified)
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
    eval_depth_comparison.py          RT-MonoDepth vs FastDepth vs YOLO26{n,s}-depth,
                                       metrik+params+FPS identik, test split identik
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

3. **Wajib** untuk training YOLO26-nano di langkah 5: konversi anotasi 2D bawaan
   SUN RGB-D sendiri (`annotation2Dfinal/index.json`) jadi label YOLO **satu kelas**
   (`obstacle`), class-agnostic. **Ini satu-satunya dataset yang dipakai untuk
   training deteksi obstacle** -- tidak ada dataset lain (COCO dsb.) yang dipakai:

   ```bash
   python -m wheelchair_nav.scripts.prepare_sunrgbd \
       --sunrgbd_root /path/to/SUNRGBD \
       --out_dir ./data/sunrgbd_processed \
       --splits_dir ./splits_sunrgbd \
       --make_yolo_labels --yolo_out_dir ./data/sunrgbd_yolo
   ```

   Skema `annotation2Dfinal/index.json` yang dipakai (diverifikasi terhadap
   *community parser* SUN RGB-D, mis. `Mask_RCNN-for-SUN-RGB-D`):
   `frames[0]["polygon"][i] = {"x": [...], "y": [...], "object": idx}` dengan
   `objects[idx]["name"]` sebagai nama kelas asli SUN RGB-D. Kelas permukaan ruangan
   yang bukan obstacle fisik (`wall`, `floor`, `ceiling`, diatur lewat
   `--exclude_classes`) dibuang -- kalau tidak, poligon dinding/lantai/langit-langit
   akan jadi kotak nyaris satu-frame-penuh dan meracuni training. `--max_box_area_ratio`
   (default `0.9`) jadi jaring pengaman kedua untuk poligon oversized lain yang lolos
   dari filter nama kelas. Scene tanpa `annotation2Dfinal/` atau JSON yang tidak
   valid dilewati dan dihitung (bukan diganti dataset lain).

   Otomatis dibagi `train`/`val`/`test` (rasio sama dengan `--val_ratio`/`--test_ratio`)
   ke `./data/sunrgbd_yolo/{images,labels}/{train,val,test}/` + `obstacle.yaml`.

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

Dijalankan **sebelum** training penuh di langkah 5. Sama pola dengan langkah 2:
`scripts/tune_yolo_obstacle.py` memakai `optuna.Study` + `TPESampler`, tiap trial
melatih **dari bobot acak** (`yolo26n.yaml`, bukan checkpoint pretrained apa pun)
untuk beberapa epoch singkat (`--epochs_per_trial`) di atas `--data` (label SUN
RGB-D dari langkah 1.3 -- satu-satunya dataset), lalu diberi skor dari validation
mAP50-95 (dimaksimalkan) -- dibaca dengan cara yang sama seperti
`evaluation/eval_detection_metrics.py`.

Hyperparameter yang dicari: optimizer (`lr0`, `lrf`, `momentum`, `weight_decay`,
`warmup_epochs`), bobot loss (`box`, `cls`), dan augmentasi (`hsv_h`, `hsv_s`, `hsv_v`,
`translate`, `scale`, `fliplr`, `mosaic`).

```bash
python -m wheelchair_nav.scripts.tune_yolo_obstacle \
    --data ./data/sunrgbd_yolo/obstacle.yaml \
    --n_trials 30 --epochs_per_trial 10 --imgsz 640 --batch 32 --device 0 \
    --out_json ./log_yolo/optuna_best_yolo_hparams.json
```

Hasil: `./log_yolo/optuna_best_yolo_hparams.json` (mAP50-95 terbaik + hyperparameter
terbaik) dan `..._trials.csv` (riwayat seluruh trial); artefak training tiap trial
ada di `./log_yolo/optuna_tuning/trial_XXX/`. JSON-nya sudah cocok langsung dipakai
flag `--hparams_json` di langkah 5.

---

## 5. Training YOLO26-nano dari scratch (deteksi bbox, class-agnostic)

YOLO26-nano dipakai **hanya** untuk bbox per obstacle; identitas kelas (nama objek)
selalu dibuang di `perception/obstacle_detector.py`, jadi tidak perlu recognition
nama objek. Training **selalu dari bobot acak** (`yolo26n.yaml`) di atas label SUN
RGB-D dari langkah 1.3 -- **tidak ada dataset lain dan tidak ada checkpoint
pretrained (COCO atau lainnya) yang dipakai**, konsisten dengan kebijakan "from
scratch" yang sama dipakai RT-MonoDepth.

Pakai hyperparameter hasil tuning langkah 4 lewat `--hparams_json`:

```bash
python -m wheelchair_nav.scripts.train_yolo_obstacle \
    --data ./data/sunrgbd_yolo/obstacle.yaml \
    --epochs 200 --imgsz 640 --batch 32 --device 0 \
    --hparams_json ./log_yolo/optuna_best_yolo_hparams.json
```

atau tanpa hasil tuning, set manual seperti biasa:

```bash
python -m wheelchair_nav.scripts.train_yolo_obstacle \
    --data ./data/sunrgbd_yolo/obstacle.yaml \
    --epochs 200 --imgsz 640 --batch 32 --device 0
```

Bobot hasil training ada di `./log_yolo/obstacle_yolo26n/weights/best.pt`, siap
dipakai `--yolo_weights` di langkah 6.

---

## 6. Testing sistem navigasi pada file video

Model depth yang dipakai untuk estimasi jarak bisa dipilih lewat `--depth_model`
-- bukan cuma RT-MonoDepth, tapi juga FastDepth dan YOLO26n/s-depth (model
pembanding di langkah 8), karena `DepthEstimator`, `FastDepthEstimator`, dan
`YoloDepthEstimator` sama-sama mengekspos interface `.infer(frame_bgr)` yang
sama -- kode navigasi (Obstacle List, sektor, hysteresis, overlay) tidak perlu
tahu model mana yang dipakai.

```bash
# RT-MonoDepth (full, default)
python -m wheelchair_nav.run_navigation \
    --video ./data/test_indoor.mp4 \
    --depth_model rtmonodepth \
    --depth_weights ./log_sunrgbd/RTMonoDepth_sunrgbd/models/best \
    --yolo_weights ./log_yolo/obstacle_yolo26n/weights/best.pt \
    --output ./out/navigation_demo.mp4 \
    --device cuda --safe_distance 1.0

# FastDepth
python -m wheelchair_nav.run_navigation \
    --video ./data/test_indoor.mp4 \
    --depth_model fastdepth \
    --depth_weights ./log_fastdepth/FastDepth_sunrgbd/models/best \
    --yolo_weights ./log_yolo/obstacle_yolo26n/weights/best.pt \
    --output ./out/navigation_demo_fastdepth.mp4 \
    --device cuda --safe_distance 1.0

# YOLO26n-depth / YOLO26s-depth (--depth_weights langsung ke file .pt, bukan folder)
python -m wheelchair_nav.run_navigation \
    --video ./data/test_indoor.mp4 \
    --depth_model yolo26n-depth \
    --depth_weights ./log_yolo_depth/yolo26n_depth_sunrgbd/weights/best.pt \
    --yolo_weights ./log_yolo/obstacle_yolo26n/weights/best.pt \
    --output ./out/navigation_demo_yolo26n_depth.mp4 \
    --device cuda --safe_distance 1.0
```

Arti `--depth_weights` tergantung `--depth_model`:
- `rtmonodepth` / `fastdepth` -> folder hasil `train_depth_sunrgbd.py` /
  `train_fastdepth_sunrgbd.py` (berisi `encoder.pth`+`depth.pth`, atau
  `fastdepth.pth`).
- `yolo26n-depth` / `yolo26s-depth` -> satu file checkpoint `.pt` hasil
  `train_yolo_depth.py` (mis. `runs/.../weights/best.pt`).

Output:
- `./out/navigation_demo.mp4` -- video **side-by-side** (RGB+overlay kiri, peta
  depth berwarna kanan, ukuran sama, lebar video jadi 2x lebar input):
  - Kotak bbox per obstacle + label jarak (`obstacle X.XXm`, merah jika
    < `safe_distance`, hijau jika bebas).
  - 5 sektor (`FL | L | CTR | R | FR`) diberi tint warna translucent sesuai
    status: **hijau** = sektor yang dipilih (jalur yang dilewati), **kuning** =
    bebas tapi tidak dipilih, **merah** = terhalang -- plus label sektor +
    jarak (meter) di tengah tiap sektor.
  - Banner bawah: decision (`FORWARD`/`TURN_LEFT`/`TURN_RIGHT`/
    `TURN_FAR_LEFT`/`TURN_FAR_RIGHT`/`STOP`, warna sesuai jenis keputusan),
    `OBS: X.XXm` (jarak obstacle terdekat di semua sektor), dan FPS.
- `./out/navigation_demo_log.csv` -- log per-frame (`frame, decision,
  num_obstacles, min_depth_m, fps, FL0, L1, CTR2, R3, FR4`), dipakai evaluasi
  end-to-end di langkah 7.

`--yolo_weights` wajib diisi dengan checkpoint hasil langkah 5 (`.../weights/best.pt`)
-- tidak ada fallback ke bobot COCO-pretrained di mana pun dalam sistem ini.

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

## 8. Model depth pembanding (apples-to-apples): FastDepth, YOLO26n-depth, YOLO26s-depth

Untuk membandingkan RT-MonoDepth secara adil, tiga model depth lain dilatih **dari
data yang persis sama** (`splits_sunrgbd/{train,val,test}.txt` dari langkah 1) dan
dievaluasi dengan **formula metrik yang persis sama** (`abs_rel, sq_rel, rmse,
rmse_log, a1, a2, a3`, identik dengan `evaluate_depth_full.py` di root repo):

- **FastDepth** -- `networks/FastDepth/model.py` (`MobileNetSkipAdd`), **sudah ada di
  repo ini tanpa diubah** (dipakai `compare_runtime.py` untuk benchmark runtime).
  Output lapisan terakhirnya sudah ReLU (selalu >= 0); di sini hanya di-*clamp* ke
  rentang metric yang sama dengan RT-MonoDepth, tanpa menambah lapisan baru.
- **YOLO26n-depth** dan **YOLO26s-depth** -- arsitektur *native* Ultralytics untuk
  monocular depth estimation (`ultralytics/cfg/models/26/yolo26-depth.yaml`, skala
  `n`/`s`), **tidak diubah**. Training, loss (SILog + gradient), dan kalibrasi skala
  metrik pasca-training sepenuhnya ditangani oleh trainer/validator bawaan
  Ultralytics (`ultralytics.models.yolo.depth`) -- kode di sini hanya membungkusnya.

Keempat model (RT-MonoDepth + 3 pembanding) tidak menyentuh
`wheelchair_nav/perception/` maupun `run_navigation.py` -- RT-MonoDepth + YOLO26-nano
detektor tetap satu-satunya stack yang dipakai sistem navigasi. `baselines/` dan
skrip `*_fastdepth_*`/`*_yolo_depth_*` di atas murni untuk studi perbandingan.

### 8a. Siapkan layout dataset yang identik untuk YOLO26{n,s}-depth

`scripts/prepare_sunrgbd.py` (langkah 1) bisa langsung menge-*mirror* (symlink,
tanpa duplikasi data) split train/val/test yang sama persis ke layout
`images/{split}/*.jpg` + `depth/{split}/*.npy` yang dipakai native depth task
Ultralytics:

```bash
python -m wheelchair_nav.scripts.prepare_sunrgbd \
    --sunrgbd_root /path/to/SUNRGBD \
    --out_dir ./data/sunrgbd_processed \
    --splits_dir ./splits_sunrgbd \
    --make_yolo_depth_layout --yolo_depth_out_dir ./data/sunrgbd_yolo_depth
```

Menghasilkan `./data/sunrgbd_yolo_depth/depth_comparison.yaml` (siap dipakai
`--data` di langkah 8c/8d) -- gambar yang di dalamnya **sama persis** dengan yang
dipakai RT-MonoDepth/FastDepth via `splits_sunrgbd/*.txt`.

### 8b. FastDepth: tuning lalu training (sama pola dengan RT-MonoDepth)

```bash
# 1) Hyperparameter tuning (Optuna, TPE) -- ruang pencarian & protokol identik
#    dengan tune_depth_sunrgbd.py (langkah 2), supaya kedua model di-tuning
#    dengan cara yang sama:
python -m wheelchair_nav.scripts.tune_fastdepth_sunrgbd \
    --splits_dir ./splits_sunrgbd \
    --height 192 --width 640 \
    --n_trials 30 --epochs_per_trial 5 \
    --out_json ./log_fastdepth/optuna_best_fastdepth_hparams.json \
    --device cuda

# 2) Training penuh, from scratch, pakai hasil tuning
python -m wheelchair_nav.scripts.train_fastdepth_sunrgbd \
    --splits_dir ./splits_sunrgbd \
    --log_dir ./log_fastdepth \
    --model_name FastDepth_sunrgbd \
    --height 192 --width 640 \
    --min_depth 0.1 --max_depth 10.0 \
    --num_epochs 40 \
    --hparams_json ./log_fastdepth/optuna_best_fastdepth_hparams.json
```

Checkpoint tersimpan di
`./log_fastdepth/FastDepth_sunrgbd/models/{weights_N,best}/fastdepth.pth`.

### 8c. YOLO26n-depth / YOLO26s-depth: tuning lalu training

```bash
# 1) Hyperparameter tuning (Optuna, TPE) -- optimizer + bobot loss depth
#    (dlog: SILog gain, dgrad: gradient-loss gain, dlam: SILog scale-invariance
#    focus) + augmentasi; skor = validation abs_rel (diminimalkan)
python -m wheelchair_nav.scripts.tune_yolo_depth \
    --variant n \
    --data ./data/sunrgbd_yolo_depth/depth_comparison.yaml \
    --n_trials 30 --epochs_per_trial 10 --imgsz 640 --batch 16 --device 0 \
    --out_json ./log_yolo_depth/optuna_best_yolo26n_depth_hparams.json

# 2) Training penuh, pakai hasil tuning
python -m wheelchair_nav.scripts.train_yolo_depth \
    --variant n \
    --data ./data/sunrgbd_yolo_depth/depth_comparison.yaml \
    --epochs 60 --imgsz 640 --batch 16 --device 0 \
    --hparams_json ./log_yolo_depth/optuna_best_yolo26n_depth_hparams.json
```

Ganti `--variant n` menjadi `--variant s` untuk YOLO26s-depth (tuning dan training
terpisah, sama perintah). Checkpoint tersimpan di
`./log_yolo_depth/yolo26{n,s}_depth_sunrgbd/weights/best.pt` -- sudah otomatis
dikalibrasi skala metriknya oleh Ultralytics di akhir training (log
`"Auto-calibration written to best.pt"`).

Default `--pretrained` kosong (`""`) -> training dari bobot acak
(`yolo26{n,s}-depth.yaml`), konsisten dengan kebijakan "from scratch" di semua
model lain; isi `--pretrained` dengan path checkpoint kalau memang butuh
memulai dari bobot tertentu.

### 8d. Jalankan perbandingan apples-to-apples

```bash
python -m wheelchair_nav.evaluation.eval_depth_comparison \
    --test_list ./splits_sunrgbd/test.txt \
    --rtmonodepth_weights_dir ./log_sunrgbd/RTMonoDepth_sunrgbd/models/best \
    --fastdepth_weights_dir ./log_fastdepth/FastDepth_sunrgbd/models/best \
    --yolo26n_depth_weights ./log_yolo_depth/yolo26n_depth_sunrgbd/weights/best.pt \
    --yolo26s_depth_weights ./log_yolo_depth/yolo26s_depth_sunrgbd/weights/best.pt \
    --device cuda \
    --out_csv ./log_sunrgbd/depth_comparison.csv
```

Boleh isi hanya sebagian flag `--*_weights*` -- model yang tidak diberi bobotnya
otomatis dilewati. Mencetak satu tabel berisi `abs_rel, sq_rel, rmse, rmse_log, a1,
a2, a3, Params(M), FPS` untuk tiap model yang diberikan, dihitung di atas **gambar
test yang sama** dan **rumus metrik yang sama** -- juga disimpan ke
`--out_csv` bila diisi.

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
| `SECTOR_TO_DECISION` | CTR2->FORWARD, L1->TURN_LEFT, R3->TURN_RIGHT, FL0->TURN_FAR_LEFT, FR4->TURN_FAR_RIGHT | pemetaan sektor->decision (bijektif) |
| `HYSTERESIS_WINDOW` | 3 | N frame majority-vote |

## Catatan / batasan

- RT-MonoDepth (`networks/RTMonoDepth/RTMonoDepth.py`) tidak diubah sama sekali;
  yang baru hanyalah training objective (supervised, bukan self-supervised) dan
  wrapper inferensi.
- Tidak ada hardware kursi roda fisik dalam repo ini -- `navigation/controller.py`
  hanya menghasilkan simulated drive command (`linear_mps`, `angular_radps`) untuk
  logging/visualisasi; ganti isinya dengan driver motor sungguhan bila diintegrasikan
  ke perangkat keras.
- Parsing anotasi 2D SUN RGB-D (`annotation2Dfinal/index.json`, langkah 1.3) adalah
  **satu-satunya** sumber dataset untuk training YOLO26-nano -- tidak ada dataset
  lain atau checkpoint pretrained (COCO atau lainnya) yang dipakai di mana pun
  dalam sistem ini (training maupun `run_navigation.py`). Scene yang tidak punya
  `annotation2Dfinal/` atau JSON-nya tidak valid dilewati dan dihitung, bukan
  diganti sumber lain -- kalau jumlah scene yang berhasil dikonversi terlalu sedikit,
  periksa `--sunrgbd_root` dan `--exclude_classes`, bukan beralih dataset.
- Hyperparameter tuning (langkah 2, 4, dan 8b/8c) memakai Optuna dengan
  `TPESampler` (Bayesian, Tree-structured Parzen Estimator) dan budget epoch yang
  sengaja lebih kecil dari training penuh -- ini proxy search, bukan pengganti
  training penuh; hasil terbaiknya tetap perlu dilatih ulang dengan
  `--num_epochs`/`--epochs` penuh di langkah 3/5/8b/8c.
- FastDepth, YOLO26n-depth, dan YOLO26s-depth (langkah 8) murni model pembanding
  untuk studi evaluasi; sistem navigasi (`run_navigation.py`) tetap hanya memakai
  RT-MonoDepth + YOLO26-nano detektor seperti dijelaskan di langkah 1-7.
