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
  config.py                        thresholds, sektor, resolusi input (288x384) semua model depth
  datasets/sunrgbd_dataset.py       PyTorch Dataset RGB + metric-depth SUN RGB-D
  training_log.py                  CSV + kurva train/val loss (dipakai trainer depth)
  scripts/
    prepare_sunrgbd.py               download-agnostic preprocessing SUN RGB-D
    tune_depth_sunrgbd.py             Optuna-TPE hyperparameter search for RT-MonoDepth
    train_depth_sunrgbd.py           training RT-MonoDepth (full) from scratch
    tune_yolo_obstacle.py             Optuna-TPE hyperparameter search for YOLO26-nano
    train_yolo_obstacle.py           fine-tuning YOLO26-nano (class-agnostic)
    tune_fastdepth_sunrgbd.py         Optuna-TPE hyperparameter search for FastDepth (baseline)
    train_fastdepth_sunrgbd.py       training FastDepth (baseline) from scratch
    tune_ghostdepth_sunrgbd.py        Optuna-TPE hyperparameter search for Ghost-Depth (baseline)
    train_ghostdepth_sunrgbd.py      training Ghost-Depth (baseline) from scratch, berHu loss
    tune_yolo_depth.py                Optuna-TPE hyperparameter search for YOLO26{n,s}-depth (baseline)
    train_yolo_depth.py              training YOLO26n-depth / YOLO26s-depth (baseline)
  perception/
    depth_estimator.py                wrapper inferensi RT-MonoDepth (full, unmodified)
    obstacle_detector.py              wrapper YOLO26-nano (bbox saja, tanpa recognition)
    obstacle_list.py                  fusi bbox + depth map -> Obstacle List
  baselines/                        model depth pembanding (apples-to-apples), lihat langkah 8
    fastdepth_estimator.py            wrapper inferensi FastDepth (unmodified)
    ghostdepth_estimator.py           wrapper inferensi Ghost-Depth (reimplementasi CNIOT '23)
    yolo_depth_estimator.py           wrapper inferensi YOLO26{n,s}-depth (unmodified)
  navigation/
    sectors.py                        partisi FOV jadi 5 sektor FL0..FR4
    free_path.py                      priority selection + hysteresis N=3
    controller.py                     decision -> simulated drive command
  run_navigation.py                  entry point: uji sistem pada file video
  evaluation/
    latency.py                        pengukuran latency bersama: mean/std/min/
                                       p50/p90/p95/p99/max + reaction distance
    eval_depth_metrics.py             abs_rel/sq_rel/rmse/rmse_log/a1-a3 + params + FPS
    eval_detection_metrics.py         mAP50/mAP50-95/precision/recall + params
                                       + distribusi latency
    eval_navigation_metrics.py        latency end-to-end + breakdown per tahap,
                                       reaction distance, missed/false-stop rate,
                                       decision accuracy, distance MAE/RMSE
    eval_depth_comparison.py          RT-MonoDepth vs FastDepth vs Ghost-Depth vs
                                       YOLO26{n,s}-depth, metrik+params+GMACs+
                                       latency identik, test split identik
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
       --out_dir ./wheelchair_nav/data/sunrgbd_processed \
       --splits_dir ./wheelchair_nav/splits_sunrgbd
   ```

   Script ini men-decode depth PNG 16-bit Kinect (bit-rotate 3-bit, standar
   SUNRGBDtoolbox) jadi metric depth dalam meter (`.npy`), lalu membagi otomatis jadi
   `train.txt` / `val.txt` / `test.txt` (80/10/10, seed=42) berisi baris
   `<path_rgb> <path_depth_npy>`. Jika hasil decode kebanyakan nol/aneh untuk sensor
   tertentu (variasi rilis dataset), coba `--depth_encoding raw_mm`.

3. **Wajib** untuk training YOLO26-nano di langkah 5: konversi anotasi 2D bawaan
   SUN RGB-D sendiri (`annotation2Dfinal/index.json`) jadi label YOLO **satu kelas**
   (`obstacle`), class-agnostic. **Ini satu-satunya dataset berlabel yang dipakai
   untuk training deteksi obstacle**; bobot awalnya di-fine-tune dari checkpoint
   COCO-pretrained (lihat langkah 5 untuk alasan metodologisnya):

   ```bash
   python -m wheelchair_nav.scripts.prepare_sunrgbd \
       --sunrgbd_root /path/to/SUNRGBD \
       --out_dir ./wheelchair_nav/data/sunrgbd_processed \
       --splits_dir ./wheelchair_nav/splits_sunrgbd \
       --make_yolo_labels --yolo_out_dir ./wheelchair_nav/data/sunrgbd_yolo
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
   ke `./wheelchair_nav/data/sunrgbd_yolo/{images,labels}/{train,val,test}/` + `obstacle.yaml`.

---

## 1b. Resolusi input: 288x384 untuk SEMUA model depth

Kelima model depth dilatih dan dievaluasi pada **konten gambar 288x384** yang sama,
diatur satu tempat di `wheelchair_nav/config.py` (`INPUT_HEIGHT`, `INPUT_WIDTH`,
`YOLO_DEPTH_IMGSZ`). Ini prasyarat supaya perbandingan di langkah 8e benar-benar
mengukur arsitektur, bukan seberapa banyak gambar yang dilihat tiap model.

**Kenapa 288x384, bukan 192x640 (default asli repo):** SUN RGB-D itu 640x480 (4:3).

| Resolusi | Aspect | Skala V / H | Distorsi | /32 | Piksel |
|---|---|---|---|---|---|
| 192x640 (default lama) | 3.333 | 0.40 / 1.00 | **2.50x gepeng** | ya | 122,880 |
| **288x384 (dipakai)** | **1.333** | **0.60 / 0.60** | **tidak ada** | ya | 110,592 |

192x640 itu warisan konfigurasi KITTI (~10:3, wajar di sana). Dipakai untuk gambar
indoor 4:3 ia **menggepengkan gambar 2.5x secara vertikal** -- menghandicap semua
model depth. 288x384 memberi skala **seragam 0.6x di kedua sumbu** (tanpa distorsi),
tetap habis dibagi 32 (syarat encoder 5-stage), dan justru **~10% lebih murah** dari
192x640. Paper Ghost-Depth sendiri independen memakai 228x304, juga tepat 4:3.

**Untuk YOLO26{n,s}-depth**: Ultralytics menerima satu `imgsz` persegi dan
*letterbox* ke dalamnya dengan mempertahankan aspect ratio. Pada `--imgsz 384`,
frame 640x480 mendarat di **tepat 288x384** konten asli (skala 0.6, diverifikasi
terhadap `ultralytics.data.augment.LetterBox`), sisanya padding abu-abu. Jadi
konten gambar yang dilihat YOLO **identik** dengan model lain; padding itu hanya
menambah biaya komputasi (kanvas 384x384 = 1.33x konten efektifnya), bukan
informasi gambar tambahan.

Catatan: `--imgsz` untuk **detektor obstacle** (`*_yolo_obstacle`) tetap 640 dan
sengaja tidak diikutkan -- itu model deteksi, bukan salah satu dari 5 model depth,
dan deteksi objek kecil memang diuntungkan resolusi lebih tinggi.

Semua flag `--height/--width/--imgsz` masih bisa ditimpa per perintah; nilai di
`config.py` hanya default bersamanya.

---

## 1c. Budget training: 100 epoch untuk SEMUA model depth

Sama alasannya dengan resolusi: membandingkan model yang dilatih dengan jumlah
epoch berbeda mencampuradukkan **arsitektur** dengan **budget training**. Resep
asal tiap model berbeda-beda (RT-MonoDepth/FastDepth 40, Ghost-Depth 55 menurut
paper sec. 4.2, YOLO26-depth 60), jadi semuanya sekarang memakai satu angka:
`TRAIN_EPOCHS = 100` di `config.py`. Angka ini cukup longgar -- pada run 100
epoch, kurva validasi kelima model sudah datar di sepertiga terakhir.

**LR schedule ikut menskala, dan ini penting.** Menaikkan budget tanpa menaikkan
periode decay justru membuang sisa epoch. Dengan `StepLR(gamma=0.1)`:

| step | Trayektori LR di 100 epoch | Epoch terbuang di lr<1e-6 |
|---|---|---|
| 25 (lama) | `1e-4 / 1e-5 / 1e-6 / 1e-7` | **25** |
| 30 (lama, Ghost-Depth) | `1e-4 / 1e-5 / 1e-6 / 1e-7` | **10** |
| **33 = 100//3 (sekarang)** | `1e-4 / 1e-5 / 1e-6 / 1e-7` | **1** |

Karena itu `--scheduler_step_size` sekarang default `None` -> dihitung otomatis
sebagai `num_epochs // 3`. Kalau Anda mengubah `--num_epochs`, step-nya ikut
menyesuaikan; kalau Anda mengisi `--scheduler_step_size` eksplisit (mis. `30`
untuk mereplikasi resep paper Ghost-Depth persis), nilai Anda tetap dipakai.

Semua ini hanya default. `--num_epochs` / `--epochs` tetap bisa ditimpa per
perintah.

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
    --splits_dir ./wheelchair_nav/splits_sunrgbd \
    --height 288 --width 384 \
    --n_trials 30 --epochs_per_trial 5 \
    --out_json ./wheelchair_nav/log_sunrgbd/optuna_best_depth_hparams.json \
    --device cuda
```

- `--n_trials` -- jumlah kombinasi hyperparameter yang dicoba TPE (disarankan 20-50).
- `--epochs_per_trial` -- budget proxy per trial, jauh lebih kecil dari
  `--num_epochs` training penuh supaya pencarian tetap cepat.
- `--storage sqlite:///./wheelchair_nav/log_sunrgbd/optuna_depth.db` (opsional) -- simpan progres
  studi supaya bisa dilanjutkan/dijalankan paralel di beberapa proses.

Hasil: `./wheelchair_nav/log_sunrgbd/optuna_best_depth_hparams.json` (nilai val-L1 terbaik +
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

Selain objektifnya (supervised vs self-supervised), resep training di sini mengikuti
`trainer.py` semirip mungkin, supaya tidak ada bagian arsitektur yang mubazir/kurang
optimal:
- **Multi-scale**: `DepthDecoder` menghasilkan 4 skala output (`disp 0-3`); loss
  dihitung di keempatnya (GT depth/mask/color di-*downsample* ke resolusi tiap skala),
  bukan cuma skala 0 -- kalau tidak, 3 dari 4 *head* dispconv tidak pernah menerima
  gradien sama sekali.
- **Smoothness dengan normalisasi mean-disparity**: `disp / mean(disp)` sebelum
  dihitung `get_smooth_loss`, sama seperti `trainer.py` -- tanpa ini, loss smoothness
  bisa "dicurangi" cukup dengan memperkecil disparitas secara keseluruhan, bukan
  benar-benar menghaluskannya.
- **Optimizer AdamW** (bukan Adam) -- match `trainer.py`, penting terutama saat
  `--weight_decay` > 0 dari hasil tuning.
- **Augmentasi warna** (`ColorJitter`, brightness/contrast/saturation/hue) pada data
  training, selain flip horizontal -- meniru `color_aug` di `trainer.py`, membantu
  ketahanan terhadap variasi pencahayaan kamera di dunia nyata.

Pakai hyperparameter hasil tuning langkah 2 langsung lewat `--hparams_json` (meng-*override*
`--learning_rate`/`--batch_size`/`--smoothness_weight`/`--si_lambda`/`--weight_decay`):

```bash
python -m wheelchair_nav.scripts.train_depth_sunrgbd \
    --splits_dir ./wheelchair_nav/splits_sunrgbd \
    --log_dir ./wheelchair_nav/log_sunrgbd \
    --model_name RTMonoDepth_sunrgbd \
    --height 288 --width 384 \
    --min_depth 0.1 --max_depth 10.0 \
    --hparams_json ./wheelchair_nav/log_sunrgbd/optuna_best_depth_hparams.json
```

atau tanpa hasil tuning, set manual seperti biasa:

```bash
python -m wheelchair_nav.scripts.train_depth_sunrgbd \
    --splits_dir ./wheelchair_nav/splits_sunrgbd \
    --log_dir ./wheelchair_nav/log_sunrgbd \
    --model_name RTMonoDepth_sunrgbd \
    --height 288 --width 384 \
    --min_depth 0.1 --max_depth 10.0 \
    --batch_size 16 --learning_rate 1e-4
```

Checkpoint tiap epoch + checkpoint terbaik (val L1 terendah) disimpan di
`./wheelchair_nav/log_sunrgbd/RTMonoDepth_sunrgbd/models/{weights_N,best}/{encoder.pth,depth.pth}`
-- format sama seperti `test_simple_full.py` di root (`encoder.pth` menyimpan juga
`height`/`width`), sehingga bisa langsung dipakai `wheelchair_nav.perception.DepthEstimator`.

Kurva train-loss vs val-loss per epoch (untuk mengecek overfitting/underfitting)
otomatis di-update tiap akhir epoch ke
`./wheelchair_nav/log_sunrgbd/RTMonoDepth_sunrgbd/training_curve.png` (+ data
mentahnya di `training_log.csv` sebelahnya) -- bisa dipantau langsung meski
training masih berjalan.

---

## 4. Hyperparameter tuning YOLO26-nano (Optuna, TPE sampler)

Dijalankan **sebelum** training penuh di langkah 5. Sama pola dengan langkah 2:
`scripts/tune_yolo_obstacle.py` memakai `optuna.Study` + `TPESampler`, tiap trial
me-*fine-tune* salinan baru dari `--pretrained` (default `yolo26n.pt`) untuk
beberapa epoch singkat (`--epochs_per_trial`) di atas `--data` (label SUN RGB-D
dari langkah 1.3), lalu diberi skor dari validation mAP50-95 (dimaksimalkan) --
dibaca dengan cara yang sama seperti `evaluation/eval_detection_metrics.py`.

> `--pretrained` di sini **harus sama** dengan yang dipakai di langkah 5. Kalau
> berbeda, hyperparameter hasil tuning (terutama `lr0` dan `warmup_epochs`)
> optimal untuk titik awal yang berbeda dari yang benar-benar dipakai training.

Hyperparameter yang dicari: optimizer (`lr0`, `lrf`, `momentum`, `weight_decay`,
`warmup_epochs`), bobot loss (`box`, `cls`), dan augmentasi (`hsv_h`, `hsv_s`, `hsv_v`,
`translate`, `scale`, `fliplr`, `mosaic`).

```bash
python -m wheelchair_nav.scripts.tune_yolo_obstacle \
    --data ./wheelchair_nav/data/sunrgbd_yolo/obstacle.yaml \
    --n_trials 30 --epochs_per_trial 10 --imgsz 640 --batch 32 --device 0 \
    --out_json ./wheelchair_nav/log_yolo/optuna_best_yolo_hparams.json
```

Hasil: `./wheelchair_nav/log_yolo/optuna_best_yolo_hparams.json` (mAP50-95 terbaik + hyperparameter
terbaik) dan `..._trials.csv` (riwayat seluruh trial); artefak training tiap trial
ada di `./wheelchair_nav/log_yolo/optuna_tuning/trial_XXX/`. JSON-nya sudah cocok langsung dipakai
flag `--hparams_json` di langkah 5.

---

## 5. Training YOLO26-nano (deteksi bbox, class-agnostic, fine-tune dari COCO)

YOLO26-nano dipakai **hanya** untuk bbox per obstacle; identitas kelas (nama objek)
selalu dibuang di `perception/obstacle_detector.py`, jadi tidak perlu recognition
nama objek. Training adalah **fine-tuning dari checkpoint COCO-pretrained**
(`yolo26n.pt`, default `--pretrained`) di atas label SUN RGB-D dari langkah 1.3.

### Kenapa detektor memakai pretrained sementara kelima model depth tidak

Ini asimetri yang **disengaja**, dan alasannya bukan kenyamanan:

- **Model depth adalah objek pembandingan.** Pertanyaan penelitiannya adalah model
  depth mana yang paling seimbang, jadi kelimanya wajib mendapat perlakuan identik.
  Checkpoint pretrained **tidak tersedia setara** untuk kelimanya (Ghost-Depth tidak
  punya sama sekali, RT-MonoDepth tidak punya versi indoor), sehingga inisialisasi
  acak adalah satu-satunya setup yang membuat perbandingannya sah.
- **Detektor bukan objek pembandingan** -- dia komponen tetap pada tahap persepsi,
  tidak dibandingkan dengan apa pun, jadi kendala keadilan itu tidak berlaku.
- **Detektor lemah justru merusak perbandingan depth.** Depth hanya dibaca **di
  dalam bbox** (`obstacle_list.py` mengambil median pada 60% pusat bbox), jadi objek
  yang tidak terdeteksi tidak terlihat oleh model depth mana pun. Recall detektor
  yang rendah akan meratakan perbedaan antar model depth yang justru ingin diukur
  `eval_navigation_metrics.py` di langkah 7.
- Kelas COCO tumpang tindih kuat dengan isi SUN RGB-D (`chair`, `couch`, `dining
  table`, `tv`, `bed`, `toilet`, `sink`, `refrigerator`, `person`, `potted plant`),
  jadi transfer-nya sangat efektif untuk domain indoor ini.

Nyatakan asimetri ini secara eksplisit saat melaporkan hasil: **model depth dilatih
hanya di SUN RGB-D, detektor di-fine-tune dari COCO.**

Pakai hyperparameter hasil tuning langkah 4 lewat `--hparams_json`:

```bash
python -m wheelchair_nav.scripts.train_yolo_obstacle \
    --data ./wheelchair_nav/data/sunrgbd_yolo/obstacle.yaml \
    --epochs 100 --imgsz 640 --batch 32 --device 0 \
    --hparams_json ./wheelchair_nav/log_yolo/optuna_best_yolo_hparams.json
```

atau tanpa hasil tuning, set manual seperti biasa:

```bash
python -m wheelchair_nav.scripts.train_yolo_obstacle \
    --data ./wheelchair_nav/data/sunrgbd_yolo/obstacle.yaml \
    --epochs 100 --imgsz 640 --batch 32 --device 0
```

Untuk ablation *from-scratch vs transfer learning*, jalankan sekali lagi dengan
`--pretrained ""` (naikkan `--epochs` ke 300-500, karena bobot acak konvergen jauh
lebih lambat) dan laporkan keduanya:

```bash
python -m wheelchair_nav.scripts.train_yolo_obstacle \
    --data ./wheelchair_nav/data/sunrgbd_yolo/obstacle.yaml \
    --pretrained "" --epochs 400 --imgsz 640 --batch 32 --device 0 \
    --name obstacle_yolo26n_scratch
```

Bobot hasil training ada di `./wheelchair_nav/log_yolo/obstacle_yolo26n/weights/best.pt`, siap
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
    --video ./wheelchair_nav/data/test_indoor.mp4 \
    --depth_model rtmonodepth \
    --depth_weights ./wheelchair_nav/log_sunrgbd/RTMonoDepth_sunrgbd/models/best \
    --yolo_weights ./wheelchair_nav/log_yolo/obstacle_yolo26n/weights/best.pt \
    --output ./wheelchair_nav/out/navigation_demo.mp4 \
    --device cuda --safe_distance 1.0

# FastDepth
python -m wheelchair_nav.run_navigation \
    --video ./wheelchair_nav/data/test_indoor.mp4 \
    --depth_model fastdepth \
    --depth_weights ./wheelchair_nav/log_fastdepth/FastDepth_sunrgbd/models/best \
    --yolo_weights ./wheelchair_nav/log_yolo/obstacle_yolo26n/weights/best.pt \
    --output ./wheelchair_nav/out/navigation_demo_fastdepth.mp4 \
    --device cuda --safe_distance 1.0

# YOLO26n-depth / YOLO26s-depth (--depth_weights langsung ke file .pt, bukan folder)
python -m wheelchair_nav.run_navigation \
    --video ./wheelchair_nav/data/test_indoor.mp4 \
    --depth_model yolo26n-depth \
    --depth_weights ./wheelchair_nav/log_yolo_depth/yolo26n_depth_sunrgbd/weights/best.pt \
    --yolo_weights ./wheelchair_nav/log_yolo/obstacle_yolo26n/weights/best.pt \
    --output ./wheelchair_nav/out/navigation_demo_yolo26n_depth.mp4 \
    --device cuda --safe_distance 1.0
```

Arti `--depth_weights` tergantung `--depth_model`:
- `rtmonodepth` / `fastdepth` -> folder hasil `train_depth_sunrgbd.py` /
  `train_fastdepth_sunrgbd.py` (berisi `encoder.pth`+`depth.pth`, atau
  `fastdepth.pth`).
- `yolo26n-depth` / `yolo26s-depth` -> satu file checkpoint `.pt` hasil
  `train_yolo_depth.py` (mis. `./wheelchair_nav/log_yolo_depth/.../weights/best.pt`).

Output:
- `./wheelchair_nav/out/navigation_demo.mp4` -- video **side-by-side** (RGB+overlay kiri, peta
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
- `./wheelchair_nav/out/navigation_demo_log.csv` -- log per-frame (`frame, decision,
  num_obstacles, min_depth_m, fps, FL0, L1, CTR2, R3, FR4`), dipakai evaluasi
  end-to-end di langkah 7.

`--yolo_weights` wajib diisi dengan checkpoint hasil langkah 5 (`.../weights/best.pt`)
-- tidak ada nilai default apa pun, supaya sistem navigasi tidak bisa tanpa sengaja
berjalan memakai `yolo26n.pt` COCO mentah yang belum pernah dilatih pada label
SUN RGB-D Anda.

---

## 7. Matriks evaluasi

### 7a. Depth: akurasi, error, jumlah parameter, FPS

```bash
python -m wheelchair_nav.evaluation.eval_depth_metrics \
    --weights_dir ./wheelchair_nav/log_sunrgbd/RTMonoDepth_sunrgbd/models/best \
    --test_list ./wheelchair_nav/splits_sunrgbd/test.txt \
    --device cuda
```

Mencetak `abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3` (definisi sama dengan
`evaluate_depth_full.py` di root repo), jumlah parameter encoder+decoder, dan FPS
inferensi murni (warmup + averaged cycles, mengikuti pola `compare_runtime.py`).

### 7b. Deteksi obstacle: mAP, precision, recall, parameter, FPS

```bash
python -m wheelchair_nav.evaluation.eval_detection_metrics \
    --weights ./wheelchair_nav/log_yolo/obstacle_yolo26n/weights/best.pt \
    --data ./wheelchair_nav/data/sunrgbd_yolo/obstacle.yaml \
    --device 0
```

### 7c. Sistem navigasi end-to-end: latency, keamanan, akurasi keputusan, error jarak

```bash
python -m wheelchair_nav.evaluation.eval_navigation_metrics \
    --nav_log ./wheelchair_nav/out/navigation_demo_log.csv \
    --safe_distance 1.0 \
    --gt_decisions ./wheelchair_nav/data/gt_decisions.csv \
    --gt_distances ./wheelchair_nav/data/gt_distances.json \
    --video ./wheelchair_nav/data/test_indoor.mp4 \
    --depth_weights ./wheelchair_nav/log_sunrgbd/RTMonoDepth_sunrgbd/models/best
```

Selalu dihitung dari log: FPS pipeline (mean/min/max), **distribusi latency
end-to-end**, **missed-stop rate** (obstacle < threshold tapi sistem tetap FORWARD --
risiko tabrakan) dan **false-stop rate** (sistem STOP padahal tidak ada obstacle <
threshold -- terlalu konservatif).

#### Kenapa latency, bukan cuma FPS

FPS adalah **throughput** -- berapa frame per detik yang sanggup diproses. Latency
adalah **delay per frame** antara foton masuk kamera dan perintah gerak berubah.
Untuk kursi roda otonom, FPS rata-rata menyembunyikan ekornya: model dengan rata-rata
8 ms yang sesekali melonjak ke 90 ms punya FPS yang sama dengan model yang stabil di
9 ms, padahal hanya yang kedua aman.

`eval_navigation_metrics.py` melaporkan tiga hal dari kolom `total_ms` / `depth_ms` /
`detect_ms` / `nav_ms` yang ditulis `run_navigation.py`:

1. **Distribusi latency end-to-end** -- mean, std, min, p50, p90, p95, p99, max (ms).
2. **Breakdown per tahap** -- berapa persen anggaran frame dihabiskan depth, deteksi,
   dan navigasi. Tahap navigasi murni CPU dan seharusnya jauh di bawah 1 ms; kalau
   tidak, ada yang salah.
3. **Reaction distance** -- jarak yang masih ditempuh kursi roda sambil "buta" selama
   frame sedang diproses, pada kecepatan jelajah controller (0.6 m/s):

   $$d_{\text{reaksi}} = v_{\text{cruise}} \times t_{\text{latency}}$$

   Bandingkan angka **p99**-nya dengan `EMERGENCY_DISTANCE_M = 0.5 m`. Kalau reaction
   distance p99 memakan bagian berarti dari margin 0.5 m itu, itu temuan yang harus
   dilaporkan, bukan catatan kaki.

Latency diukur dengan `time.perf_counter()` (wall clock), **bukan** CUDA events --
karena CUDA events hanya menghitung kernel GPU dan diam-diam membuang biaya resize,
transfer H2D dan D2H yang benar-benar dibayar sistem. Setiap sampel dibatasi
`torch.cuda.synchronize()`, sebab tanpa itu timer hanya mencatat waktu submit antrian,
bukan waktu selesai.

> Log CSV lama yang belum punya kolom latency tetap bisa dibaca -- bagian FPS dan
> safety tetap keluar, bagian latency dilewati dengan pesan.

Opsional dengan ground truth:
- `--gt_decisions` (CSV `frame,decision`) -> akurasi keputusan + confusion matrix.
- `--gt_distances` (JSON list `{"frame", "bbox":[x1,y1,x2,y2], "distance_m"}`, jarak
  hasil ukur manual/tape-measure pada beberapa frame referensi) + `--video` +
  `--depth_weights` -> MAE/RMSE estimasi jarak RT-MonoDepth+YOLO terhadap jarak
  sebenarnya.

---

## 8. Model depth pembanding (apples-to-apples): FastDepth, Ghost-Depth, YOLO26n-depth, YOLO26s-depth

Untuk membandingkan RT-MonoDepth secara adil, empat model depth lain dilatih **dari
data yang persis sama** (`splits_sunrgbd/{train,val,test}.txt` dari langkah 1) dan
dievaluasi dengan **formula metrik yang persis sama** (`abs_rel, sq_rel, rmse,
rmse_log, a1, a2, a3`, identik dengan `evaluate_depth_full.py` di root repo):

- **FastDepth** -- `networks/FastDepth/model.py` (`MobileNetSkipAdd`), **sudah ada di
  repo ini tanpa diubah** (dipakai `compare_runtime.py` untuk benchmark runtime).
  Output lapisan terakhirnya sudah ReLU (selalu >= 0); di sini hanya di-*clamp* ke
  rentang metric yang sama dengan RT-MonoDepth, tanpa menambah lapisan baru.
- **Ghost-Depth** -- `networks/GhostDepth/ghost_depth.py`, **reimplementasi** dari
  Quan et al., *"Ghost-Depth: A Lightweight Encoder-Decoder Network for Monocular
  Depth Estimation"*, CNIOT '23 ([doi:10.1145/3603781.3603861](https://doi.org/10.1145/3603781.3603861)).
  Encoder GhostNet + decoder Ghost convolution + skip connection iAFF di jalur
  40-channel, dilatih dengan **berHu loss** (Eq. 3 di paper). Lihat langkah 8c dan
  catatan reproduksi di bawah.
- **YOLO26n-depth** dan **YOLO26s-depth** -- arsitektur *native* Ultralytics untuk
  monocular depth estimation (`ultralytics/cfg/models/26/yolo26-depth.yaml`, skala
  `n`/`s`), **tidak diubah**. Training, loss (SILog + gradient), dan kalibrasi skala
  metrik pasca-training sepenuhnya ditangani oleh trainer/validator bawaan
  Ultralytics (`ultralytics.models.yolo.depth`) -- kode di sini hanya membungkusnya.

Kelima model (RT-MonoDepth + 4 pembanding) tidak menyentuh
`wheelchair_nav/perception/` -- RT-MonoDepth + YOLO26-nano detektor tetap stack
default sistem navigasi. `baselines/` dan skrip `*_fastdepth_*`/`*_ghostdepth_*`/
`*_yolo_depth_*` di atas murni untuk studi perbandingan; `run_navigation.py`
tetap bisa memakai model manapun lewat `--depth_model` kalau ingin dibandingkan
end-to-end.

### 8a. Siapkan layout dataset yang identik untuk YOLO26{n,s}-depth

`scripts/prepare_sunrgbd.py` (langkah 1) bisa langsung menge-*mirror* (symlink,
tanpa duplikasi data) split train/val/test yang sama persis ke layout
`images/{split}/*.jpg` + `depth/{split}/*.npy` yang dipakai native depth task
Ultralytics:

```bash
python -m wheelchair_nav.scripts.prepare_sunrgbd \
    --sunrgbd_root /path/to/SUNRGBD \
    --out_dir ./wheelchair_nav/data/sunrgbd_processed \
    --splits_dir ./wheelchair_nav/splits_sunrgbd \
    --make_yolo_depth_layout --yolo_depth_out_dir ./wheelchair_nav/data/sunrgbd_yolo_depth
```

Menghasilkan `./wheelchair_nav/data/sunrgbd_yolo_depth/depth_comparison.yaml` (siap dipakai
`--data` di langkah 8d) -- gambar yang di dalamnya **sama persis** dengan yang
dipakai RT-MonoDepth/FastDepth via `splits_sunrgbd/*.txt`.

### 8b. FastDepth: tuning lalu training (sama pola dengan RT-MonoDepth)

```bash
# 1) Hyperparameter tuning (Optuna, TPE) -- ruang pencarian & protokol identik
#    dengan tune_depth_sunrgbd.py (langkah 2), supaya kedua model di-tuning
#    dengan cara yang sama:
python -m wheelchair_nav.scripts.tune_fastdepth_sunrgbd \
    --splits_dir ./wheelchair_nav/splits_sunrgbd \
    --height 288 --width 384 \
    --n_trials 30 --epochs_per_trial 5 \
    --out_json ./wheelchair_nav/log_fastdepth/optuna_best_fastdepth_hparams.json \
    --device cuda

# 2) Training penuh, from scratch, pakai hasil tuning
python -m wheelchair_nav.scripts.train_fastdepth_sunrgbd \
    --splits_dir ./wheelchair_nav/splits_sunrgbd \
    --log_dir ./wheelchair_nav/log_fastdepth \
    --model_name FastDepth_sunrgbd \
    --height 288 --width 384 \
    --min_depth 0.1 --max_depth 10.0 \
    --hparams_json ./wheelchair_nav/log_fastdepth/optuna_best_fastdepth_hparams.json
```

Checkpoint tersimpan di
`./wheelchair_nav/log_fastdepth/FastDepth_sunrgbd/models/{weights_N,best}/fastdepth.pth`.

Sama seperti RT-MonoDepth, kurva train vs val loss otomatis tersimpan di
`./wheelchair_nav/log_fastdepth/FastDepth_sunrgbd/training_curve.png`
(+ `training_log.csv`).

### 8c. Ghost-Depth: tuning lalu training

Reimplementasi dari Quan et al., CNIOT '23
([doi:10.1145/3603781.3603861](https://doi.org/10.1145/3603781.3603861)).
Arsitekturnya (`networks/GhostDepth/ghost_depth.py`), sesuai paper sec. 3.1:

```
input        3 x H     x W
encoder /2  16 x H/2   x W/2    -> skip (addition)
encoder /4  24 x H/4   x W/4    -> skip (addition)
encoder /8  40 x H/8   x W/8    -> skip (iAFF)          <- hanya di sini
encoder /16 80 x H/16  x W/16   -> skip (addition)
encoder /32 -> conv penurun channel -> 4x upsampling module
             (bilinear x2 -> fuse skip -> Ghost-A -> Ghost-B)
output       1 x H/2   x W/2
```

- **Encoder**: GhostNet [Han et al., CVPR 2020] tanpa layer klasifikasi.
- **Decoder**: Ghost convolution menggantikan konvolusi 3x3 biasa; **Ghost-A**
  menurunkan jumlah channel, **Ghost-B** merapikan tanpa mengubah channel.
- **iAFF** [Dai et al., WACV 2021] hanya dipasang di skip 40-channel -- persis
  konfigurasi terbaik di ablation Table 3 paper; skip lain pakai penjumlahan biasa.
- **Loss**: berHu / *reverse Huber* (Eq. 3), dengan `c = 0.2 x max|error|` per
  gambar. Default `--loss berhu`; pakai `--loss l1_silog` kalau ingin meng-ablasi
  arsitektur di bawah objective yang sama dengan RT-MonoDepth/FastDepth.

```bash
# 1) Hyperparameter tuning (Optuna, TPE) -- protokol & objective identik
#    dengan tune_depth_sunrgbd.py / tune_fastdepth_sunrgbd.py
python -m wheelchair_nav.scripts.tune_ghostdepth_sunrgbd \
    --splits_dir ./wheelchair_nav/splits_sunrgbd \
    --height 288 --width 384 \
    --n_trials 30 --epochs_per_trial 5 \
    --out_json ./wheelchair_nav/log_ghostdepth/optuna_best_ghostdepth_hparams.json \
    --device cuda

# 2) Training penuh, from scratch, pakai hasil tuning.
#    Default optimizer mengikuti paper sec. 4.2: Adam(0.9, 0.999), wd 1e-4,
#    lr 1e-4 turun 10x tiap 30 epoch, batch 8, 55 epoch.
python -m wheelchair_nav.scripts.train_ghostdepth_sunrgbd \
    --splits_dir ./wheelchair_nav/splits_sunrgbd \
    --log_dir ./wheelchair_nav/log_ghostdepth \
    --model_name GhostDepth_sunrgbd \
    --height 288 --width 384 \
    --min_depth 0.1 --max_depth 10.0 \
    --hparams_json ./wheelchair_nav/log_ghostdepth/optuna_best_ghostdepth_hparams.json
```

Checkpoint tersimpan di
`./wheelchair_nav/log_ghostdepth/GhostDepth_sunrgbd/models/{weights_N,best}/ghostdepth.pth`,
kurva training di `.../GhostDepth_sunrgbd/training_curve.png` (+ `training_log.csv`).

**Catatan reproduksi (penting kalau angka ini mau dikutip sebagai "Ghost-Depth"):**
paper menjelaskan arsitektur dalam bentuk prosa, bukan tabel layer, jadi ada
beberapa detail yang harus disimpulkan. Semuanya didokumentasikan di docstring
`networks/GhostDepth/ghost_depth.py`, yang utama:

- Jumlah parameter reimplementasi ini **2.79M** (default) atau **2.57M** dengan
  `--no_final_conv`; paper melaporkan **2.71M**, di antara keduanya -- jadi tidak
  ada bacaan yang persis. Perbedaannya di apakah `ConvBnAct(160->960)` GhostNet
  ikut dihitung sebagai "layer klasifikasi" yang dibuang atau tidak.
- Paper melatih encoder dari **bobot ImageNet**; di repo ini Ghost-Depth dilatih
  **from scratch**, konsisten dengan RT-MonoDepth dan FastDepth, supaya
  perbandingannya mengisolasi arsitektur bukan pretraining. **Akurasinya karena
  itu tidak diharapkan mereproduksi angka paper** (REL 0.149 / δ1 0.797 di
  NYU-Depth V2 dengan input 228x304).

### 8d. YOLO26n-depth / YOLO26s-depth: tuning lalu training

```bash
# 1) Hyperparameter tuning (Optuna, TPE) -- optimizer + bobot loss depth
#    (dlog: SILog gain, dgrad: gradient-loss gain, dlam: SILog scale-invariance
#    focus) + augmentasi; skor = validation abs_rel (diminimalkan)
python -m wheelchair_nav.scripts.tune_yolo_depth \
    --variant n \
    --data ./wheelchair_nav/data/sunrgbd_yolo_depth/depth_comparison.yaml \
    --n_trials 30 --epochs_per_trial 10 --imgsz 384 --batch 16 --device 0 \
    --out_json ./wheelchair_nav/log_yolo_depth/optuna_best_yolo26n_depth_hparams.json

# 2) Training penuh, pakai hasil tuning
python -m wheelchair_nav.scripts.train_yolo_depth \
    --variant n \
    --data ./wheelchair_nav/data/sunrgbd_yolo_depth/depth_comparison.yaml \
    --imgsz 384 --batch 16 --device 0 \
    --hparams_json ./wheelchair_nav/log_yolo_depth/optuna_best_yolo26n_depth_hparams.json
```

Ganti `--variant n` menjadi `--variant s` untuk YOLO26s-depth (tuning dan training
terpisah, sama perintah). Checkpoint tersimpan di
`./wheelchair_nav/log_yolo_depth/yolo26{n,s}_depth_sunrgbd/weights/best.pt` -- sudah otomatis
dikalibrasi skala metriknya oleh Ultralytics di akhir training (log
`"Auto-calibration written to best.pt"`).

Default `--pretrained` kosong (`""`) -> training dari bobot acak
(`yolo26{n,s}-depth.yaml`), konsisten dengan kebijakan "from scratch" di semua
model lain; isi `--pretrained` dengan path checkpoint kalau memang butuh
memulai dari bobot tertentu.

### 8e. Jalankan perbandingan apples-to-apples

```bash
python -m wheelchair_nav.evaluation.eval_depth_comparison \
    --test_list ./wheelchair_nav/splits_sunrgbd/test.txt \
    --rtmonodepth_weights_dir ./wheelchair_nav/log_sunrgbd/RTMonoDepth_sunrgbd/models/best \
    --fastdepth_weights_dir ./wheelchair_nav/log_fastdepth/FastDepth_sunrgbd/models/best \
    --ghostdepth_weights_dir ./wheelchair_nav/log_ghostdepth/GhostDepth_sunrgbd/models/best \
    --yolo26n_depth_weights ./wheelchair_nav/log_yolo_depth/yolo26n_depth_sunrgbd/weights/best.pt \
    --yolo26s_depth_weights ./wheelchair_nav/log_yolo_depth/yolo26s_depth_sunrgbd/weights/best.pt \
    --device cuda \
    --out_csv ./wheelchair_nav/log_sunrgbd/depth_comparison.csv
```

Boleh isi hanya sebagian flag `--*_weights*` -- model yang tidak diberi bobotnya
otomatis dilewati. Dihitung di atas **gambar test yang sama** dan **rumus metrik yang
sama**, lalu mencetak **dua tabel** (keduanya masuk ke `--out_csv` bila diisi):

1. **Tabel akurasi + biaya** -- `abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3,
   Params(M), GMACs, GMACs@ref, FeedHxW, FPS, p95(ms)`.
2. **Tabel latency** -- `mean, std, min, p50, p90, p95, p99, max` (ms) + FPS, diukur
   pada jalur `.infer()` penuh (resize → H2D → forward → D2H) dengan protokol identik
   untuk kelima model.

`p95(ms)` sengaja muncul di kedua tabel: **mean hanya bilang seberapa cepat model
*biasanya*, p95 yang menentukan apakah model itu cukup cepat.** FPS diturunkan dari
mean latency (`1000 / mean_ms`), jadi keduanya tidak mungkin saling bertentangan.

Atur banyaknya sampel dengan `--latency_cycles` (default 200). p99 diinterpolasi dari
~1% sampel teratas, jadi **naikkan ke 500+ kalau p99-nya mau dikutip di paper.** Nama
lama `--fps_cycles` / `--fps_warmup` tetap berfungsi sebagai alias.

**Soal dua kolom GMACs.** Kelima model melihat **konten gambar di resolusi yang
sama** (288x384, lihat kotak di bawah), tapi bentuk *tensor*-nya berbeda: model
Ultralytics *letterbox* ke kanvas persegi `imgsz x imgsz` = 384x384, jadi tensornya
lebih besar dari konten yang dibawanya (sisanya padding abu-abu). Karena MACs *dan*
FPS berskala dengan ukuran **tensor**, tabelnya melaporkan:

- `GMACs` -- di bentuk tensor *as-deployed* tiap model (biaya nyata saat dijalankan;
  overhead padding YOLO ikut terhitung di sini, sebagaimana mestinya)
- `GMACs@ref` -- semua model di **satu** resolusi yang sama (`--macs_ref_hw`,
  default = `INPUT_HEIGHT x INPUT_WIDTH`), untuk membandingkan **arsitektur**
- `FeedHxW` -- bentuk tensor yang benar-benar dipakai, supaya tabelnya
  mendokumentasikan bebannya sendiri

Pakai `--macs_ref_hw none` kalau hanya ingin kolom *as-deployed*.

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
  **satu-satunya** sumber dataset berlabel untuk training YOLO26-nano; yang berasal
  dari luar hanyalah bobot awal COCO-pretrained (`yolo26n.pt`), lihat langkah 5
  untuk alasan metodologisnya. Kelima model depth **tidak** memakai checkpoint
  pretrained apa pun. Scene yang tidak punya
  `annotation2Dfinal/` atau JSON-nya tidak valid dilewati dan dihitung, bukan
  diganti sumber lain -- kalau jumlah scene yang berhasil dikonversi terlalu sedikit,
  periksa `--sunrgbd_root` dan `--exclude_classes`, bukan beralih dataset.
- Hyperparameter tuning (langkah 2, 4, dan 8b/8c) memakai Optuna dengan
  `TPESampler` (Bayesian, Tree-structured Parzen Estimator) dan budget epoch yang
  sengaja lebih kecil dari training penuh -- ini proxy search, bukan pengganti
  training penuh; hasil terbaiknya tetap perlu dilatih ulang dengan
  `--num_epochs`/`--epochs` penuh di langkah 3/5/8b/8c/8d.
- FastDepth, Ghost-Depth, YOLO26n-depth, dan YOLO26s-depth (langkah 8) adalah model
  pembanding untuk studi evaluasi; sistem navigasi (`run_navigation.py`) secara
  default tetap memakai RT-MonoDepth + YOLO26-nano detektor seperti dijelaskan di
  langkah 1-7, tapi `--depth_model` bisa menukar model depth-nya kalau ingin
  membandingkan dampaknya end-to-end.
- Ghost-Depth adalah **reimplementasi** dari paper (bukan kode resmi penulisnya, yang
  tidak dirilis). Beberapa detail arsitektur disimpulkan dari prosa paper dan
  didokumentasikan di docstring `networks/GhostDepth/ghost_depth.py` -- baca itu dulu
  sebelum mengutip angkanya sebagai hasil "Ghost-Depth".
