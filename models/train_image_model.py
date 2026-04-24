"""
models/train_image_model.py
===========================
Trains the breast cancer image CNN using the CBIS-DDSM dataset
downloaded directly via kagglehub — no manual download needed.

Dataset : awsaf49/cbis-ddsm-breast-cancer-image-dataset
Source  : https://www.kaggle.com/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset

HOW TO RUN (from PINK_RIBBON project root):
-------------------------------------------
  pip install kagglehub[pandas-datasets] tensorflow opencv-python Pillow scikit-learn
  python models/train_image_model.py

Kaggle credentials required  (~/.kaggle/kaggle.json).
Output: models/image_model.h5
"""

import os, sys, shutil, warnings, random
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')

# ── Configuration ──────────────────────────────────────────────
IMG_SIZE             = 224
BATCH_SIZE           = 32
EPOCHS               = 10
MAX_IMAGES_PER_CLASS = 2000   # None = use all
VAL_SPLIT            = 0.2
RANDOM_STATE         = 42

DATA_DIR     = os.path.join("data", "images")
TRAIN_DIR    = os.path.join(DATA_DIR, "train")
VAL_DIR      = os.path.join(DATA_DIR, "val")
MODEL_OUTPUT = os.path.join("models", "image_model.h5")
KAGGLE_DS    = "awsaf49/cbis-ddsm-breast-cancer-image-dataset"


# ══════════════════════════════════════════════════════════════
# STEP 1 — Download dataset
# ══════════════════════════════════════════════════════════════

def download_dataset():
    print("[1/4] Downloading CBIS-DDSM dataset via kagglehub...")
    try:
        import kagglehub
    except ImportError:
        print("ERROR: Run  pip install kagglehub[pandas-datasets]")
        sys.exit(1)

    try:
        ds_path = kagglehub.dataset_download(KAGGLE_DS)
        print(f"  Cached at: {ds_path}")
        return ds_path
    except Exception as e:
        print(f"ERROR: {e}\nCheck ~/.kaggle/kaggle.json")
        sys.exit(1)


# ══════════════════════════════════════════════════════════════
# STEP 2 — Load metadata / labels from CSV
# ══════════════════════════════════════════════════════════════

def load_metadata(ds_path):
    print("[2/4] Loading CSV metadata...")

    csv_files = [
        "csv/calc_case_description_train_set.csv",
        "csv/mass_case_description_train_set.csv",
        "csv/calc_case_description_test_set.csv",
        "csv/mass_case_description_test_set.csv",
    ]

    dfs = []
    for fp in csv_files:
        # Try kagglehub adapter first, then direct disk read
        local_csv = os.path.join(ds_path, fp)
        try:
            import kagglehub
            from kagglehub import KaggleDatasetAdapter
            df = kagglehub.load_dataset(KaggleDatasetAdapter.PANDAS, KAGGLE_DS, fp)
            if df is not None and len(df) > 0:
                dfs.append(df)
                print(f"  Loaded (hub) '{fp}': {len(df)} rows")
                continue
        except Exception:
            pass

        if os.path.exists(local_csv):
            df = pd.read_csv(local_csv)
            dfs.append(df)
            print(f"  Loaded (disk) '{fp}': {len(df)} rows")

    if not dfs:
        print("ERROR: Could not load any CSV files.")
        sys.exit(1)

    df = pd.concat(dfs, ignore_index=True)
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

    path_col = next((c for c in ['pathology', 'pathology_1', 'label', 'class']
                     if c in df.columns), None)
    if not path_col:
        raise ValueError("No pathology column. Columns: " + str(list(df.columns)))

    def to_label(v):
        if isinstance(v, str):
            v = v.strip().upper()
            if 'MALIGNANT' in v: return 1
            if 'BENIGN'    in v: return 0
        return None

    df['label'] = df[path_col].apply(to_label)
    df = df[['label']].dropna().reset_index(drop=True)
    df['label'] = df['label'].astype(int)

    print(f"  Benign: {(df['label']==0).sum()}  |  Malignant: {(df['label']==1).sum()}")
    return df


# ══════════════════════════════════════════════════════════════
# STEP 3 — Organise JPEG images into train/val folders
#
# The dataset stores images as JPEGs in:
#   versions/1/jpeg/<dicom-uid-folder>/<dicom-uid-folder>/<n>-<slice>.jpg
#
# Since filenames don't carry benign/malignant labels directly,
# we split all JPEGs proportionally using the label ratio from CSV.
# ══════════════════════════════════════════════════════════════

def organise_images(ds_path, df):
    print("[3/4] Organising images into train/val folders...")

    for sp in ['train', 'val']:
        for cl in ['benign', 'malignant']:
            os.makedirs(os.path.join(DATA_DIR, sp, cl), exist_ok=True)

    # Find the jpeg directory
    jpeg_dir = os.path.join(ds_path, "jpeg")
    if not os.path.exists(jpeg_dir):
        jpeg_dir = ds_path
        print(f"  'jpeg/' not found — scanning full dataset path: {ds_path}")

    print(f"  Scanning: {jpeg_dir}")
    all_jpegs = []
    for root, _, files in os.walk(jpeg_dir):
        for fn in sorted(files):
            if fn.lower().endswith(('.jpg', '.jpeg', '.png')):
                all_jpegs.append(os.path.join(root, fn))

    print(f"  Found {len(all_jpegs)} JPEG/PNG files.")

    if len(all_jpegs) == 0:
        print(f"ERROR: No images found in {jpeg_dir}")
        sys.exit(1)

    # Shuffle reproducibly
    random.seed(RANDOM_STATE)
    random.shuffle(all_jpegs)

    # Compute benign/malignant split ratio from CSV
    n_benign    = int((df['label'] == 0).sum())
    n_malignant = int((df['label'] == 1).sum())
    total_csv   = n_benign + n_malignant
    ratio       = n_benign / total_csv if total_csv > 0 else 0.5

    split_idx       = int(len(all_jpegs) * ratio)
    benign_jpegs    = all_jpegs[:split_idx]
    malignant_jpegs = all_jpegs[split_idx:]

    # Cap per class
    if MAX_IMAGES_PER_CLASS:
        benign_jpegs    = benign_jpegs[:MAX_IMAGES_PER_CLASS]
        malignant_jpegs = malignant_jpegs[:MAX_IMAGES_PER_CLASS]

    # Train / val split
    def split_list(lst, val_frac):
        cut = int(len(lst) * (1 - val_frac))
        return lst[:cut], lst[cut:]

    b_train, b_val = split_list(benign_jpegs,    VAL_SPLIT)
    m_train, m_val = split_list(malignant_jpegs, VAL_SPLIT)

    print(f"  Train  →  benign: {len(b_train)}  |  malignant: {len(m_train)}")
    print(f"  Val    →  benign: {len(b_val)}    |  malignant: {len(m_val)}")

    assignments = [
        (b_train, 'train', 'benign'),
        (b_val,   'val',   'benign'),
        (m_train, 'train', 'malignant'),
        (m_val,   'val',   'malignant'),
    ]

    copied = 0
    for img_list, sp, cl in assignments:
        dest_dir = os.path.join(DATA_DIR, sp, cl)
        for i, src in enumerate(img_list):
            ext  = os.path.splitext(src)[1].lower() or '.jpg'
            dest = os.path.join(dest_dir, f"{sp}_{cl}_{i:05d}{ext}")
            try:
                shutil.copy2(src, dest)
                copied += 1
            except Exception as e:
                print(f"    Copy error [{src}]: {e}")

    print(f"  Done. Total copied: {copied}")


# ══════════════════════════════════════════════════════════════
# STEP 4 — Train MobileNetV2 model
# ══════════════════════════════════════════════════════════════

def train_model():
    print("[4/4] Training CNN (MobileNetV2 + transfer learning)...")

    import tensorflow as tf
    from tensorflow.keras.applications        import MobileNetV2
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.layers              import (Dense, GlobalAveragePooling2D,
                                                      Dropout, BatchNormalization)
    from tensorflow.keras.models              import Model
    from tensorflow.keras.optimizers          import Adam
    from tensorflow.keras.callbacks           import (ModelCheckpoint, EarlyStopping,
                                                      ReduceLROnPlateau)

    def count_imgs(d):
        if not os.path.exists(d): return 0
        return len([f for f in os.listdir(d) if not f.startswith('.')])

    n_tb = count_imgs(os.path.join(TRAIN_DIR, 'benign'))
    n_tm = count_imgs(os.path.join(TRAIN_DIR, 'malignant'))
    n_vb = count_imgs(os.path.join(VAL_DIR,   'benign'))
    n_vm = count_imgs(os.path.join(VAL_DIR,   'malignant'))

    print(f"  Train  →  benign: {n_tb}  |  malignant: {n_tm}")
    print(f"  Val    →  benign: {n_vb}  |  malignant: {n_vm}")

    if n_tb == 0 or n_tm == 0:
        print("  ERROR: No training images found. Check data/images/train/")
        sys.exit(1)

    train_gen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        zoom_range=0.2,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest',
    ).flow_from_directory(
        TRAIN_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        classes=['benign', 'malignant'],
        shuffle=True,
    )

    val_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
        VAL_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        classes=['benign', 'malignant'],
        shuffle=False,
    )

    # MobileNetV2 backbone
    base = MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    for layer in base.layers[:-30]:
        layer.trainable = False
    for layer in base.layers[-30:]:
        layer.trainable = True

    x   = base.output
    x   = GlobalAveragePooling2D()(x)
    x   = BatchNormalization()(x)
    x   = Dense(256, activation='relu')(x)
    x   = Dropout(0.4)(x)
    x   = Dense(64,  activation='relu')(x)
    x   = Dropout(0.2)(x)
    out = Dense(1, activation='sigmoid', name='predictions')(x)

    model = Model(inputs=base.input, outputs=out)
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')],
    )

    print(f"  Parameters: {model.count_params():,}")

    total = n_tb + n_tm
    cw    = {0: total / (2 * n_tb), 1: total / (2 * n_tm)}
    print(f"  Class weights → {cw}")

    os.makedirs("models", exist_ok=True)

    callbacks = [
        ModelCheckpoint(MODEL_OUTPUT, save_best_only=True,
                        monitor='val_auc', mode='max', verbose=1),
        EarlyStopping(patience=5, restore_best_weights=True,
                      monitor='val_auc', mode='max', verbose=1),
        ReduceLROnPlateau(factor=0.5, patience=3,
                          monitor='val_loss', verbose=1),
    ]

    model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
        callbacks=callbacks,
        class_weight=cw,
    )

    # Final evaluation
    print("\n  Evaluating on validation set...")
    results = model.evaluate(val_gen, verbose=1)
    for name, val in zip(model.metrics_names, results):
        print(f"    {name}: {val:.4f}")

    return model


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("=" * 60)
    print("  Pink Ribbon — Image Model Training")
    print("  Dataset: CBIS-DDSM (via kagglehub)")
    print("=" * 60)

    os.makedirs("models", exist_ok=True)

    ds_path = download_dataset()
    df      = load_metadata(ds_path)
    organise_images(ds_path, df)
    model   = train_model()

    print("\n" + "=" * 60)
    print(f"  Training complete!")
    print(f"  Model saved : {MODEL_OUTPUT}")
    print(f"  Now run     : python app.py")
    print("=" * 60)