import os
from pathlib import Path
import sys
import numpy  as np
import wfdb
from scipy.signal import butter, filtfilt
from sklearn.model_selection import train_test_split
from sklearn.ensemble      import RandomForestClassifier
from sklearn.metrics       import classification_report, confusion_matrix
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv1D, MaxPooling1D,
    Flatten, Dense
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping # 导入 EarlyStopping
import matplotlib.pyplot as plt
import seaborn as sns # 用于绘制热力图，方便展示混淆矩阵

# ———— 参数设置 ————
BASE_DIR   = Path(__file__).resolve().parent

DATA_DIR = Path('D:/Chrome/mit')

FS          = 250
EPOCH_SEC   = 30
EPOCH_LEN   = FS * EPOCH_SEC
TEST_SIZE   = 0.15
VAL_SIZE    = 0.15
RANDOM_SEED = 42

# CNN 训练参数
CNN_EPOCHS = 10 # <-- 增加 CNN 的训练轮次
BATCH_SIZE = 32
PATIENCE = 3 # <-- Early Stopping 的容忍次数，即连续多少个 epoch 验证指标没有改善就停止

# 检查数据目录
if not DATA_DIR.exists():
    print(f"Error: DATA_DIR '{DATA_DIR}' 不存在，请检查数据解压路径。", file=sys.stderr)
    print("Please download a sleep database (e.g., PhysioNet Sleep-Apnea Expanded, or Sleep-EDF Expanded) and extract it to the specified DATA_DIR.")
    sys.exit(1)

# ———— 辅助函数 ————
def bandpass_filter(sig, low=0.4, high=40, fs=FS, order=4):
    """带通滤波"""
    nyq = 0.5 * fs
    lowcut = low / nyq
    highcut = high / nyq
    if highcut >= 1:
         highcut = 0.99 # 避免超过奈奎斯特频率
    if lowcut >= highcut:
         lowcut = highcut * 0.5 # 避免低频高于高频

    # 检查信号长度，filtfilt 需要信号长度大于 3 * order
    if len(sig) < 3 * order:
        # print(f"Warning: Signal length ({len(sig)}) is too short for filter order ({order}). Returning original signal.", file=sys.stderr)
        # 对于太短的信号（不应该是完整的 epoch），直接返回原信号，load_record_epochs 会检查长度
        return sig

    b, a = butter(order, [lowcut, highcut], btype='band')
    try:
        return filtfilt(b, a, sig)
    except ValueError as e:
        print(f"Filtering error: {e} for signal length {len(sig)}", file=sys.stderr)
        # Fallback or raise error if filtering fails
        return sig # Return original or handle as appropriate


def load_record_epochs(recname):
    """加载单个记录的信号和标注，切分并预处理 epoch"""
    rec_path = DATA_DIR / recname
    print(f"Processing record: {recname}")
    try:
        rec = wfdb.rdrecord(str(rec_path))
        if rec.p_signal is None or rec.p_signal.shape[1] < 1:
            print(f"  Error: record {recname} signal is invalid or has less than 1 channel.", file=sys.stderr)
            return [], []

        channel_idx = -1
        target_channels = ['EEG', 'Pz-Oz', 'Pz-O2']
        used_channel_name = ""
        for i, name in enumerate(rec.sig_name):
             if any(tc.lower() in name.lower() for tc in target_channels): # Case-insensitive check
                  channel_idx = i
                  used_channel_name = name
                  print(f"  Using channel: {name} (index {i})")
                  break
        if channel_idx == -1:
             channel_idx = 0 # Fallback to first channel if none found
             used_channel_name = rec.sig_name[0]
             print(f"  Could not find specific EEG channel, using default channel 0: {used_channel_name}")

        sig = rec.p_signal[:, channel_idx] # shape (n_samples,)

        ann = None
        for ext in ['st', 'apn']: # Common sleep stage extensions
             try:
                  ann = wfdb.rdann(str(rec_path), extension=ext)
                  print(f"  Loaded annotations with extension .{ext}")
                  break
             except FileNotFoundError:
                  continue
             except Exception as e:
                  print(f"  Error loading annotation {ext} for {recname}: {e}", file=sys.stderr)
                  continue

        if ann is None:
             print(f"  Error: No usable annotations found for {recname} with extensions .st or .apn", file=sys.stderr)
             return [], []

        # --- Stage mapping (already implements 4 classes) ---
        stage_char_map = {
            'W': 0,    # Wake
            '1': 1,    # Stage 1 -> Light Sleep
            '2': 1,    # Stage 2 -> Light Sleep
            '3': 2,    # Stage 3 -> Deep Sleep
            '4': 2,    # Stage 4 -> Deep Sleep (if database uses Stage 4)
            'R': 3     # REM
        }
        valid_stage_chars = set(stage_char_map.keys())
        num_defined_classes = len(set(stage_char_map.values())) # Should be 4

    except FileNotFoundError:
        print(f"  Error: Record files for {recname} not found.", file=sys.stderr)
        return [], []
    except Exception as e:
        print(f"  Error loading record {recname}: {e}", file=sys.stderr)
        return [], []

    Xs, labels = [], []
    valid_epochs_count = 0

    # Iterate through annotations assuming sample points mark the START of epochs
    for i, (aux_note, sample) in enumerate(zip(ann.aux_note, ann.sample)):
        if not aux_note:
            continue

        cleaned_note = aux_note.strip('\x00').split(' ')[0]
        if not cleaned_note or cleaned_note[0] not in valid_stage_chars:
             continue

        stage_char = cleaned_note[0]
        lab = stage_char_map[stage_char]

        start, end = sample, sample + EPOCH_LEN

        if end <= len(sig):
            try:
                ep = sig[start:end]
                if len(ep) != EPOCH_LEN:
                     print(f"  Warning: record {recname}, annotation {i}, extracted epoch length ({len(ep)}) != expected ({EPOCH_LEN}). Skipping.", file=sys.stderr)
                     continue

                ep_filtered = bandpass_filter(ep)

                if not np.isfinite(ep_filtered).all() or np.std(ep_filtered) < 1e-7:
                     continue

                ep_normalized = (ep_filtered - np.mean(ep_filtered)) / (np.std(ep_filtered) + 1e-6)

                Xs.append(ep_normalized.reshape(-1,1))
                labels.append(lab)
                valid_epochs_count += 1

            except Exception as e:
                 print(f"  Error processing epoch for annotation {i} (aux='{aux_note}', sample={sample}): {e}", file=sys.stderr)
                 continue
        else:
             break # Stop if epoch goes beyond signal end

    print(f"  Successfully loaded {valid_epochs_count} valid epochs.")
    if valid_epochs_count > 0:
        loaded_classes = set(labels)
        missing_classes = set(stage_char_map.values()) - loaded_classes
        if missing_classes:
             print(f"  Warning: Record {recname} is missing expected classes {missing_classes}.", file=sys.stderr)

    return Xs, labels

# ———— 获取记录列表 ————
record_names = []
try:
    potential_record_names = wfdb.get_record_list(str(DATA_DIR))
    if potential_record_names:
         record_names = [rn for rn in potential_record_names if not rn.upper() == 'RECORDS']
         print(f"wfdb.get_record_list scanned {len(record_names)} records.")
    else:
         print("wfdb.get_record_list returned empty. Falling back to manual .hea scan.")
         record_names = sorted(fp.stem for fp in DATA_DIR.iterdir() if fp.suffix == '.hea' and fp.stem.lower() != 'records')
         print(f"Manual .hea scan found {len(record_names)} records.")

except Exception as e:
    print(f"Error getting record list using wfdb: {e}", file=sys.stderr)
    print("Attempting manual .hea file scan...")
    record_names = sorted(fp.stem for fp in DATA_DIR.iterdir() if fp.suffix == '.hea' and fp.stem.lower() != 'records')
    print(f"Manual .hea scan found {len(record_names)} records.")


if not record_names:
    print("Error: No record files (.hea) found in DATA_DIR. Please check the path and contents.", file=sys.stderr)
    sys.exit(1)

# ———— 构建数据集 ————
all_X, all_y = [], []

print(f"\nStarting to load data from {len(record_names)} records...")
for recname in record_names:
    Xr, yr = load_record_epochs(recname)
    all_X.extend(Xr)
    all_y.extend(yr)

print(f"\nTotal epochs collected: {len(all_X)}")

if not all_X:
    print("Error: No epochs collected. Please check data files, annotation files, and the label parsing logic in load_record_epochs.", file=sys.stderr)
    sys.exit(1)

all_X = np.stack(all_X, axis=0)
all_y = np.array(all_y)

target_names = ['Wake','Light Sleep','Deep Sleep','REM']
target_int_map = {name: i for i, name in enumerate(target_names)}
int_target_map = {i: name for i, name in enumerate(target_names)}
num_classes = len(target_names) # Number of classes is 4


# ———— 可视化：数据类别分布 ————
print("\nVisualizing data class distribution...")
unique_classes, class_counts = np.unique(all_y, return_counts=True)

plt.figure(figsize=(10, 6))
class_labels = [int_target_map.get(i, f'Unknown_{i}') for i in unique_classes]
plt.bar(class_labels, class_counts, color='skyblue')
plt.title('Dataset Class Distribution (Before Split)')
plt.xlabel('Sleep Stage')
plt.ylabel('Number of Epochs')
plt.xticks(rotation=0)
for i, count in enumerate(class_counts):
    plt.text(i, count + 5, str(count), ha='center', va='bottom')
plt.tight_layout()
plt.show()


# ———— 可视化：样本 epoch ————
print("\nVisualizing sample processed epochs...")
num_samples_to_show = min(4, len(all_X))
if num_samples_to_show > 0:
    sample_indices = []
    unique_classes_present = np.unique(all_y)
    class_indices = {cls: np.where(all_y == cls)[0] for cls in unique_classes_present}

    for cls in unique_classes_present:
        indices = class_indices[cls]
        if len(indices) > 0 and len(sample_indices) < num_samples_to_show:
             sample_indices.append(np.random.choice(indices))

    if len(sample_indices) < num_samples_to_show:
         added_count = 0
         existing_indices_set = set(sample_indices)
         for i in range(len(all_X)):
              if i not in existing_indices_set:
                   sample_indices.append(i)
                   added_count += 1
                   if len(sample_indices) >= num_samples_to_show:
                        break

    plt.figure(figsize=(12, num_samples_to_show * 3))
    for i, idx in enumerate(sample_indices):
        plt.subplot(num_samples_to_show, 1, i + 1)
        plt.plot(all_X[idx].flatten())
        plt.title(f'Sample Processed Epoch - Class: {int_target_map.get(all_y[idx], "Unknown")}')
        plt.xlabel('Sample Point')
        plt.ylabel('Amplitude (Normalized)')
        plt.grid(True)
    plt.tight_layout()
    plt.show()


# ———— 数据划分 ————
print("\nSplitting dataset...")
unique_y, counts_y = np.unique(all_y, return_counts=True)
if np.any(counts_y < 2):
     print("Warning: Some classes have fewer than 2 samples and cannot be stratified. This might cause errors or warnings in train_test_split.", file=sys.stderr)

try:
    X_temp, X_test, y_temp, y_test = train_test_split(
        all_X, all_y, test_size=TEST_SIZE,
        random_state=RANDOM_SEED, stratify=all_y
    )
    val_ratio = VAL_SIZE / (1 - TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio,
        random_state=RANDOM_SEED, stratify=y_temp # Stratify based on y_temp
    )
except ValueError as e:
     print(f"Error during train_test_split (stratify issue): {e}", file=sys.stderr)
     print("This usually happens if a class has only one sample in a subset and stratify is used.")
     print("Consider removing classes with very few samples or adjusting split sizes (TEST_SIZE/VAL_SIZE).")
     sys.exit(1)


print(f"Dataset shapes after split: X_train={X_train.shape}, y_train={y_train.shape}, X_val={X_val.shape}, y_val={y_val.shape}, X_test={X_test.shape}, y_test={y_test.shape}")


# ———— 定义 CNN 特征提取器 ————
# Input shape depends on EPOCH_LEN (FS * EPOCH_SEC)
inp = Input(shape=(EPOCH_LEN,1))
x   = Conv1D(32, kernel_size=64, strides=2, activation='relu')(inp)
x   = MaxPooling1D(8)(x)
x   = Conv1D(64, kernel_size=32, strides=2, activation='relu')(x)
x   = MaxPooling1D(8)(x)
x   = Flatten()(x)
feat = Dense(128, activation='relu', name='feat')(x) # Define feature layer

# ———— CNN 训练 ————
# Build the full CNN classification model for pre-training
head = Dense(num_classes, activation='softmax')(feat)
cnn_cls = Model(inputs=inp, outputs=head)

# Compile model
cnn_cls.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=Adam(1e-3),
    metrics=['accuracy']
)

print("\nCNN Classifier Model Summary:")
cnn_cls.summary()

# Define Early Stopping callback
early_stopping = EarlyStopping(
    monitor='val_accuracy', # Monitor validation accuracy
    patience=PATIENCE,      # Number of epochs with no improvement after which training will be stopped
    mode='max',             # Stop when validation accuracy stops increasing
    restore_best_weights=True # Restore model weights from the epoch with the best value of the monitored quantity.
)


# Train model
print("\nStarting CNN pre-training...")
history = cnn_cls.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=CNN_EPOCHS,      # Use the defined number of epochs
    batch_size=BATCH_SIZE,  # Use the defined batch size
    verbose=1,              # Show training progress
    callbacks=[early_stopping] # Add Early Stopping callback
)
print("CNN pre-training finished.")

# ———— 可视化：CNN 训练历史 ————
print("\nVisualizing CNN training history...")
history_dict = history.history

plt.figure(figsize=(12, 5))

# Plot loss
plt.subplot(1, 2, 1) # Create subplot for loss
plt.plot(history_dict['loss'], label='Train Loss')
plt.plot(history_dict['val_loss'], label='Validation Loss')
plt.title('CNN Training History: Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# Plot accuracy
plt.subplot(1, 2, 2) # Create subplot for accuracy
plt.plot(history_dict['accuracy'], label='Train Accuracy')
plt.plot(history_dict['val_accuracy'], label='Validation Accuracy')
plt.title('CNN Training History: Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.tight_layout() # Adjust layout
plt.show()


# Create CNN feature extractor model
# Ensure we use the model with the best weights if early stopping triggered
cnn_feat_extractor = Model(inputs=cnn_cls.input, outputs=cnn_cls.get_layer('feat').output)
print("CNN feature extractor model created from trained weights.")


# ———— 特征提取 & 随机森林分类 ————
print("\nExtracting features with trained CNN...")
X_train_feat = cnn_feat_extractor.predict(X_train, batch_size=BATCH_SIZE) # Use defined batch size
X_test_feat  = cnn_feat_extractor.predict(X_test,  batch_size=BATCH_SIZE) # Use defined batch size
print(f"Extracted feature shapes: X_train_feat={X_train_feat.shape}, X_test_feat={X_test_feat.shape}")

print("\nTraining Random Forest classifier...")
rf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_SEED, n_jobs=-1, class_weight='balanced')
rf.fit(X_train_feat, y_train)
print("Random Forest training finished.")

print("\nPredicting on test set...")
y_pred = rf.predict(X_test_feat)

# ———— 评估 ————
print("\n=== Evaluation on Test Set ===")
print("Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=target_names))


# ———— 可视化：混淆矩阵 ————
print("\nVisualizing Confusion Matrix...")
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=target_names, yticklabels=target_names)
plt.title('CNN+RF混淆矩阵')
plt.xlabel('预测标签')
plt.ylabel('真实标签')
plt.tight_layout()
plt.show()

print("\nScript finished.")