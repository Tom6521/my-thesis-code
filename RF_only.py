import wfdb
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.utils.class_weight import compute_class_weight # 如果使用RF的 class_weight='balanced'，则非必需
import neurokit2 as nk
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)
warnings.filterwarnings("ignore", message="Mean of empty slice")

database_dir = 'D:/Chrome/mit/'
record_names = ['slp01a', 'slp01b', 'slp02a', 'slp02b', 'slp03', 'slp04', 'slp14', 'slp16', 'slp32', 'slp37', 'slp41', 'slp45', 'slp48', 'slp59', 'slp60', 'slp61', 'slp66', 'slp67x']
segment_length = 7500
target_channel_index = 0 # ECG 通道

# 标签映射
new_label_map = { 'W': 0, '1': 1, '2': 1, '3': 2, '4': 2, 'R': 3 }
new_target_names = ['Awake', 'Light', 'Deep', 'REM']
num_classes = len(new_target_names)

#  包含 R 波峰值幅度的特征列表 ---
feature_list = [
'HRV_MeanNN', 'HRV_SDNN', 'HRV_RMSSD', 'HRV_pNN50',
'ECG_Rate_Mean',
'Signal_Mean', 'Signal_Std',
'R_Amplitude_Mean', 'R_Amplitude_Std'
]
num_features = len(feature_list)
#--- 2. 数据加载和特征提取循环 ---
all_X_features = []
all_y_labels = []

print(f"Starting data loading and feature extraction ({num_classes} Classes - Time Domain + R-Amp Features)...")
print(f"Extracting features: {feature_list}")

for record_name in record_names:
    record_path_base = os.path.join(database_dir, record_name)
    print(f"\nProcessing record: {record_name} at {record_path_base}")
    try:
        header = wfdb.rdheader(record_path_base)
        fs = header.fs
        record = wfdb.rdrecord(record_path_base)
        signal_data = record.p_signal
        if signal_data.ndim == 1: signal_data = signal_data.reshape(-1, 1)
        ann_st = wfdb.rdann(record_path_base, 'st')
        annotation_samples = ann_st.sample
        labels_mapped = [new_label_map.get(aux.strip(), -1) for aux in ann_st.aux_note]

        record_X_clean_features = []
        record_y_clean = []
        processed_segments = 0
        skipped_boundary, skipped_peaks, skipped_nan, skipped_hrv_error, skipped_r_amp_error = 0, 0, 0, 0, 0

        # print(f"  Processing segments for {record_name}...")
        for i, label in enumerate(labels_mapped):
             if label != -1:
                start_sample = annotation_samples[i]
                end_sample = start_sample + segment_length

                if end_sample <= signal_data.shape[0]:
                    segment = signal_data[start_sample:end_sample, target_channel_index] # 获取一维片段

                    # 初始化特征为 NaN
                    r_amp_mean, r_amp_std = np.nan, np.nan
                    mean_rate, signal_mean, signal_std = np.nan, np.nan, np.nan
                    hrv_values = {fname: np.nan for fname in feature_list if 'HRV_' in fname}

                    try:
                        # 1. R 波峰值
                        _, rpeaks_info = nk.ecg_peaks(segment, sampling_rate=fs, method='pantompkins1985', correct_artifacts=True)
                        rpeaks_indices = rpeaks_info['ECG_R_Peaks']

                        if len(rpeaks_indices) < 4:
                            skipped_peaks += 1
                            continue

                        # 2. HRV 和心率
                        hrv_features_df = nk.hrv(rpeaks_indices, sampling_rate=fs, show=False)
                        for fname in hrv_values.keys():
                             if fname in hrv_features_df.columns:
                                 hrv_values[fname] = hrv_features_df[fname].iloc[0]

                        if not pd.isna(hrv_values.get('HRV_MeanNN')) and hrv_values.get('HRV_MeanNN') != 0:
                             mean_rate = 60000.0 / hrv_values['HRV_MeanNN']

                        # 3. 信号统计量
                        signal_mean=np.mean(segment)
                        signal_std=np.std(segment)

                        try:
                            valid_rpeaks = rpeaks_indices[(rpeaks_indices >= 0) & (rpeaks_indices < len(segment))].astype(int)

                            if len(valid_rpeaks) >= 1:
                                r_amplitudes = segment[valid_rpeaks]
                                r_amp_mean = np.mean(r_amplitudes)
                                if len(valid_rpeaks) > 1:
                                    r_amp_std = np.std(r_amplitudes)
                                else:
                                    r_amp_std = 0

                        except Exception as r_amp_e:
                             skipped_r_amp_error += 1
                        #--- 新增结束 ---

                        # 5. 组装特征向量
                        feature_vector = []
                        possible_nan = False
                        for feature_name in feature_list:
                            feature_value = np.nan
                            if feature_name in hrv_values:
                                feature_value = hrv_values[feature_name]
                            elif feature_name == 'ECG_Rate_Mean':
                                feature_value = mean_rate
                            elif feature_name == 'Signal_Mean':
                                feature_value = signal_mean
                            elif feature_name == 'Signal_Std':
                                feature_value = signal_std
                            elif feature_name == 'R_Amplitude_Mean':
                                feature_value = r_amp_mean
                            elif feature_name == 'R_Amplitude_Std':
                                feature_value = r_amp_std

                            feature_vector.append(feature_value)
                            if pd.isna(feature_value): possible_nan = True

                        # 6. 检查 NaN 值
                        if possible_nan and pd.isna(feature_vector).any():
                            skipped_nan += 1
                            continue

                        # 添加特征和标签
                        record_X_clean_features.append(feature_vector)
                        record_y_clean.append(label)
                        processed_segments += 1

                    except Exception as outer_e:
                        skipped_hrv_error += 1
                        continue
                else:
                    skipped_boundary += 1 # 片段边界错误

        #--- 记录摘要 ---
        if record_X_clean_features:
             all_X_features.extend(record_X_clean_features)
             all_y_labels.extend(record_y_clean)
             print(f"  -> 完成记录 {record_name}。处理了 {processed_segments} 个片段。")
        else:
            print(f"  -> 记录 {record_name} 未处理任何有效片段。")

    except Exception as e:
        print(f"  处理记录 {record_name} 时出错: {e}")

print("\n数据加载和特征提取完成。")

#--- 3. 合并和最终确定数据 ---
if not all_X_features:
    raise ValueError("未提取任何特征。")

X_combined_features = np.array(all_X_features, dtype=np.float32)
y_combined = np.array(all_y_labels, dtype=np.int32)

print(f"\n总共有特征的片段数量: {X_combined_features.shape[0]}") # 形状应为 (N, 9)
print(f"合并后的特征形状 (X): {X_combined_features.shape}")
print(f"合并后的标签形状 (y): {y_combined.shape}")

# 检查/填充 NaNs/Infs
if np.isnan(X_combined_features).any() or np.isinf(X_combined_features).any():
    print("警告: 正在填充 NaNs/Infs...")
    col_mean = np.nanmean(np.where(np.isinf(X_combined_features), np.nan, X_combined_features), axis=0)
    inds = np.where(np.isnan(X_combined_features) | np.isinf(X_combined_features))

    if np.isnan(col_mean).any():
        raise ValueError("NaN 填充失败，某些列全为 NaN/Inf。")

    # 用相应的列平均值填充 NaN/Inf
    X_combined_features[inds] = np.take(col_mean, inds[1])
    print("NaNs/Infs 填充完毕。")


unique_labels, counts = np.unique(y_combined, return_counts=True)
print("合并数据中的标签分布 (4 类):")
label_dist_dict = {}
for label_val, count in zip(unique_labels, counts):
    label_dist_dict[new_target_names[label_val]] = count
print(label_dist_dict)

#--- 4. 分割数据 (特征) ---
print("\n正在分割特征数据...")
if X_combined_features.shape[0] < 2 or len(unique_labels) < 2:
    raise ValueError("数据不足，无法进行训练/验证分割。")

X_train_features, X_val_features, y_train, y_val = train_test_split(
    X_combined_features, y_combined, test_size=0.2, random_state=42, stratify=y_combined)

print(f"训练集特征形状: {X_train_features.shape}, 标签形状: {y_train.shape}")
print(f"验证集特征形状: {X_val_features.shape}, 标签形状: {y_val.shape}")

#--- 5. 特征标准化 ---
print("\n正在标准化特征...")
scaler = StandardScaler()
X_train_features_scaled = scaler.fit_transform(X_train_features)
X_val_features_scaled = scaler.transform(X_val_features)

#--- 6. 定义和训练 RandomForest 模型 ---
print("\n正在定义和训练 RandomForest 模型 (4 类 - 包含 R 波幅度特征)...")
model = RandomForestClassifier(
    n_estimators=150, max_depth=15, min_samples_leaf=5,
    random_state=42, class_weight='balanced', n_jobs=-1) # class_weight='balanced' 自动处理类别不平衡
model.fit(X_train_features_scaled, y_train)
print("模型训练完成。")

#--- 7. 评估模型 ---
print("\n正在评估模型在验证集上的性能...")
y_pred = model.predict(X_val_features_scaled)

val_acc = accuracy_score(y_val, y_pred)
print(f"验证集准确率: {val_acc:.4f}")

#--- 8. 混淆矩阵和分类报告 ---
print("\n正在生成性能指标 (4 类)...")

cm = confusion_matrix(y_val, y_pred)
print("\n混淆矩阵:")
print(cm)

report = classification_report(y_val, y_pred, target_names=new_target_names, zero_division=0)
print("\n分类报告:")
print(report)

# 绘制混淆矩阵
plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=new_target_names, yticklabels=new_target_names)
plt.title('混淆矩阵 (4 类 - 包含 R 波幅度特征)')
plt.ylabel('真实标签')
plt.xlabel('预测标签')
plt.show()

#--- 9. 特征重要性 ---
try:
    importances = model.feature_importances_
    num_actual_features = X_train_features_scaled.shape[1]
    indices = np.argsort(importances)[::-1]

    print("\n特征重要性排名 (包含 R 波幅度特征):")
    feature_names_actual = feature_list[:num_actual_features]
    if len(feature_names_actual) != num_actual_features:
         feature_names_sorted = [f"Idx {j}" for j in indices]
    else:
         feature_names_sorted = [feature_names_actual[j] for j in indices]

    for i in range(num_actual_features):
        print(f"{i + 1}. 特征 '{feature_names_sorted[i]}' ({importances[indices[i]]:.4f})")

    # 绘制特征重要性条形图
    plt.figure(figsize=(11, 7))
    plt.title("特征重要性")
    plt.bar(range(num_actual_features), importances[indices], align='center')
    plt.xticks(range(num_actual_features), feature_names_sorted, rotation=90)
    plt.xlim([-1, num_actual_features])
    plt.tight_layout()
    plt.show()

except Exception as e:
    print(f"\n无法生成特征重要性: {e}")

print("\n脚本执行完毕。")
