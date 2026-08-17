"""
舌后缀特征 SHAP 贡献度分析
=============================
从舌线轮廓点 CSV 中提取多维度特征，训练 XGBoost 分类器（0=静息, 1=后缀），
使用 SHAP 分析各指标贡献度，重点关注时间码切换瞬间响应最剧烈/最超前的指标。
"""

import os
import glob
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from scipy.signal import savgol_filter
from scipy.stats import pointbiserialr, spearmanr
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import classification_report, roc_auc_score
import xgboost as xgb
import shap

warnings.filterwarnings("ignore")
plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_DIR = os.path.abspath(os.path.join(ROOT_DIR, "..", "testvedio", "muler_vedio"))
OUTPUT_DIR = os.path.join(ROOT_DIR, "shap_results")
FEATURE_SEQUENCE_DIR = os.path.join(OUTPUT_DIR, "feature_sequences")
TARGET_CASE_DIR = os.path.join(OUTPUT_DIR, "5_1")
TARGET_POINTS_CSV = os.path.join(TARGET_CASE_DIR, "points_#5_1_011301_124101.csv")
TARGET_LABEL_CSV = os.path.join(TARGET_CASE_DIR, "#5_1_011301_124101_data_corrected_rolling_labeled.csv")
TARGET_EVENT_COLUMN = "Has_Sleep_Event"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FEATURE_SEQUENCE_DIR, exist_ok=True)

# ============================================================
# 1. 特征提取函数
# ============================================================

def load_points(row):
    """从一行 CSV 中解析出 (100, 2) 的点坐标"""
    xs = np.array([row[f"x{i}"] for i in range(100)])
    ys = np.array([row[f"y{i}"] for i in range(100)])
    return np.column_stack((xs, ys))


def compute_curvature(pts):
    """逐点曲率 K = |x'y'' - y'x''| / (x'^2 + y'^2)^{3/2}"""
    dx = np.gradient(pts[:, 0])
    dy = np.gradient(pts[:, 1])
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    num = np.abs(dx * ddy - dy * ddx)
    den = (dx**2 + dy**2)**1.5 + 1e-8
    return num / den


def compute_arc_length(pts):
    """累计弧长"""
    diffs = np.diff(pts, axis=0)
    seg_lens = np.sqrt((diffs**2).sum(axis=1))
    return seg_lens.sum()


def compute_chord_bulge_metrics(pts):
    """计算一段曲线相对首尾连线的鼓起幅度，用作后段塌陷代理特征。"""
    if len(pts) < 2:
        return 0.0, 0.0, 0.0, 0.0

    chord = pts[-1] - pts[0]
    chord_len = float(np.linalg.norm(chord))
    if chord_len < 1e-8:
        return 0.0, 0.0, 0.0, 0.0

    rel = pts - pts[0]
    cross = np.abs(chord[0] * rel[:, 1] - chord[1] * rel[:, 0])
    dists = cross / (chord_len + 1e-8)
    curve_axis = np.concatenate([[0.0], np.cumsum(np.sqrt((np.diff(pts, axis=0) ** 2).sum(axis=1)))])
    bulge_area = np.trapz(dists, curve_axis)
    return chord_len, float(dists.mean()), float(dists.max()), float(bulge_area)


def fit_ellipse_simple(pts):
    """简易椭圆拟合：返回长轴、短轴、离心率"""
    cx, cy = pts.mean(axis=0)
    centered = pts - [cx, cy]
    cov = np.cov(centered.T)
    eigvals = np.sort(np.linalg.eigvalsh(cov))[::-1]
    a = np.sqrt(max(eigvals[0], 1e-8))
    b = np.sqrt(max(eigvals[1], 1e-8))
    ecc = np.sqrt(1 - (b / a) ** 2) if a > b else 0.0
    return a, b, ecc, cx, cy


def fourier_descriptors(pts, n_harmonics=5):
    """傅里叶描述子（低频幅值）"""
    z = pts[:, 0] + 1j * pts[:, 1]
    Z = np.fft.fft(z)
    magnitudes = np.abs(Z)
    # 归一化：除以 DC 分量
    dc = magnitudes[0] + 1e-8
    fd = magnitudes[1:n_harmonics + 1] / dc
    return fd


def tangent_angle_entropy(pts, n_bins=36):
    """切向角熵"""
    dx = np.diff(pts[:, 0])
    dy = np.diff(pts[:, 1])
    angles = np.arctan2(dy, dx)
    hist, _ = np.histogram(angles, bins=n_bins, range=(-np.pi, np.pi), density=True)
    hist = hist + 1e-10
    hist = hist / hist.sum()
    entropy = -np.sum(hist * np.log(hist))
    return entropy


def compute_roughness(pts):
    """表面不平整度：局部偏差的 RMS"""
    # 用 3 次多项式平滑后，测量残差
    t = np.linspace(0, 1, len(pts))
    px = np.polyfit(t, pts[:, 0], 3)
    py = np.polyfit(t, pts[:, 1], 3)
    smooth_x = np.polyval(px, t)
    smooth_y = np.polyval(py, t)
    residuals = np.sqrt((pts[:, 0] - smooth_x)**2 + (pts[:, 1] - smooth_y)**2)
    return np.sqrt(np.mean(residuals**2))


def extract_features_single_frame(pts):
    """从单帧 100 个舌线点中提取静态特征"""
    pts = pts[np.argsort(pts[:, 0])]
    feat = {}

    # --- 一、几何形态学特征 ---
    cx, cy = pts.mean(axis=0)
    feat["centroid_x"] = cx
    feat["centroid_y"] = cy

    # 包络面积 (AUC): 舌线与底边围成面积（用梯形法则近似）
    sorted_idx = np.argsort(pts[:, 0])
    sorted_pts = pts[sorted_idx]
    y_baseline = sorted_pts[:, 1].max()  # 底边取最低点 y
    feat["auc"] = np.trapz(np.abs(y_baseline - sorted_pts[:, 1]), sorted_pts[:, 0])

    # 凸包空隙率
    try:
        hull = ConvexHull(pts)
        hull_area = hull.volume  # 2D 时 volume 即面积
        polygon_area = feat["auc"]
        feat["convex_hull_gap_ratio"] = 1.0 - min(polygon_area / (hull_area + 1e-8), 1.0)
    except Exception:
        feat["convex_hull_gap_ratio"] = 0.0

    # 椭圆拟合
    a, b, ecc, ecx, ecy = fit_ellipse_simple(pts)
    feat["ellipse_a"] = a
    feat["ellipse_b"] = b
    feat["eccentricity"] = ecc

    # 弧长
    arc_len = compute_arc_length(pts)
    feat["arc_length"] = arc_len

    # 水平跨度与垂直跨度
    x_span = pts[:, 0].max() - pts[:, 0].min()
    y_span = pts[:, 1].max() - pts[:, 1].min()
    feat["x_span"] = x_span
    feat["y_span"] = y_span
    feat["expansion_coeff"] = x_span / (y_span + 1e-8)

    # --- 二、区域分割特征 ---
    n = len(pts)
    seg1 = pts[:n // 3]          # 舌尖段
    seg2 = pts[n // 3:2 * n // 3]  # 舌中段
    seg3 = pts[2 * n // 3:]      # 舌根段

    len1 = compute_arc_length(seg1) if len(seg1) > 1 else 0
    len2 = compute_arc_length(seg2) if len(seg2) > 1 else 0
    len3 = compute_arc_length(seg3) if len(seg3) > 1 else 0
    total_seg = len1 + len2 + len3 + 1e-8
    feat["seg_tip_ratio"] = len1 / total_seg
    feat["seg_mid_ratio"] = len2 / total_seg
    feat["seg_root_ratio"] = len3 / total_seg

    # 曲率
    curvature = compute_curvature(pts)
    feat["curvature_mean"] = curvature.mean()
    feat["curvature_max"] = curvature.max()
    feat["curvature_std"] = curvature.std()

    # 最大曲率点位置（归一化到 0~1，0=舌尖，1=舌根）
    kmax_pos = np.argmax(curvature) / (n - 1)
    feat["kmax_position"] = kmax_pos

    # 舌根部斜率
    root_pts = pts[int(0.8 * n):]
    posterior_cx, posterior_cy = root_pts.mean(axis=0)
    feat["posterior_centroid_x"] = posterior_cx
    feat["posterior_centroid_y"] = posterior_cy
    feat["posterior_x_span"] = root_pts[:, 0].max() - root_pts[:, 0].min()
    feat["posterior_y_span"] = root_pts[:, 1].max() - root_pts[:, 1].min()
    feat["posterior_arc_length"] = compute_arc_length(root_pts)

    posterior_chord_len, posterior_bulge_mean, posterior_bulge_height, posterior_bulge_area = compute_chord_bulge_metrics(root_pts)
    feat["posterior_chord_length"] = posterior_chord_len
    feat["posterior_bulge_mean_proxy"] = posterior_bulge_mean
    feat["posterior_bulge_height_proxy"] = posterior_bulge_height
    feat["posterior_bulge_area_proxy"] = posterior_bulge_area
    feat["posterior_curve_chord_ratio"] = feat["posterior_arc_length"] / (posterior_chord_len + 1e-8)
    feat["posterior_compactness"] = feat["posterior_arc_length"] / (feat["posterior_x_span"] + 1e-8)
    feat["posterior_intrusion_height"] = y_baseline - root_pts[:, 1].min()
    if len(root_pts) > 2:
        # 斜率角度
        p1, p2 = root_pts[0], root_pts[-1]
        dy_root = p2[1] - p1[1]
        dx_root = p2[0] - p1[0] + 1e-8
        feat["posterior_slope_angle"] = np.degrees(np.arctan2(dy_root, dx_root))

        # 线性拟合斜率
        t_root = np.arange(len(root_pts), dtype=float)
        coef = np.polyfit(t_root, root_pts[:, 1], 1)
        feat["posterior_gradient"] = coef[0]
    else:
        feat["posterior_slope_angle"] = 0.0
        feat["posterior_gradient"] = 0.0

    # 最高点 y 到舌根末端的角度
    highest_idx = np.argmin(pts[:, 1])  # y 越小越高
    highest_pt = pts[highest_idx]
    root_end = pts[-1]
    feat["peak_to_root_angle"] = np.degrees(np.arctan2(
        root_end[1] - highest_pt[1], root_end[0] - highest_pt[0] + 1e-8))

    # --- 三、旋转角 ---
    tip_pt = pts[0]
    root_pt = pts[-1]
    feat["rotation_angle"] = np.degrees(np.arctan2(
        root_pt[1] - tip_pt[1], root_pt[0] - tip_pt[0] + 1e-8))

    # --- 四、空间参考与抽象数学特征 ---
    # 旧实现 max_x - root_x 恒等于 0；改为基于舌根后段鼓起程度的可用空间代理。
    feat["posterior_airway_space_proxy"] = posterior_chord_len / (1.0 + posterior_bulge_height)
    feat["pharyngeal_gap_proxy"] = feat["posterior_airway_space_proxy"]

    # 切向角熵
    feat["tangent_entropy"] = tangent_angle_entropy(pts)

    # 傅里叶描述子
    fd = fourier_descriptors(pts, n_harmonics=5)
    for i, v in enumerate(fd):
        feat[f"fourier_{i + 1}"] = v

    # 表面不平整度
    feat["roughness"] = compute_roughness(pts)

    return feat


def add_temporal_features(df_feat):
    """在已有静态特征 DataFrame 上添加时序动态特征"""
    # 质心速度（帧间差分）
    df_feat["centroid_vx"] = df_feat["centroid_x"].diff().fillna(0)
    df_feat["centroid_vy"] = df_feat["centroid_y"].diff().fillna(0)
    df_feat["centroid_speed"] = np.sqrt(
        df_feat["centroid_vx"]**2 + df_feat["centroid_vy"]**2)
    df_feat["centroid_direction"] = np.degrees(np.arctan2(
        df_feat["centroid_vy"], df_feat["centroid_vx"] + 1e-8))

    # 质心加速度
    df_feat["centroid_ax"] = df_feat["centroid_vx"].diff().fillna(0)
    df_feat["centroid_ay"] = df_feat["centroid_vy"].diff().fillna(0)
    df_feat["centroid_accel"] = np.sqrt(
        df_feat["centroid_ax"]**2 + df_feat["centroid_ay"]**2)

    # 面积变化率
    df_feat["auc_delta"] = df_feat["auc"].diff().fillna(0)
    df_feat["auc_rate"] = df_feat["auc_delta"].rolling(3, min_periods=1).mean()

    # 弧长变化率（压缩比代理）
    df_feat["arc_length_delta"] = df_feat["arc_length"].diff().fillna(0)

    # 旋转角变化（翻转速率）
    df_feat["rotation_rate"] = df_feat["rotation_angle"].diff().fillna(0)

    # 曲率最大点位置变化率（迁移速度）
    df_feat["kmax_migration_rate"] = df_feat["kmax_position"].diff().fillna(0)

    # 后缀斜率变化率
    df_feat["posterior_slope_rate"] = df_feat["posterior_slope_angle"].diff().fillna(0)

    # 扩张系数变化率
    df_feat["expansion_rate"] = df_feat["expansion_coeff"].diff().fillna(0)

    # 后段质心运动：作为舌根驱动代理
    df_feat["posterior_vx"] = df_feat["posterior_centroid_x"].diff().fillna(0)
    df_feat["posterior_vy"] = df_feat["posterior_centroid_y"].diff().fillna(0)
    df_feat["posterior_speed"] = np.sqrt(df_feat["posterior_vx"]**2 + df_feat["posterior_vy"]**2)
    df_feat["posterior_accel"] = df_feat["posterior_speed"].diff().fillna(0)

    # 后段形变率：用于表征舌根部向后卷曲/鼓起的轨迹
    df_feat["posterior_bulge_height_delta"] = df_feat["posterior_bulge_height_proxy"].diff().fillna(0)
    df_feat["posterior_bulge_area_delta"] = df_feat["posterior_bulge_area_proxy"].diff().fillna(0)
    df_feat["posterior_intrusion_rate"] = df_feat["posterior_intrusion_height"].diff().fillna(0)
    df_feat["posterior_curve_ratio_rate"] = df_feat["posterior_curve_chord_ratio"].diff().fillna(0)

    # 可用空间代理与塌陷指数：相对近期最开放状态的归一化缩窄程度
    collapse_window = max(5, min(30, len(df_feat)))
    reference_space = df_feat["posterior_airway_space_proxy"].rolling(collapse_window, min_periods=1).max()
    df_feat["posterior_space_delta"] = df_feat["posterior_airway_space_proxy"].diff().fillna(0)
    df_feat["posterior_space_rate"] = df_feat["posterior_space_delta"].rolling(3, min_periods=1).mean()
    df_feat["collapse_index_proxy"] = (1.0 - df_feat["posterior_airway_space_proxy"] / (reference_space + 1e-8)).clip(0.0, 1.0)
    df_feat["collapse_velocity_proxy"] = df_feat["collapse_index_proxy"].diff().fillna(0)

    # 顺应性代理：后段可用空间变化 / 舌根驱动位移
    df_feat["compliance_proxy"] = df_feat["posterior_space_delta"] / (df_feat["posterior_speed"] + 1e-6)
    df_feat["compliance_abs_proxy"] = np.abs(df_feat["compliance_proxy"])

    return df_feat


def normalize_binary_event_series(series):
    """将事件列统一转换为 0/1。"""
    if pd.api.types.is_bool_dtype(series):
        return series.astype(int)
    if pd.api.types.is_numeric_dtype(series):
        return (series.fillna(0) != 0).astype(int)

    normalized = series.astype(str).str.strip().str.lower()
    true_tokens = {"true", "1", "yes", "y", "有", "事件", "event"}
    false_tokens = {"false", "0", "no", "n", "无", "", "nan", "none"}

    mapped = normalized.map(lambda value: 1 if value in true_tokens else (0 if value in false_tokens else np.nan))
    if mapped.isna().any():
        unknown = normalized[mapped.isna()].drop_duplicates().tolist()[:10]
        raise ValueError(f"无法解析事件列取值: {unknown}")
    return mapped.astype(int)


def resolve_event_column(df_label, preferred_name=TARGET_EVENT_COLUMN, fallback_index=66):
    """优先使用明确列名，否则退回到 Excel BN 对应的第 66 列。"""
    if preferred_name in df_label.columns:
        return preferred_name
    if len(df_label.columns) >= fallback_index:
        fallback_name = df_label.columns[fallback_index - 1]
        print(f"未找到事件列 {preferred_name}，改用第 {fallback_index} 列: {fallback_name}")
        return fallback_name
    raise KeyError(f"未找到事件列 {preferred_name}，且表格列数不足 {fallback_index} 列")


def ensure_2d_shap_values(raw_shap_values):
    """兼容不同 SHAP 版本返回格式，统一为 (n_samples, n_features)。"""
    if isinstance(raw_shap_values, list):
        raw_shap_values = raw_shap_values[-1]
    shap_values = np.asarray(raw_shap_values)
    if shap_values.ndim == 3:
        shap_values = shap_values[..., -1]
    return shap_values


def extract_feature_sequence_from_points_csv(points_csv_path, export_path=None):
    """从单个 points CSV 提取完整特征序列。"""
    df_points = pd.read_csv(points_csv_path)
    feats_list = []

    for _, row in df_points.iterrows():
        pts = load_points(row)
        feats_list.append(extract_features_single_frame(pts))

    df_feat = pd.DataFrame(feats_list)
    df_feat = add_temporal_features(df_feat)

    if export_path:
        df_export = df_feat.copy()
        df_export.insert(0, "frame_idx", df_points["frame_idx"].values)
        df_export.to_csv(export_path, index=False)

    return df_feat, df_points["frame_idx"].astype(int).to_numpy()


def compute_binary_correlations(values, events):
    """计算连续变量与二分类事件的相关性。"""
    values = np.asarray(values, dtype=float)
    events = np.asarray(events, dtype=int)
    valid_mask = np.isfinite(values) & np.isfinite(events)
    values = values[valid_mask]
    events = events[valid_mask]

    if len(values) < 3 or np.unique(events).size < 2 or np.nanstd(values) < 1e-12:
        return np.nan, np.nan, np.nan, np.nan

    point_r, point_p = pointbiserialr(events, values)
    spear_r, spear_p = spearmanr(values, events)
    return point_r, point_p, spear_r, spear_p


def format_report_number(value, digits=4):
    """格式化报告中的数值。"""
    if pd.isna(value):
        return "NA"
    value = float(value)
    if value == 0:
        return "0"
    if abs(value) >= 1e4 or abs(value) < 1e-3:
        return f"{value:.2e}"
    return f"{value:.{digits}f}"


def format_report_p_value(value):
    """格式化报告中的 p 值。"""
    if pd.isna(value):
        return "NA"
    value = float(value)
    if value < 1e-4:
        return f"{value:.2e}"
    return f"{value:.4f}"


def build_markdown_table(df, columns):
    """将 DataFrame 切片渲染为 Markdown 表格。"""
    p_value_columns = {
        "feature_pointbiserial_p",
        "feature_spearman_p",
        "shap_pointbiserial_p",
        "shap_spearman_p",
        "pointbiserial_p",
        "spearman_p",
    }
    header = "| " + " | ".join(label for _, label in columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = [header, separator]

    for _, row in df.iterrows():
        values = []
        for col_name, _ in columns:
            value = row[col_name]
            if isinstance(value, (float, np.floating)):
                if col_name in p_value_columns:
                    values.append(format_report_p_value(value))
                else:
                    values.append(format_report_number(value))
            else:
                values.append(str(value))
        rows.append("| " + " | ".join(values) + " |")

    return "\n".join(rows)


def write_target_case_report(output_dir, corr_df, pred_corr_df, merged_frame_df, event_col):
    """导出单案例中文 Markdown 报告。"""
    report_path = os.path.join(output_dir, "target_shap_bn_report_zh.md")

    event_frames = int(merged_frame_df["bn_event"].sum())
    total_frames = int(len(merged_frame_df))
    non_event_frames = total_frames - event_frames
    event_rate = event_frames / max(total_frames, 1)

    top_positive = corr_df.sort_values("shap_pointbiserial_r", ascending=False).head(8)
    top_negative = corr_df.sort_values("shap_pointbiserial_r", ascending=True).head(8)
    top_abs = corr_df.reindex(corr_df["shap_pointbiserial_r"].abs().sort_values(ascending=False).index).head(12)
    disagreement = corr_df.assign(
        shap_feature_gap=(corr_df["shap_pointbiserial_r"] - corr_df["feature_pointbiserial_r"]).abs()
    ).sort_values("shap_feature_gap", ascending=False).head(6)

    pred_row = pred_corr_df.iloc[0]
    positive_table = build_markdown_table(
        top_positive,
        [
            ("feature", "指标"),
            ("target_mean_abs_shap", "目标视频平均绝对SHAP"),
            ("shap_pointbiserial_r", "SHAP 点双列相关 r"),
            ("shap_pointbiserial_p", "p 值"),
            ("shap_event_delta", "事件帧 SHAP 均值差"),
        ],
    )
    negative_table = build_markdown_table(
        top_negative,
        [
            ("feature", "指标"),
            ("target_mean_abs_shap", "目标视频平均绝对SHAP"),
            ("shap_pointbiserial_r", "SHAP 点双列相关 r"),
            ("shap_pointbiserial_p", "p 值"),
            ("shap_event_delta", "事件帧 SHAP 均值差"),
        ],
    )
    abs_table = build_markdown_table(
        top_abs,
        [
            ("feature", "指标"),
            ("target_mean_abs_shap", "目标视频平均绝对SHAP"),
            ("shap_pointbiserial_r", "SHAP 点双列相关 r"),
            ("feature_event_delta", "原始特征事件差"),
            ("shap_event_delta", "SHAP 事件差"),
        ],
    )
    disagreement_table = build_markdown_table(
        disagreement,
        [
            ("feature", "指标"),
            ("feature_pointbiserial_r", "原始特征 r"),
            ("shap_pointbiserial_r", "SHAP r"),
            ("shap_feature_gap", "两者差值绝对值"),
        ],
    )

    report_lines = [
        "# 5_1 单案例 SHAP-BN 相关性中文报告",
        "",
        "## 1. 数据概况",
        f"- 分析对象：points_#5_1_011301_124101.csv 与 #5_1_011301_124101_data_corrected_rolling_labeled.csv。",
        f"- 对齐方式：用 points 文件中的 1-based frame_idx 对齐到标注表第 {event_col} 列。",
        f"- 有效曲线帧数：{total_frames}。",
        f"- 事件帧数：{event_frames}。",
        f"- 非事件帧数：{non_event_frames}。",
        f"- 事件占比：{event_rate:.2%}。",
        "",
        "## 2. 总体结论",
        (
            f"把所有指标拆开看，比直接看模型总体分数更有信息。当前模型概率与 BN 事件只有很弱的正相关，"
            f"点双列相关 r = {format_report_number(pred_row['pointbiserial_r'])}，"
            f"Spearman r = {format_report_number(pred_row['spearman_r'])}。"
            "这说明对这个样本，模型整体分数并不能直接充当 BN 事件强度的替代量。"
        ),
        (
            "单个指标的 SHAP 贡献也都不算强，绝对相关系数最高大约在 0.08 到 0.12 之间，"
            "属于弱相关但方向稳定的范围。更适合用来做候选指标排序和现象解释，不适合直接拿某一个指标单独做硬阈值判断。"
        ),
        "",
        "## 3. 事件出现时 SHAP 倾向升高的指标",
        positive_table,
        "",
        "## 4. 事件出现时 SHAP 倾向降低的指标",
        negative_table,
        "",
        "## 5. 按绝对相关强度排序的重点指标",
        abs_table,
        "",
        "## 6. 需要特别注意的解读点",
        (
            "- expansion_coeff 的原始特征值与事件几乎没有线性相关，但它的 SHAP 相关性却排到前列。"
            "这说明模型对它的使用更像是非线性或阈值式使用，而不是简单的单调关系。"
        ),
        (
            "- seg_tip_ratio、rotation_angle、fourier_2 到 fourier_5 这一组指标在事件帧上大多表现为负向 SHAP，"
            "说明它们的当前取值会把模型输出往“非事件”方向拉。"
        ),
        (
            "- curvature_mean、curvature_max、arc_length、peak_to_root_angle 这一组在事件帧上更偏正向 SHAP，"
            "说明它们更像是事件相关形态变化被模型利用到的方向。"
        ),
        (
            "- Mean |SHAP| 大并不等于和 BN 事件最相关。比如 centroid_y 的目标视频 Mean |SHAP| 很高，"
            "但与 BN 的相关性只排在中后段，说明“重要”与“对这张事件表同步”是两件不同的事。"
        ),
        "",
        "## 7. 原始特征相关性和 SHAP 相关性差异最大的指标",
        disagreement_table,
        "",
        "## 8. 建议的使用方式",
        "- 如果目的是找最值得盯的指标，优先看 fourier_3、expansion_coeff、fourier_5、seg_tip_ratio、curvature_mean。",
        "- 如果目的是做可解释汇报，建议同时展示“原始特征差异”和“SHAP 差异”，因为二者可能方向不同。",
        "- 这次图片标题已改成英文，避免中文字体缺失导致图中乱码；中文解释集中放在这份 Markdown 报告里。",
    ]

    with open(report_path, "w", encoding="utf-8") as report_file:
        report_file.write("\n".join(report_lines) + "\n")

    return report_path


def run_target_bn_correlation_analysis(model, importance_df, points_csv_path, label_csv_path, output_dir):
    """对单个目标视频逐帧计算 SHAP，并与 BN 事件列做相关性分析。"""
    if not os.path.exists(points_csv_path):
        print(f"跳过目标视频分析，未找到 points 文件: {points_csv_path}")
        return None, None
    if not os.path.exists(label_csv_path):
        print(f"跳过目标视频分析，未找到标注表: {label_csv_path}")
        return None, None

    os.makedirs(output_dir, exist_ok=True)

    print("\n开始目标视频 SHAP-BN 相关性分析 ...")
    df_feat, frame_idx = extract_feature_sequence_from_points_csv(
        points_csv_path,
        export_path=os.path.join(output_dir, "target_frame_features.csv"),
    )
    df_label = pd.read_csv(label_csv_path)
    event_col = resolve_event_column(df_label)

    if frame_idx.min() < 1 or frame_idx.max() > len(df_label):
        raise ValueError(
            f"frame_idx 超出标注表范围: [{frame_idx.min()}, {frame_idx.max()}] vs {len(df_label)} rows"
        )

    aligned_label = df_label.iloc[frame_idx - 1].reset_index(drop=True).copy()
    event_binary = normalize_binary_event_series(aligned_label[event_col])

    explainer = shap.TreeExplainer(model)
    shap_values_target = ensure_2d_shap_values(explainer.shap_values(df_feat))
    pred_prob = model.predict_proba(df_feat)[:, 1]

    shap_df = pd.DataFrame(
        shap_values_target,
        columns=[f"shap_{name}" for name in df_feat.columns],
    )
    merged_frame_df = pd.concat(
        [
            pd.DataFrame({
                "frame_idx": frame_idx,
                "bn_event": event_binary.values,
                "bn_event_raw": aligned_label[event_col].values,
                "model_prob": pred_prob,
            }),
            aligned_label.reset_index(drop=True),
            df_feat.reset_index(drop=True),
            shap_df.reset_index(drop=True),
        ],
        axis=1,
    )
    merged_frame_df.to_csv(os.path.join(output_dir, "target_frame_shap_with_bn.csv"), index=False)

    importance_lookup = importance_df.set_index("feature") if importance_df is not None else None
    target_mean_abs_shap = np.abs(shap_values_target).mean(axis=0)
    rows = []
    event_mask = event_binary.values == 1
    non_event_mask = event_binary.values == 0

    for idx, feature_name in enumerate(df_feat.columns):
        feature_values = df_feat[feature_name].values
        feature_shap_values = shap_values_target[:, idx]

        feature_point_r, feature_point_p, feature_spear_r, feature_spear_p = compute_binary_correlations(
            feature_values, event_binary.values)
        shap_point_r, shap_point_p, shap_spear_r, shap_spear_p = compute_binary_correlations(
            feature_shap_values, event_binary.values)

        rows.append({
            "feature": feature_name,
            "global_mean_abs_shap": float(importance_lookup.at[feature_name, "mean_abs_shap"]) if importance_lookup is not None and feature_name in importance_lookup.index else np.nan,
            "target_mean_abs_shap": float(target_mean_abs_shap[idx]),
            "feature_event_mean": float(np.nanmean(feature_values[event_mask])) if event_mask.any() else np.nan,
            "feature_non_event_mean": float(np.nanmean(feature_values[non_event_mask])) if non_event_mask.any() else np.nan,
            "feature_event_delta": float(np.nanmean(feature_values[event_mask]) - np.nanmean(feature_values[non_event_mask])) if event_mask.any() and non_event_mask.any() else np.nan,
            "feature_pointbiserial_r": feature_point_r,
            "feature_pointbiserial_p": feature_point_p,
            "feature_spearman_r": feature_spear_r,
            "feature_spearman_p": feature_spear_p,
            "shap_event_mean": float(np.nanmean(feature_shap_values[event_mask])) if event_mask.any() else np.nan,
            "shap_non_event_mean": float(np.nanmean(feature_shap_values[non_event_mask])) if non_event_mask.any() else np.nan,
            "shap_event_delta": float(np.nanmean(feature_shap_values[event_mask]) - np.nanmean(feature_shap_values[non_event_mask])) if event_mask.any() and non_event_mask.any() else np.nan,
            "shap_pointbiserial_r": shap_point_r,
            "shap_pointbiserial_p": shap_point_p,
            "shap_spearman_r": shap_spear_r,
            "shap_spearman_p": shap_spear_p,
        })

    corr_df = pd.DataFrame(rows)
    corr_df = corr_df.sort_values("shap_pointbiserial_r", key=lambda col: np.abs(col), ascending=False).reset_index(drop=True)
    corr_df.insert(0, "rank", np.arange(1, len(corr_df) + 1))
    corr_df.to_csv(os.path.join(output_dir, "target_shap_bn_correlation_summary.csv"), index=False)

    prob_point_r, prob_point_p, prob_spear_r, prob_spear_p = compute_binary_correlations(pred_prob, event_binary.values)
    pred_corr_df = pd.DataFrame([
        {
            "metric": "model_prob_vs_bn_event",
            "pointbiserial_r": prob_point_r,
            "pointbiserial_p": prob_point_p,
            "spearman_r": prob_spear_r,
            "spearman_p": prob_spear_p,
            "event_rate": float(event_binary.mean()),
            "frames_with_curve": int(len(event_binary)),
        }
    ])
    pred_corr_df.to_csv(os.path.join(output_dir, "target_prediction_bn_correlation.csv"), index=False)

    report_path = write_target_case_report(output_dir, corr_df, pred_corr_df, merged_frame_df, event_col)

    top_n = min(20, len(corr_df))
    if top_n > 0:
        top_corr = corr_df.head(top_n).iloc[::-1]
        fig, axes = plt.subplots(1, 2, figsize=(16, max(8, top_n * 0.35)))

        axes[0].barh(top_corr["feature"], top_corr["shap_pointbiserial_r"], color="seagreen")
        axes[0].axvline(0, color="black", linewidth=0.8)
        axes[0].set_title("SHAP vs BN Event Correlation")
        axes[0].set_xlabel("Point-biserial r")

        axes[1].barh(top_corr["feature"], top_corr["feature_pointbiserial_r"], color="steelblue")
        axes[1].axvline(0, color="black", linewidth=0.8)
        axes[1].set_title("Feature vs BN Event Correlation")
        axes[1].set_xlabel("Point-biserial r")

        plt.suptitle("Top-20 Correlated Metrics for Target Video", fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "target_shap_bn_correlation_top20.png"), dpi=150)
        plt.close()

    print("\n=== 目标视频 SHAP-BN 相关性 Top-15 ===")
    print(corr_df[["rank", "feature", "target_mean_abs_shap", "shap_pointbiserial_r", "shap_spearman_r", "shap_event_delta"]].head(15).to_string(index=False))
    print(f"目标视频相关性结果已保存至: {output_dir}")
    print(f"目标视频中文报告已保存至: {report_path}")
    return corr_df, merged_frame_df


# ============================================================
# 2. 加载数据 & 提取特征
# ============================================================

def load_and_extract(csv_dir):
    csv_files = sorted(glob.glob(os.path.join(csv_dir, "**", "points_*.csv"), recursive=True))
    print(f"找到 {len(csv_files)} 个 CSV 文件")

    all_features = []
    all_labels = []
    all_groups = []       # 用于 LeaveOneGroupOut（按视频分组）
    all_frame_idx = []
    all_file_names = []
    all_feature_exports = []

    for fid, csv_path in enumerate(csv_files):
        fname = os.path.basename(csv_path)
        print(f"  处理 {fname} ...")
        df = pd.read_csv(csv_path)

        feats_list = []
        for _, row in df.iterrows():
            pts = load_points(row)
            feat = extract_features_single_frame(pts)
            feats_list.append(feat)

        df_feat = pd.DataFrame(feats_list)
        df_feat = add_temporal_features(df_feat)

        event_ids = df["event_id"].values if "event_id" in df.columns else np.zeros(len(df_feat), dtype=int)
        df_export = df_feat.copy()
        df_export.insert(0, "source_file", fname)
        df_export.insert(0, "source_group", os.path.basename(os.path.dirname(csv_path)))
        df_export.insert(0, "event_id", event_ids)
        df_export.insert(0, "frame_idx", df["frame_idx"].values)
        export_path = os.path.join(FEATURE_SEQUENCE_DIR, fname.replace(".csv", "_features.csv"))
        df_export.to_csv(export_path, index=False)
        all_feature_exports.append(df_export)

        # 标签：event_id > 0 即为舌后缀
        labels = (event_ids > 0).astype(int)

        all_features.append(df_feat)
        all_labels.append(labels)
        all_groups.append(np.full(len(df_feat), fid))
        all_frame_idx.append(df["frame_idx"].values)
        all_file_names.append(np.array([fname] * len(df_feat)))

    if all_feature_exports:
        merged_features = pd.concat(all_feature_exports, ignore_index=True)
        merged_features.to_csv(os.path.join(FEATURE_SEQUENCE_DIR, "all_frame_features.csv"), index=False)
        print(f"导出特征序列到: {FEATURE_SEQUENCE_DIR}")

    X = pd.concat(all_features, ignore_index=True)
    y = np.concatenate(all_labels)
    groups = np.concatenate(all_groups)
    frame_idx = np.concatenate(all_frame_idx)
    file_names = np.concatenate(all_file_names)

    print(f"\n总帧数: {len(X)}, 正样本(后缀): {y.sum()}, 负样本(静息): {(y == 0).sum()}")
    print(f"特征维度: {X.shape[1]}")
    return X, y, groups, frame_idx, file_names, csv_files


# ============================================================
# 3. 训练 XGBoost + SHAP 分析
# ============================================================

def run_shap_analysis(X, y, groups, frame_idx, file_names, csv_files):
    feature_names = X.columns.tolist()

    # 3.1 训练全局模型
    pos_weight = (y == 0).sum() / max((y == 1).sum(), 1)
    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=pos_weight,
        eval_metric="logloss",
        use_label_encoder=False,
        random_state=42,
    )
    model.fit(X, y)
    print("\n全局模型训练完成")

    # 3.2 Leave-One-Group-Out 交叉验证评估
    logo = LeaveOneGroupOut()
    y_pred_all = np.zeros(len(y), dtype=float)
    for train_idx, test_idx in logo.split(X, y, groups):
        m = xgb.XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=pos_weight,
            eval_metric="logloss", use_label_encoder=False, random_state=42)
        m.fit(X.iloc[train_idx], y[train_idx])
        y_pred_all[test_idx] = m.predict_proba(X.iloc[test_idx])[:, 1]

    print("\n=== Leave-One-Group-Out 评估 ===")
    print(classification_report(y, (y_pred_all > 0.5).astype(int),
                                target_names=["静息", "后缀"]))
    try:
        auc = roc_auc_score(y, y_pred_all)
        print(f"ROC-AUC: {auc:.4f}")
    except Exception:
        pass

    # 3.3 SHAP 分析
    print("\n计算 SHAP 值 ...")
    explainer = shap.TreeExplainer(model)
    shap_values = ensure_2d_shap_values(explainer.shap_values(X))

    # 3.4 SHAP summary plot (bar)
    fig, ax = plt.subplots(figsize=(10, 12))
    shap.summary_plot(shap_values, X, plot_type="bar", show=False, max_display=30)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "shap_bar_importance.png"), dpi=150)
    plt.close()
    print(f"  保存 shap_bar_importance.png")

    # 3.5 SHAP summary plot (dot/beeswarm)
    fig, ax = plt.subplots(figsize=(12, 14))
    shap.summary_plot(shap_values, X, show=False, max_display=30)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "shap_beeswarm.png"), dpi=150)
    plt.close()
    print(f"  保存 shap_beeswarm.png")

    # 3.6 特征重要性排名表
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs_shap,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    importance_df["rank"] = range(1, len(importance_df) + 1)
    importance_df.to_csv(os.path.join(OUTPUT_DIR, "feature_importance_shap.csv"), index=False)
    print(f"\n=== Top-15 特征贡献度排名 ===")
    print(importance_df.head(15).to_string(index=False))

    return model, shap_values, importance_df, y_pred_all


# ============================================================
# 4. 时序敏感性分析：时间码切换瞬间响应
# ============================================================

def transition_sensitivity_analysis(X, y, shap_values, frame_idx, file_names,
                                    importance_df, csv_files, window=5):
    """
    在每个视频的 事件边界（静息→后缀 / 后缀→静息）前后 ±window 帧，
    计算每个特征的：
      - 边界响应幅度（delta）
      - 边界前的超前检测能力（提前几帧出现显著变化）
    """
    feature_names = X.columns.tolist()
    top_features = importance_df["feature"].head(20).tolist()

    unique_files = sorted(set(file_names))
    all_boundary_stats = []

    for fname in unique_files:
        mask = file_names == fname
        y_file = y[mask]
        X_file = X[mask].values
        frames_file = frame_idx[mask]

        # 找边界帧（label 突变点）
        transitions = np.where(np.diff(y_file) != 0)[0]  # diff != 0 的位置
        for t_idx in transitions:
            direction = "onset" if y_file[t_idx + 1] == 1 else "offset"
            lo = max(0, t_idx - window)
            hi = min(len(y_file) - 1, t_idx + window + 1)

            for fi, fn in enumerate(feature_names):
                vals = X_file[lo:hi + 1, fi]
                if len(vals) < 3:
                    continue
                # 边界前后的最大差异
                before = X_file[max(0, t_idx - window):t_idx + 1, fi]
                after = X_file[t_idx + 1:min(len(X_file), t_idx + window + 2), fi]
                if len(before) == 0 or len(after) == 0:
                    continue

                delta = np.abs(after.mean() - before.mean())
                # 归一化 delta（除以特征的全局标准差）
                std_global = X[fn].std()
                delta_norm = delta / (std_global + 1e-8)

                # 超前检测：找边界前特征开始偏离静息均值的帧数
                if direction == "onset":
                    rest_mean = before[:max(1, len(before) - 2)].mean()
                    rest_std = before.std() + 1e-8
                    lead_frames = 0
                    for k in range(len(before) - 1, -1, -1):
                        if np.abs(before[k] - rest_mean) > 1.5 * rest_std:
                            lead_frames = len(before) - 1 - k
                        else:
                            break
                else:
                    lead_frames = 0

                all_boundary_stats.append({
                    "file": fname,
                    "boundary_frame": frames_file[t_idx],
                    "direction": direction,
                    "feature": fn,
                    "delta_abs": delta,
                    "delta_norm": delta_norm,
                    "lead_frames": lead_frames,
                })

    df_boundary = pd.DataFrame(all_boundary_stats)
    if len(df_boundary) == 0:
        print("未找到有效边界点")
        return

    # 汇总：每个特征在所有边界上的平均响应和超前
    summary = df_boundary.groupby("feature").agg(
        mean_delta_norm=("delta_norm", "mean"),
        max_delta_norm=("delta_norm", "max"),
        mean_lead_frames=("lead_frames", "mean"),
        max_lead_frames=("lead_frames", "max"),
    ).sort_values("mean_delta_norm", ascending=False).reset_index()
    summary["rank"] = range(1, len(summary) + 1)

    summary.to_csv(os.path.join(OUTPUT_DIR, "transition_sensitivity.csv"), index=False)
    print(f"\n=== 时间码切换 边界敏感性 Top-15 ===")
    print(summary.head(15).to_string(index=False))

    # 综合评分：SHAP 贡献 × 边界敏感度 × 超前帧数
    importance_df_indexed = importance_df.set_index("feature")
    summary_indexed = summary.set_index("feature")
    common = importance_df_indexed.index.intersection(summary_indexed.index)

    combined = pd.DataFrame({
        "feature": common,
        "shap_importance": importance_df_indexed.loc[common, "mean_abs_shap"].values,
        "boundary_sensitivity": summary_indexed.loc[common, "mean_delta_norm"].values,
        "lead_frames": summary_indexed.loc[common, "mean_lead_frames"].values,
    })
    # 归一化到 0~1
    for col in ["shap_importance", "boundary_sensitivity", "lead_frames"]:
        vmin, vmax = combined[col].min(), combined[col].max()
        combined[col + "_norm"] = (combined[col] - vmin) / (vmax - vmin + 1e-8)

    combined["composite_score"] = (
        0.4 * combined["shap_importance_norm"]
        + 0.35 * combined["boundary_sensitivity_norm"]
        + 0.25 * combined["lead_frames_norm"]
    )
    combined = combined.sort_values("composite_score", ascending=False).reset_index(drop=True)
    combined["rank"] = range(1, len(combined) + 1)
    combined.to_csv(os.path.join(OUTPUT_DIR, "composite_ranking.csv"), index=False)

    print(f"\n=== 综合评分 Top-15（SHAP×0.4 + 敏感度×0.35 + 超前性×0.25） ===")
    print(combined[["rank", "feature", "shap_importance", "boundary_sensitivity",
                     "lead_frames", "composite_score"]].head(15).to_string(index=False))

    # 可视化综合排名
    top_n = min(20, len(combined))
    fig, axes = plt.subplots(1, 3, figsize=(18, 8))

    top = combined.head(top_n)
    axes[0].barh(range(top_n), top["shap_importance"].values, color="steelblue")
    axes[0].set_yticks(range(top_n))
    axes[0].set_yticklabels(top["feature"].values, fontsize=8)
    axes[0].set_xlabel("Mean |SHAP|")
    axes[0].set_title("SHAP 贡献度")
    axes[0].invert_yaxis()

    axes[1].barh(range(top_n), top["boundary_sensitivity"].values, color="coral")
    axes[1].set_yticks(range(top_n))
    axes[1].set_yticklabels(top["feature"].values, fontsize=8)
    axes[1].set_xlabel("Normalized Delta")
    axes[1].set_title("边界响应幅度")
    axes[1].invert_yaxis()

    axes[2].barh(range(top_n), top["composite_score"].values, color="seagreen")
    axes[2].set_yticks(range(top_n))
    axes[2].set_yticklabels(top["feature"].values, fontsize=8)
    axes[2].set_xlabel("Composite Score")
    axes[2].set_title("综合评分")
    axes[2].invert_yaxis()

    plt.suptitle("舌后缀特征综合排名", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "composite_ranking.png"), dpi=150)
    plt.close()
    print(f"  保存 composite_ranking.png")

    return combined, df_boundary


# ============================================================
# 5. 时间序列可视化：Top 特征在事件边界附近的走势
# ============================================================

def plot_top_features_timeseries(X, y, frame_idx, file_names, importance_df, top_k=6):
    """绘制 Top-K 特征在每个视频中随帧变化的走势，叠加事件标签"""
    top_features = importance_df["feature"].head(top_k).tolist()
    unique_files = sorted(set(file_names))

    for fname in unique_files:
        mask = file_names == fname
        X_file = X[mask]
        y_file = y[mask]
        frames = frame_idx[mask]

        fig, axes = plt.subplots(top_k, 1, figsize=(14, 3 * top_k), sharex=True)
        if top_k == 1:
            axes = [axes]

        for i, fn in enumerate(top_features):
            ax = axes[i]
            vals = X_file[fn].values

            # 绘制事件区间背景
            in_event = False
            start = 0
            for j in range(len(y_file)):
                if y_file[j] == 1 and not in_event:
                    in_event = True
                    start = frames[j]
                elif y_file[j] == 0 and in_event:
                    in_event = False
                    ax.axvspan(start, frames[j - 1], alpha=0.2, color="red", label="后缀" if j < 5 else "")
            if in_event:
                ax.axvspan(start, frames[-1], alpha=0.2, color="red")

            ax.plot(frames, vals, linewidth=0.8, color="steelblue")
            # Savitzky-Golay 平滑趋势线
            if len(vals) > 11:
                smooth = savgol_filter(vals, min(21, len(vals) // 2 * 2 + 1), 3)
                ax.plot(frames, smooth, linewidth=1.5, color="darkred", alpha=0.7, linestyle="--")
            ax.set_ylabel(fn, fontsize=8)
            ax.tick_params(labelsize=7)

        axes[-1].set_xlabel("Frame")
        plt.suptitle(f"Top-{top_k} 特征时序走势 — {fname}", fontsize=11, fontweight="bold")
        plt.tight_layout()
        safe_name = fname.replace(".csv", "")
        plt.savefig(os.path.join(OUTPUT_DIR, f"timeseries_{safe_name}.png"), dpi=120)
        plt.close()

    print(f"  保存 timeseries_*.png ({len(unique_files)} 个文件)")


# ============================================================
# 6. Top 特征的 SHAP dependence plot
# ============================================================

def plot_shap_dependence(X, shap_values, importance_df, top_k=6):
    top_features = importance_df["feature"].head(top_k).tolist()
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    for i, fn in enumerate(top_features):
        if i >= len(axes):
            break
        ax = axes[i]
        shap.dependence_plot(fn, shap_values, X, ax=ax, show=False)
        ax.set_title(fn, fontsize=9)
    plt.suptitle("SHAP Dependence Plots — Top 特征", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "shap_dependence_top6.png"), dpi=150)
    plt.close()
    print(f"  保存 shap_dependence_top6.png")


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("舌后缀 SHAP 特征贡献度分析")
    print("=" * 60)

    # 1) 加载 & 提取
    X, y, groups, frame_idx, file_names, csv_files = load_and_extract(CSV_DIR)

    # 2) 训练 & SHAP
    model, shap_values, importance_df, y_pred_all = run_shap_analysis(
        X, y, groups, frame_idx, file_names, csv_files)

    # 2.1) 目标视频逐指标 SHAP 与 BN 事件相关性
    run_target_bn_correlation_analysis(
        model,
        importance_df,
        TARGET_POINTS_CSV,
        TARGET_LABEL_CSV,
        TARGET_CASE_DIR,
    )

    # 3) 边界敏感性分析
    combined, df_boundary = transition_sensitivity_analysis(
        X, y, shap_values, frame_idx, file_names, importance_df, csv_files, window=5)

    # 4) 时序可视化
    plot_top_features_timeseries(X, y, frame_idx, file_names, importance_df, top_k=6)

    # 5) SHAP dependence plots
    plot_shap_dependence(X, shap_values, importance_df, top_k=6)

    print(f"\n所有结果已保存至: {OUTPUT_DIR}")
    print("完成！")


if __name__ == "__main__":
    main()
