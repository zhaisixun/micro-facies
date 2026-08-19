"""
GR 砂泥界面识别（高 GR=泥，低 GR=砂）
1. 普通半幅点：全井泥/砂参考下半幅穿越；较轻平滑减轻薄层被抹。局部 GR 极小（左突）用局部泥顶/泥底半幅补薄层。纯泥高 GR：邻域未到砂侧则删界面，防泥里毛刺划层。
2. 突变界面：梯度峰（钟形底、漏斗顶、箱形顶底等陡变）。
3. 形态：仅砂体段标 Bell/Funnel/Box/Normal；箱形要求中部 GR 相对稳定。泥岩间隔不单列；不明显则 Normal 仍半幅。Bell 顶更靠泥（高于半幅阈值穿越）；Funnel 底更靠砂（低于半幅侧穿越）。
约定：深度宜浅→深递增；递减则警告并重排。
"""

import os
import pandas as pd
import numpy as np
from scipy.signal import medfilt, find_peaks
from scipy.ndimage import gaussian_filter1d


# ---------- 可调参数 ----------
MEDIAN_KERNEL = 1  # 中值滤波去噪，越大越平滑
GAUSSIAN_SIGMA = 1  # 高斯平滑（用于梯度、形态分类）；半幅穿越单独用较轻平滑见下
# 半幅/薄层检测用：0 表示不再做高斯，仅用中值结果，利于薄砂体回返不被抹掉
GAUSSIAN_SIGMA_FOR_HALF = 0
GRAD_PROMINENCE_QUANTILE = 0.85  # 梯度峰 prominence 阈值，越大越敏感
MERGE_EPS_DEPTH = 0.25  # 合并界面 eps，越大越宽松
RAMP_FRACTION = 0.25  # 渐变带宽度阈值，越大越宽松

# 局部薄砂体：GR 向左突出（局部极小）的 prominence 占全幅比例、峰间最小距离(米)
THIN_SAND_PROMINENCE_FRAC = 0.10 
THIN_SAND_MIN_DISTANCE_M = 0.12
THIN_SAND_LOBE_HALF_WINDOW_M = 4.0  # 局部泥顶/泥底搜索半窗(米)

# 箱形：中部相对平稳（相对全幅的标准差、极差上限）
BOX_MID_STD_FRAC = 0.11
BOX_MID_RANGE_FRAC = 0.30
# 钟形/漏斗：顶底渐变宽度不对称倍数
BELL_FUNNEL_RAMP_RATIO = 1.28

# 纯泥高 GR 段：邻域内 GR 从未低到「砂侧」则去掉界面（防泥里毛刺被划层）
PURE_MUD_WIN_M = 0.5
PURE_MUD_MIN_FRAC = 0.38  # 须 min(邻域GR) <= sand + frac*amp 才保留

# Bell 顶：泥→砂 穿越阈值 = sand + BELL_TOP_FRAC * (mud-sand)，>0.5 更靠泥（GR 更高）
BELL_TOP_FRAC = 0.75  # 泥→砂 穿越阈值，越大越靠泥
# Funnel 底：砂→泥 穿越阈值 = sand + FUNNEL_BOTTOM_FRAC * (mud-sand)，<0.5 更靠砂（GR 更低）
FUNNEL_BOTTOM_FRAC = 0.28  # 砂→泥 穿越阈值，越大越靠砂

# 判定“陡变”：渐变带宽度 < 该值(米) 视为突变侧（与采样间距相关）
ABRUPT_WIDTH_FACTOR = 5.0  # 陡变带宽度阈值，调大：更不容易因宽度判成陡 → 更多 Normal、更少 Box。调小：更容易判两侧陡 → 更多 Box。
ABRUPT_WIDTH_MIN_M = 0.28  # 陡变带宽度最小值，调大：更偏「宽渐变」→ Normal/Bell/Funnel 比例变。调小：更敏感。
# 局部 |梯度| 超过全曲线分位比例则视为陡变
GRAD_STEEP_FRAC_OF_P90 = 0.42  # 局部 |梯度| 超过全曲线分位比例则视为陡变，调大：更难算陡 → 更多 Normal。调小：更容易算陡 → 更多 Bell/Funnel/Box。


# 计算泥岩和砂岩的参考值
def _mud_sand_levels(gr_smooth: np.ndarray): 
    mud = float(np.percentile(gr_smooth, 60))    # 6分位数
    sand = float(np.percentile(gr_smooth, 40))  # 4分位数
    half = (mud + sand) / 2.0  # 半幅参考值
    amp = max(mud - sand, 1e-6)
    return mud, sand, half, amp


def _interpolate_crossing(depth, gr, i_lo, i_hi, level, expect_decrease: bool):
    """[i_lo,i_hi] 内 GR 穿越 level；True=泥→砂(沿深GR降)，False=砂→泥(GR升)。lobe 局部半幅同此。"""
    n = len(gr)
    i_lo, i_hi = int(np.clip(i_lo, 0, n - 2)), int(np.clip(i_hi, 0, n - 2))
    if i_lo > i_hi:
        i_lo, i_hi = i_hi, i_lo
    for i in range(i_lo, i_hi + 1):
        g0, g1 = gr[i], gr[i + 1]
        if abs(g1 - g0) < 1e-12 or (g0 - level) * (g1 - level) > 0:
            continue
        t = (level - g0) / (g1 - g0)
        if not (0 <= t <= 1):
            continue
        if expect_decrease and g1 >= g0:
            continue
        if not expect_decrease and g1 <= g0:
            continue
        return float(depth[i] + t * (depth[i + 1] - depth[i]))
    return None


def _half_pick(d, iface, detail):
    """半幅事件字典（全局半幅与 lobe 共用结构）。"""
    return {"Depth": float(d), "Category": "HalfAmplitude", "Interface": iface, "Detail": detail}


def detect_lobe_half_crossings(depth, gr, mud, sand, amp, dz):
    """薄砂体/GR左突：局部极小两侧取泥顶泥底，局部半幅补穿越（全局半幅过高时薄层落不到半幅线）。"""
    picks = []
    n = len(gr)
    if n < 7:
        return picks
    prom = max(THIN_SAND_PROMINENCE_FRAC * amp, 1e-6)  # 局部极小 prominence 阈值，不够小的会被忽略
    dist = max(2, int(THIN_SAND_MIN_DISTANCE_M / max(dz, 1e-6)))  # 局部极小间最小距离
    span = int(np.clip(THIN_SAND_LOBE_HALF_WINDOW_M / max(dz, 1e-6), 8, 150))  # 局部极小搜索半窗

    mins, _ = find_peaks(-gr, prominence=prom, distance=dist)  # 返回极小值点的索引。局部极小，找出极小值小于prom的点，distance=dist是为了找距离大于dist的点

    for im in mins:   # im是极小值点的索引
        vmin = float(gr[im])  # gr极小值
        if vmin > mud - 0.11 * amp:  # 局部极小值不够低，则忽略
            continue
        i0 = max(0, im - span)  # 局部极小搜索半窗  靠近井段开头时，im-span 可能 < 0，就强制变 0
        i1 = min(n - 1, im + span)  # 局部极小搜索半窗  靠近井段末尾时，im+span 可能 > 最后下标，就强制到 n-1
        if im <= i0 or im >= i1:  # 局部极小不在搜索半窗内，则忽略
            continue
        ileft = i0 + int(np.argmax(gr[i0 : im + 1]))  # 局部极小左侧最大值
        iright = im + int(np.argmax(gr[im : i1 + 1]))  # 局部极小右侧最大值
        if ileft >= im or iright <= im:
            continue
        mud_top = float(gr[ileft])  # gr值
        mud_bot = float(gr[iright])
        if mud_top < vmin + 0.04 * amp or mud_bot < vmin + 0.04 * amp:  # 泥gr比砂gr不够高，忽略
            continue
        half_t = (mud_top + vmin) / 2.0  # 泥顶半幅值
        half_b = (vmin + mud_bot) / 2.0  # 泥底半幅值
        if half_t <= vmin + 1e-6 or half_b <= vmin + 1e-6:  # 半幅值比极小值高，忽略
            continue
        dp_top = _interpolate_crossing(depth, gr, ileft, im, half_t, True)  # 
        dp_bot = _interpolate_crossing(depth, gr, im, iright, half_b, False)
        if dp_top is None or dp_bot is None or dp_bot <= dp_top:
            continue
        picks.append(_half_pick(dp_top, "SandTop", "lobe_local_half"))
        picks.append(_half_pick(dp_bot, "SandBase", "lobe_local_half"))
    return picks


def augment_half_picks_avoid_duplicate(global_picks, extra_picks, eps=0.12):
    """lobe 补点：若与全局半幅同类型界面已很近则不再重复添加。"""
    out = list(global_picks)
    for p in extra_picks:
        if any(
            q["Interface"] == p["Interface"] and abs(q["Depth"] - p["Depth"]) < eps
            for q in global_picks
        ):
            continue
        out.append(p)
    return out

# 只要上下gr值跨过半幅参考值了，就记录该深度为半幅穿越点
# 找到第一个gr值半幅穿越对应的深度点，
# 找到所有半幅穿越点，返回一个列表，每个元素是一个字典，包含深度、类别、界面类型和详细信息
def detect_half_amplitude_crossings(depth, gr_smooth, half, amp):
    picks, n = [], len(gr_smooth)
    for i in range(n - 1):
        g0, g1 = gr_smooth[i], gr_smooth[i + 1]  # 当前与下一 GR
        d0, d1 = depth[i], depth[i + 1]  # 当前与下一深度
        if (g0 - half) * (g1 - half) > 0 or abs(g1 - g0) < 1e-9:  # 上下gr在半幅线同侧或无变化，跳过
            continue
        t = (half - g0) / (g1 - g0)  # 得到穿过点在两点中间对应的gr值
        if not (0 <= t <= 1):
            continue
        dp = d0 + t * (d1 - d0)  #    d1-d0=0.1
        if g1 < g0:  # GR 降 → 泥→砂
            picks.append(_half_pick(dp, "SandTop", "mud_to_sand"))
        else:  # GR 升 → 砂→泥
            picks.append(_half_pick(dp, "SandBase", "sand_to_mud"))
    return picks


def detect_abrupt_interfaces(depth, grad):
    # 测井图深度纵轴；求导对深度自变量（横轴数值列）
    gabs = np.abs(grad[np.isfinite(grad)])  # 梯度绝对值，去 NaN
    prom = (max(float(np.percentile(gabs, GRAD_PROMINENCE_QUANTILE) * 0.35), 1e-9) if gabs.size else 1e-9)  # prominence
    picks = []
    pos_idx, _ = find_peaks(grad, prominence=prom)  # 正梯度峰，砂→泥
    neg_idx, _ = find_peaks(-grad, prominence=prom)  # 负梯度峰，泥→砂
    for i in pos_idx:
        picks.append({"Depth": float(depth[i]), "Category": "Abrupt", "Interface": "SandBase", "Detail": "sharp_sand_to_mud"})
    for i in neg_idx:
        picks.append({"Depth": float(depth[i]), "Category": "Abrupt", "Interface": "SandTop", "Detail": "sharp_mud_to_sand"})
    return picks


def _ramp_width_depth(
    depth, gr, i_center, mud, sand, going_sand: bool, max_span=80
):
    amp = max(mud - sand, 1e-6)
    low = sand + RAMP_FRACTION * amp
    high = mud - RAMP_FRACTION * amp
    n = len(gr)

    def walk(start, step):
        i = start
        first_high, first_low = None, None
        for _ in range(max_span):
            if i < 0 or i >= n:
                break
            g = gr[i]
            if first_high is None and g >= high:
                first_high = i
            if first_high is not None and g <= low:
                first_low = i
                break
            i += step
        return first_high, first_low

    if going_sand:
        fh, fl = walk(i_center, 1)
        if fh is None or fl is None:
            fh, fl = walk(i_center, -1)
    else:
        fh, fl = walk(i_center, 1)
        if fh is None or fl is None:
            fh, fl = walk(i_center, -1)

    if fh is None or fl is None:
        return float("nan")
    i0, i1 = sorted((fh, fl))
    return float(abs(depth[i1] - depth[i0]))


def _local_grad_max_abs(grad, i_center, win=12):
    lo = max(0, i_center - win)
    hi = min(len(grad), i_center + win + 1)
    chunk = grad[lo:hi]
    if chunk.size == 0:
        return 0.0
    return float(np.nanmax(np.abs(chunk)))


def _sand_body_middle_stats(gr, i_top, i_bot, amp):
    """中部去掉上下各1/4，std/极差相对全幅，供箱形判定。"""
    span = i_bot - i_top
    if span < 6:
        i0, i1 = i_top, i_bot
    else:
        q = max(1, span // 4)
        i0, i1 = i_top + q, i_bot - q
    if i1 <= i0:
        i0, i1 = i_top, i_bot
    seg = gr[i0 : i1 + 1]
    if seg.size == 0:
        return 1.0, 1.0
    amp = max(amp, 1e-6)
    std_f = float(np.std(seg) / amp)
    range_f = float((np.max(seg) - np.min(seg)) / amp)
    return std_f, range_f


def classify_sand_body_shape(depth, gr_smooth, grad, i_top, i_bot, mud, sand):
    """箱：两侧陡且中部稳。钟：顶缓底陡。漏：顶陡底缓。双陡中摆大→按渐变宽偏钟/漏或 Normal。"""
    dz = float(np.median(np.diff(depth))) if len(depth) > 1 else 0.1
    dz = max(dz, 1e-6)
    abrupt_w_max = max(ABRUPT_WIDTH_FACTOR * dz, ABRUPT_WIDTH_MIN_M)
    amp = max(mud - sand, 1e-6)

    w_top = _ramp_width_depth(depth, gr_smooth, i_top, mud, sand, going_sand=True)
    w_bot = _ramp_width_depth(depth, gr_smooth, i_bot, mud, sand, going_sand=False)

    gfin = grad[np.isfinite(grad)]
    g_ref = float(np.percentile(np.abs(gfin), 90)) if gfin.size else 1e-9
    g_ref = max(g_ref, 1e-9)
    thr_g = GRAD_STEEP_FRAC_OF_P90 * g_ref
    thr_g_strict = 0.52 * g_ref

    g_top = _local_grad_max_abs(grad, i_top)
    g_bot = _local_grad_max_abs(grad, i_bot)

    steep_top_grad = g_top >= thr_g
    steep_bot_grad = g_bot >= thr_g
    narrow_top = np.isfinite(w_top) and w_top < abrupt_w_max
    narrow_bot = np.isfinite(w_bot) and w_bot < abrupt_w_max

    steep_top = steep_top_grad and (narrow_top or g_top >= thr_g_strict)
    steep_bot = steep_bot_grad and (narrow_bot or g_bot >= thr_g_strict)

    if np.isfinite(w_top) and w_top >= abrupt_w_max * 1.35:
        steep_top = False
    if np.isfinite(w_bot) and w_bot >= abrupt_w_max * 1.35:
        steep_bot = False

    gradual_top = not steep_top
    gradual_bot = not steep_bot

    mid_std_f, mid_range_f = _sand_body_middle_stats(gr_smooth, i_top, i_bot, amp)
    is_box_middle = (mid_std_f < BOX_MID_STD_FRAC) and (
        mid_range_f < BOX_MID_RANGE_FRAC
    )

    w_top_v = float(w_top) if np.isfinite(w_top) else 0.0
    w_bot_v = float(w_bot) if np.isfinite(w_bot) else 0.0
    top_wider = w_top_v >= BELL_FUNNEL_RAMP_RATIO * (w_bot_v + 1e-6)
    bot_wider = w_bot_v >= BELL_FUNNEL_RAMP_RATIO * (w_top_v + 1e-6)

    shape, note = "Normal", "half_amplitude_no_clear_shape"

    if gradual_top and steep_bot:
        shape, note = "Bell", "upper_gradual_lower_abrupt"
    elif steep_top and gradual_bot:
        shape, note = "Funnel", "upper_abrupt_lower_gradual"
    elif steep_top and steep_bot:
        if is_box_middle:
            shape, note = "Box", "both_abrupt_stable_middle"
        elif top_wider and not bot_wider:
            shape, note = "Bell", "both_steep_wiggly_mid_favor_gradual_top"
        elif bot_wider and not top_wider:
            shape, note = "Funnel", "both_steep_wiggly_mid_favor_gradual_bot"
        else:
            shape, note = "Normal", "both_abrupt_but_unstable_middle"
    elif not steep_top and not steep_bot:
        if top_wider and not bot_wider:
            shape, note = "Bell", "wide_top_narrow_bot_ramps"
        elif bot_wider and not top_wider:
            shape, note = "Funnel", "narrow_top_wide_bot_ramps"

    w_top_out = w_top_v
    w_bot_out = w_bot_v

    return {
        "Shape": shape,
        "RampWidthTop_m": w_top_out,
        "RampWidthBottom_m": w_bot_out,
        "Note": note,
        "SteepTop": steep_top,
        "SteepBottom": steep_bot,
        "MidStdFrac": mid_std_f,
        "MidRangeFrac": mid_range_f,
        "i_top": i_top,
        "i_bot": i_bot,
    }


def _depth_at_grad_extreme(depth, grad, i_center, mud_to_sand: bool, win=18):
    """mud_to_sand：负梯度最强点；sand_to_mud：正梯度最强点。"""
    lo = max(0, i_center - win)
    hi = min(len(grad), i_center + win + 1)
    chunk = grad[lo:hi]
    if chunk.size == 0:
        return None
    if mud_to_sand:
        j = lo + int(np.nanargmin(chunk))
    else:
        j = lo + int(np.nanargmax(chunk))
    return float(depth[j])


def refine_interval_boundaries(depth, gr_smooth, grad, top_d, bot_d, i_top, i_bot, mud, sand, amp, shape):
    """Normal/Box 保持半幅顶底；Bell 顶偏泥阈值+底正梯度峰；Funnel 顶负梯度峰+底偏砂阈值。"""
    span = max(8, min(40, (i_bot - i_top) // 3 + 5))
    i0t, i1t = max(0, i_top - span), min(len(depth) - 2, i_top + span)
    i0b, i1b = max(0, i_bot - span), min(len(depth) - 2, i_bot + span)

    top_method = "HalfAmplitude"
    bot_method = "HalfAmplitude"
    new_top, new_bot = float(top_d), float(bot_d)

    bell_thresh = sand + BELL_TOP_FRAC * amp
    funnel_bot_thresh = sand + FUNNEL_BOTTOM_FRAC * amp

    if shape == "Bell":
        cd = _interpolate_crossing(depth, gr_smooth, i0t, i1t, bell_thresh, True)
        if cd is not None:
            new_top, top_method = cd, "NearMud_frac75"
        db = _depth_at_grad_extreme(depth, grad, i_bot, mud_to_sand=False)
        if db is not None:
            new_bot, bot_method = db, "AbruptGrad_base"
    elif shape == "Funnel":
        dt = _depth_at_grad_extreme(depth, grad, i_top, mud_to_sand=True)
        if dt is not None:
            new_top, top_method = dt, "AbruptGrad_top"
        cd = _interpolate_crossing(depth, gr_smooth, i0b, i1b, funnel_bot_thresh, False)
        if cd is not None:
            new_bot, bot_method = cd, "NearSand_frac28"

    if new_top >= new_bot:
        return float(top_d), float(bot_d), "HalfAmplitude", "HalfAmplitude"

    return new_top, new_bot, top_method, bot_method

# 顶→底配对得砂体段；classify/refine 需整段看顶缓底陡等形态
def build_sand_intervals_from_half(half_amp_picks):
    events = sorted(
        [
            (p["Depth"], p["Interface"])
            for p in half_amp_picks
            if p["Category"] == "HalfAmplitude"
        ],
        key=lambda x: x[0],
    )
    intervals = []
    pending_top = None
    for d, iface in events:
        if iface == "SandTop":
            pending_top = d
        elif iface == "SandBase" and pending_top is not None:
            if d > pending_top:
                intervals.append((pending_top, d))
            pending_top = None
    return intervals

def _drop_pure_mud_picks(depth, gr, picks, sand, amp, dz):
    """邻域最低 GR 仍高于 sand+frac*amp → 未进入砂侧，删（高泥里假界面）。"""
    if not picks:
        return picks
    amp = max(float(amp), 1e-9)
    thr = sand + PURE_MUD_MIN_FRAC * amp
    w = max(2, int(PURE_MUD_WIN_M / max(dz, 1e-6)))
    out = []
    for p in picks:
        i = int(np.clip(np.searchsorted(depth, p["Depth"]), 0, len(gr) - 1))
        lo, hi = max(0, i - w), min(len(gr), i + w + 1)
        if np.min(gr[lo:hi]) <= thr:
            out.append(p)
    return out


# 半幅与梯度峰地质同一界面、数学深度可能不同，近则合并
def merge_close_picks(picks, eps=MERGE_EPS_DEPTH):
    if not picks:
        return []
    picks = sorted(picks, key=lambda x: x["Depth"])
    merged = []
    for p in picks:
        if (
            merged
            and merged[-1]["Interface"] == p["Interface"]
            and abs(p["Depth"] - merged[-1]["Depth"]) <= eps
        ):
            last = merged[-1]
            n = last.pop("_n", 1)
            new_d = (last["Depth"] * n + p["Depth"]) / (n + 1)
            cats = {last["Category"], p["Category"]}
            last["Depth"] = float(new_d)
            last["Category"] = (
                "HalfAmplitude+Abrupt" if len(cats) > 1 else last["Category"]
            )
            ds = set(last["Detail"].split(";")) | {p["Detail"]}
            last["Detail"] = ";".join(sorted(ds))
            last["_n"] = n + 1
        else:
            q = dict(p)
            q["_n"] = 1
            merged.append(q)
    for m in merged:
        m.pop("_n", None)
    return merged


def process_gr_segmentation(
    input_excel,
    output_csv,
    depth_col=None,
    gr_col=None,
    also_export_intervals=True,
):
    df = pd.read_excel(input_excel)
    depth = df[depth_col].values
    gr = df[gr_col].values

    gr_clean = medfilt(gr.astype(float), kernel_size=MEDIAN_KERNEL)  # 中值去噪
    dz = max(float(np.median(np.diff(depth))) if len(depth) > 1 else 0.1, 1e-6)  # 采样间距   
    gr_half = (
        gaussian_filter1d(gr_clean, sigma=float(GAUSSIAN_SIGMA_FOR_HALF))   # 如果使用高斯滤波，否则用中值滤波
        if GAUSSIAN_SIGMA_FOR_HALF
        else gr_clean
    )  # 半幅用较轻平滑；0 则仅中值

    gr_smooth = gaussian_filter1d(gr_clean, sigma=GAUSSIAN_SIGMA)  # 形态/梯度用高斯   ##是否要做？

    mud, sand, half, amp = _mud_sand_levels(gr_smooth)  # 全井泥砂参考与半幅
    grad = np.gradient(gr_smooth, depth)  # 梯度，对gr_smooth求导，depth为自变量

    half_global_only = detect_half_amplitude_crossings(depth, gr_half, half, amp)  # 全局半幅穿越
    lobe_picks = detect_lobe_half_crossings(depth, gr_half, mud, sand, amp, dz)  # 局部薄砂体半幅穿越
    half_picks = augment_half_picks_avoid_duplicate(
        half_global_only, lobe_picks, eps=0.12
    )
    abrupt_picks = detect_abrupt_interfaces(depth, grad)  # 突变峰，grad 来自 gr_smooth
    half_picks = _drop_pure_mud_picks(depth, gr_smooth, half_picks, sand, amp, dz)
    abrupt_picks = _drop_pure_mud_picks(depth, gr_smooth, abrupt_picks, sand, amp, dz)

    combined = half_picks + abrupt_picks
    merged_interfaces = merge_close_picks(combined, eps=MERGE_EPS_DEPTH)

    interval_rows = []
    for top_d, bot_d in build_sand_intervals_from_half(half_picks):
        i_top = int(np.clip(np.searchsorted(depth, top_d), 0, len(depth) - 1))  # 顶界索引
        i_bot = int(np.clip(np.searchsorted(depth, bot_d), 0, len(depth) - 1))  # 底界索引
        if i_bot <= i_top:
            continue
        info = classify_sand_body_shape(depth, gr_smooth, grad, i_top, i_bot, mud, sand)
        rt, rb, tm, bm = refine_interval_boundaries(
            depth, gr_smooth, grad, top_d, bot_d, i_top, i_bot, mud, sand, amp, info["Shape"]
        )
        interval_rows.append({
            "Top": rt, "Bottom": rb, "Type": info["Shape"],
            "RampWidthTop_m": info["RampWidthTop_m"], "RampWidthBottom_m": info["RampWidthBottom_m"],
            "MidStdFrac": info.get("MidStdFrac", np.nan), "MidRangeFrac": info.get("MidRangeFrac", np.nan),
            "Note": info["Note"], "TopPick": tm, "BottomPick": bm,
        })

    out_dir = os.path.dirname(os.path.abspath(output_csv)) or "."
    os.makedirs(out_dir, exist_ok=True)

    df_iface = pd.DataFrame(merged_interfaces)
    df_iface = df_iface.sort_values("Depth").reset_index(drop=True)
    df_iface.to_csv(output_csv, index=False, encoding="utf-8-sig")

    print(f"泥岩参考(约 p90): {mud:.3f}  砂岩参考(约 p10): {sand:.3f}  半幅: {half:.3f}")
    print(
        f"半幅: 全局 {len(half_global_only)} + 薄层局部 {len(lobe_picks)} "
        f"→ 去重后 {len(half_picks)}  | 突变峰 {len(abrupt_picks)}  | 合并界面 {len(merged_interfaces)}"
    )
    print(f"界面结果已保存: {output_csv}")

    if also_export_intervals and interval_rows:
        base, ext = os.path.splitext(output_csv)
        interval_path = f"{base}_intervals{ext or '.csv'}"
        pd.DataFrame(interval_rows).to_csv(interval_path, index=False, encoding="utf-8-sig")
        print(f"砂体段（仅 GR 较低段）已保存: {interval_path}  (共 {len(interval_rows)} 段)")


if __name__ == "__main__":
    process_gr_segmentation(
        r"C:\Users\ZSX\Desktop\EP15-2-1.xlsx",
        "techlog_import_tops_15-2-1.csv",
    )
