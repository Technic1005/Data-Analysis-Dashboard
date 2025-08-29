import h5py
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="DRe25 数据看板", layout="wide")
st.title("📊 DRe25 数据看板")

# -------- 工具函数 --------
def list_datasets(h5obj, prefix=""):
    paths = []
    if isinstance(h5obj, h5py.Dataset):
        return [prefix.rstrip("/")]
    elif isinstance(h5obj, h5py.Group):
        for k, v in h5obj.items():
            paths.extend(list_datasets(v, f"{prefix}/{k}"))
    return paths

def read_two_columns(ds: h5py.Dataset):
    shape = ds.shape
    if len(shape) == 2:
        if shape[1] == 2:
            return ds[:,0].ravel(), ds[:,1].ravel()
        if shape[0] == 2:
            return ds[0,:].ravel(), ds[1,:].ravel()
    elif len(shape) == 3:
        if shape[1] == 2:
            return ds[:,0,0].ravel(), ds[:,1,0].ravel()
        if shape[0] == 2:
            return ds[0,:,0].ravel(), ds[1,:,0].ravel()
    return None

def robust_min(arr):
    arr = np.asarray(arr)
    arr = arr[np.isfinite(arr)]
    return np.min(arr) if arr.size else np.inf

# -------- 主体 --------
uploaded = st.file_uploader("上传 .mat（v7.3 HDF5）文件", type=["mat"])
if uploaded is not None:
    with h5py.File(uploaded, "r") as f:
        all_ds = list_datasets(f, "")
        candidate = []
        for p in all_ds:
            ds = f[p]
            if read_two_columns(ds) is not None:
                candidate.append(p)

        st.sidebar.header("选择要分析的变量")
        selected = st.sidebar.multiselect("变量", candidate)

        if selected:
            # 全局最小时间
            global_min_t = np.inf
            for path in selected:
                ds = f[path]
                cols = read_two_columns(ds)
                if cols is None:
                    continue
                t_raw, _ = cols
                global_min_t = min(global_min_t, robust_min(t_raw))

            is_days = 50000 < global_min_t < 1_000_000

            # 解析并对齐
            processed = {}
            for path in selected:
                ds = f[path]
                cols = read_two_columns(ds)
                if cols is None:
                    continue
                t_raw, y = cols
                mask = np.isfinite(t_raw) & np.isfinite(y)
                t_raw = t_raw[mask]
                y = y[mask]
                if is_days:
                    t = (t_raw - global_min_t) * 86400.0
                else:
                    t = t_raw - global_min_t
                order = np.argsort(t)
                processed[path] = pd.DataFrame({"time": t[order], "value": y[order]})

            if not processed:
                st.warning("所选变量未能解析成两列数据，请检查数据形状。")
            else:
                # -------- 自动 y 轴开关 --------
                auto_y = st.sidebar.checkbox("缩放 X 时自动调整 Y 轴", value=True)

                # -------- 自定义分组绘图 --------
                st.sidebar.header("自定义图组")
                if "groups" not in st.session_state:
                    st.session_state.groups = []

                num_groups = st.sidebar.number_input("需要几张图？", min_value=1, max_value=10, value=1, step=1)

                # 保证 session_state.groups 长度和 num_groups 对齐
                while len(st.session_state.groups) < num_groups:
                    st.session_state.groups.append([])
                while len(st.session_state.groups) > num_groups:
                    st.session_state.groups.pop()

                # 分组选择
                for i in range(num_groups):
                    st.session_state.groups[i] = st.sidebar.multiselect(
                        f"图 {i+1} 包含的变量",
                        options=list(processed.keys()),
                        default=st.session_state.groups[i],
                        key=f"group_{i}"
                    )

                groups = st.session_state.groups

                # -------- 同步 X 轴滑块 --------
                t_min = min(df["time"].iloc[0] for df in processed.values())
                t_max = max(df["time"].iloc[-1] for df in processed.values())
                t_range = st.slider("选择时间区间 (s)", float(t_min), float(t_max), (float(t_min), float(t_max)))

                # 绘图
                for i, group in enumerate(groups, start=1):
                    if not group:
                        continue
                    fig = go.Figure()
                    for name in group:
                        df = processed[name]
                        df_plot = df[(df["time"] >= t_range[0]) & (df["time"] <= t_range[1])]
                        fig.add_trace(go.Scatter(
                            x=df_plot["time"], y=df_plot["value"], mode="lines", name=name,
                            hovertemplate="t=%{x:.6f}s<br>%{meta}=%{y:.6g}<extra></extra>",
                            meta=name
                        ))
                    fig.update_layout(
                        title=f"图 {i} ：{'、'.join(group)}",
                        xaxis_title="时间 (s)",
                        yaxis_title="数值",
                        hovermode="x unified",
                        legend=dict(orientation="h", y=-0.2),
                        yaxis=dict(autorange=auto_y),
                        xaxis=dict(range=t_range)  # 强制同步 X 轴范围
                    )
                    st.plotly_chart(fig, use_container_width=True)

                # -------- 数值面板 --------
                st.subheader("🔎 数值面板（选择时间查看各变量数值）")
                t_pick = st.slider("选择具体时间点 (s)", float(t_min), float(t_max), float(t_min))
                rows_table = []
                for name, df in processed.items():
                    idx = np.searchsorted(df["time"].values, t_pick)
                    if idx == 0:
                        ti, yi = df["time"].iloc[0], df["value"].iloc[0]
                    elif idx >= len(df):
                        ti, yi = df["time"].iloc[-1], df["value"].iloc[-1]
                    else:
                        t0, t1 = df["time"].iloc[idx-1], df["time"].iloc[idx]
                        if abs(t_pick - t0) <= abs(t_pick - t1):
                            ti, yi = t0, df["value"].iloc[idx-1]
                        else:
                            ti, yi = t1, df["value"].iloc[idx]
                    rows_table.append({"变量": name, "时间(s)": float(ti), "数值": float(yi)})
                st.dataframe(pd.DataFrame(rows_table), use_container_width=True)
