# src/main.py
from pathlib import Path
import pandas as pd
import sys
try:
    # mode 1: run as module -> python -m src.main
    from . import io as cio
    from . import plot as cplots
    from . import lag as clag
except Exception:
    # mode 2: run as script -> python src/main.py
    ROOT = Path(__file__).resolve().parents[1]  # project root
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from src import io as cio
    from src import plot as cplots
    from src import lag as clag

# Paths
DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "data" / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
# POT quantiles
QUANTILES = [0.90, 0.95, 0.99]

# Driver-specific configurations
CONFIGS = {
    "tide": {
        "min_separation_days": 3,
        "fixed_window_days": 10,
        "max_lag": 5,
        "min_overlap": 3
    },
    "flow": {
        "min_separation_days": 3,
        "fixed_window_days": 40,
        "max_lag": 30,
        "min_overlap": 10
    }
}

#=============================================================================
# Step 1: Load Data
#=============================================================================
# Load all data at once with correct column names
rain = cio.load_source_data(
    DATA_DIR / "daily_rainfall_1981_2012.csv",
    date_col="timedate",
    value_col="Precipitation",
    dayfirst=True
).rename(columns={"value": "rain"})  

tide = cio.load_source_data(
    DATA_DIR / "tide_81_12_modified.csv",
    date_col="timedate",
    value_col="tide",
    dayfirst=True
).rename(columns={"value": "tide"})  

flow = cio.load_source_data(
    DATA_DIR / "flow_81_12.csv",
    date_col="timedate",
    value_col="Flow",
    dayfirst=True
).rename(columns={"value": "flow"})  

wl = cio.load_source_data(
    DATA_DIR / "ct_waterlevel_81_12.csv",
    date_col="timedate",
    value_col="Water Level",
    dayfirst=True
).rename(columns={"value": "wl"}) 

#=============================================================================
# Step 2: Plot Climatology
#=============================================================================
plot_data = cio.prepare_for_plotting({
    "rain": rain,
    "tide": tide,
    "flow": flow,
    "wl": wl
})
# Merge for plotting
df_plot = cio.merge_source_data(plot_data)
# Generate plot
out_png = str(OUTPUT_DIR / "climatology_plus_normalised.png")
cplots.plot_climatology_plus_normalised(
    df_plot, out_png,
    time_col="timedate",
    title_map={
        "rain": "(a) Rainfall (mm/day)",
        "tide": "(b) Tide level (m)",
        "flow": "(c) River flow (m³/s)",
        "water level": "(d) Water level (m)",
    },
    short_map={
        "rain": "Rain",
        "tide": "Tide",
        "flow": "Flow",
        "water level": "Water level",
    },
    include_normalised=True
)
print(f"✓ Saved: {out_png}")

#=============================================================================
# Step 3: Global Lag Analysis
#=============================================================================
# Merge all data for lag analysis
df_wide = pd.merge(tide, flow, on="timedate", how="inner")
df_wide = pd.merge(df_wide, wl, on="timedate", how="inner")
df_wide = df_wide.dropna().sort_values("timedate").reset_index(drop=True)
print(f"Merged data: {len(df_wide)} rows")

# Analyze forward lags
result = clag.estimate_forward_lags(
    df_wide=df_wide,
    target="wl",
    drivers=["tide", "flow"],
    max_lag_days=30,
    method="spearman",
    positive_only=True
)

print("\nGlobal Lag Analysis:")
for driver in ["tide", "flow"]:
    print(f"  {driver:8s}: {result.best_lags[driver]:3d} days, "
          f"Correlation = {result.corr_at_best[driver]:.3f}")

# Save global lag results
global_summary = pd.DataFrame({
    "Driver": list(result.best_lags.keys()),
    "Best_Lag_Days": list(result.best_lags.values()),
    "Correlation": list(result.corr_at_best.values())
})
global_summary.to_csv(OUTPUT_DIR / "global_lag_summary.csv", index=False)
print(f"\n✓ Saved: global_lag_summary.csv")

# Save aligned data
aligned_df = clag.apply_lags_and_align(df_wide, result.best_lags)
aligned_df.to_csv(OUTPUT_DIR / "aligned_data.csv", index=False)
print(f"✓ Saved: aligned_data.csv")

#=============================================================================
# Step 4: POT Event-Driven Lag Analysis
#=============================================================================

# Analyse Tide → WL
tide_summary = clag.run_pot_lag_analysis(
    driver_name="tide",
    target_name="wl",
    driver_df=tide,
    target_df=wl,
    quantiles=QUANTILES,
    config=CONFIGS["tide"],
    output_dir=OUTPUT_DIR
)

# Analyse Flow → WL
flow_summary = clag.run_pot_lag_analysis(
    driver_name="flow",
    target_name="wl",
    driver_df=flow,
    target_df=wl,
    quantiles=QUANTILES,
    config=CONFIGS["flow"],
    output_dir=OUTPUT_DIR
)


#=============================================================================
# Step 5: Combined Summary
#=============================================================================

tide_summary["Driver"] = "Tide"
flow_summary["Driver"] = "Flow"
combined = pd.concat([tide_summary, flow_summary], ignore_index=True)

# 重新排序列
cols_order = ["Driver", "quantile", "N_events", "N_valid_lags",
              "median_lag", "IQR_lag", "median_r", "IQR_r"]
combined = combined[[c for c in cols_order if c in combined.columns]]

print("\nCombined Summary:")
print(combined.to_string(index=False))

# 保存
combined.to_csv(OUTPUT_DIR / "pot_combined_summary.csv", index=False)
print(f"\n✓ Saved: pot_combined_summary.csv")

#=============================================================================
# 6. 探索 POT 事件统计
#=============================================================================

print("\n" + "="*70)
print("第6步：POT 事件统计")
print("="*70)

for name, df_data in [("Tide", tide), ("Flow", flow)]:
    print(f"\n{name}:")
    col = name.lower()
    
    for q in QUANTILES:
        events, threshold = clag.pick_pot_events(
            df=df_data,
            value_col=col,
            quantile=q,
            min_separation_days=3
        )
        print(f"  Q{int(q*100)}: {len(events):3d} 事件, 阈值 = {threshold:.2f}")

#=============================================================================
# 完成
#=============================================================================

print("\n" + "="*70)
print("分析完成！")
print("="*70)
print(f"\n输出目录:")
print(f"  主目录: {OUTPUT_DIR}")
print(f"  滞后分析: {OUTPUT_DIR}")

print("\n主要输出文件:")
print(f"  1. climatology_plus_normalised.png  - 气候态图")
print(f"  2. global_lag_summary.csv           - 全局最优滞后")
print(f"  3. aligned_data.csv                 - 时间对齐数据")
print(f"  4. tide_pot_summary.csv             - 潮汐POT摘要")
print(f"  5. flow_pot_summary.csv             - 径流POT摘要")
print(f"  6. pot_combined_summary.csv         - 综合对比表")
print(f"  7. tide_events_q*.csv               - 潮汐事件详情")
print(f"  8. flow_events_q*.csv               - 径流事件详情")

print("\n运行方式:")
print("  python -m src.main")
print("  或")
print("  python src/main.py")