# src/main.py
from pathlib import Path
import pandas as pd
import sys
try:
    # 方式1：作为包运行 -> python -m src.main
    from . import io as cio
except Exception:
    # 方式2：直接脚本运行 -> python src/main.py
    ROOT = Path(__file__).resolve().parents[1]  # 项目根：CompoundFlood_vmd/
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from src import io as cio






data_dir = Path(__file__).resolve().parents[1] / "data"

rain   = cio.load_source_data(data_dir / "daily_rainfall_1981_2012.csv", date_col="timedate", value_col="Precipitation", dayfirst=True)
tide   = cio.load_source_data(data_dir / "tide_81_12_modified.csv", date_col="timedate", value_col="tide", dayfirst=True)
flow  = cio.load_source_data(data_dir / "flow_81_12.csv", date_col="timedate", value_col="Flow", dayfirst=True)
water_level  = cio.load_source_data(data_dir / "ct_waterlevel_81_12.csv", date_col="timedate", value_col="Water Level", dayfirst=True)

ds_map = {
        "rain":  rain,
        "tide":  tide,
        "flow": flow,
        "water level": water_level,
    }

df_all = cio.merge_source_data(ds_map)

print("=== 合并后前10行预览 ===")
print(df_all.head(10).to_string(index=False))

out = data_dir / "merged.csv"
df_all.to_csv(out, index=False)
print(f"\n已保存: {out}")


