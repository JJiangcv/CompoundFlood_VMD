# src/main.py
from pathlib import Path
import pandas as pd
import sys
try:
    # 方式1：作为模块运行 -> python -m src.main
    from . import io as cio
    from . import plot as cplots
except Exception:
    # 方式2：直接脚本运行 -> python src/main.py
    ROOT = Path(__file__).resolve().parents[1]  # 项目根：CompoundFlood_vmd/
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from src import io as cio
    from src import plot as cplots



data_dir = Path(__file__).resolve().parents[1] / "data"
output_dir = Path(__file__).resolve().parents[1] / "data"/"output"

rain   = cio.load_source_data(data_dir /"raw"/ "daily_rainfall_1981_2012.csv", date_col="timedate", value_col="Precipitation", dayfirst=True)
tide   = cio.load_source_data(data_dir /"raw"/ "tide_81_12_modified.csv", date_col="timedate", value_col="tide", dayfirst=True)
flow  = cio.load_source_data(data_dir /"raw"/ "flow_81_12.csv", date_col="timedate", value_col="Flow", dayfirst=True)
water_level  = cio.load_source_data(data_dir /"raw"/ "ct_waterlevel_81_12.csv", date_col="timedate", value_col="Water Level", dayfirst=True)

ds_map = {
        "rain":  rain,
        "tide":  tide,
        "flow": flow,
        "water level": water_level,
    }

df_all = cio.merge_source_data(ds_map)

out_png = str((output_dir / "climatology_plus_normalised.png"))
cplots.plot_climatology_plus_normalised(df_all, out_png, time_col="timedate",
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
                                        include_normalised=True)


