"""Round-7 (3.7): per-building mask of hours that were interpolated/filled
in data preparation (raw BDG2 NaN after UTC->local conversion, incl. the DST
spring-forward hour), reproducing prepare_revision.py exactly up to the
interpolation step. Output: results/final_r7/interp_mask.csv (building_id, idx)."""
import os
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
B = os.path.join(ROOT, "data", "bdg2")
sel = pd.read_csv(os.path.join(B, "selected_v3", "_selection.txt"), sep="\t",
                  header=None, names=["building_id", "tag", "compl"])
meta = pd.read_csv(os.path.join(B, "metadata.csv")).set_index("building_id")
elec = pd.read_csv(os.path.join(B, "electricity_cleaned.csv"), parse_dates=["timestamp"],
                   usecols=["timestamp"] + sel["building_id"].tolist()).set_index("timestamp")
local_idx = pd.date_range("2016-01-01", "2016-12-31 23:00", freq="h")
rows = []
for bid in sel["building_id"]:
    s = elec[bid]
    s = s[(s.index >= "2016-01-01") & (s.index < "2017-01-01")]
    s = s.tz_localize("UTC").tz_convert(meta.loc[bid, "timezone"])
    s.index = s.index.tz_localize(None)
    s = s[~s.index.duplicated(keep="first")].reindex(local_idx)
    for i in s.index[s.isna()]:
        rows.append({"building_id": bid, "idx": local_idx.get_loc(i)})
out = pd.DataFrame(rows)
os.makedirs(os.path.join(ROOT, "results", "final_r7"), exist_ok=True)
out.to_csv(os.path.join(ROOT, "results", "final_r7", "interp_mask.csv"), index=False)
print(len(out), "filled hours;", out.groupby("building_id").size().describe().round(1).to_dict())
