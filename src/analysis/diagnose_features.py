import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

sys.path.append(str(Path(__file__).resolve().parents[1]))
from common import full_path, load_json


def main():
    x = np.load(full_path("cache/X_train_raw.npy"))
    langs = np.load(full_path("cache/train_languages.npy"), allow_pickle=True)
    feature_names = load_json(full_path("cache/feature_names.json"))["feature_names"]

    group_a = np.isin(langs, ["Python", "Java"])
    group_b = langs == "C++"

    rows = []
    for idx, name in enumerate(feature_names):
        a = x[group_a, idx]
        b = x[group_b, idx]
        ks_stat = float(ks_2samp(a, b).statistic)
        mean_train = float(a.mean())
        mean_ood = float(b.mean())
        zero_rate_ood = float((b == 0.0).mean())

        is_broken = bool((mean_train > 0.5) and (mean_ood == 0.0))
        is_unstable = bool(ks_stat > 0.2)

        rows.append({
            "feature_idx": idx,
            "feature_name": name,
            "ks_stat": ks_stat,
            "mean_train": mean_train,
            "mean_ood": mean_ood,
            "zero_rate_ood": zero_rate_ood,
            "is_broken": is_broken,
            "is_unstable": is_unstable,
        })

    report = pd.DataFrame(rows)
    out_dir = full_path("analysis")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "feature_health_report.csv"
    report.to_csv(out_path, index=False)

    print("saved", out_path)

    most_broken = report.sort_values(["zero_rate_ood", "ks_stat"], ascending=[False, False]).head(5)
    most_drifting = report.sort_values("ks_stat", ascending=False).head(5)

    print("top5_most_broken")
    print(most_broken[["feature_idx", "feature_name", "zero_rate_ood", "mean_train", "mean_ood", "is_broken"]].to_string(index=False))

    print("top5_most_drifting")
    print(most_drifting[["feature_idx", "feature_name", "ks_stat", "mean_train", "mean_ood", "is_unstable"]].to_string(index=False))


if __name__ == "__main__":
    main()
