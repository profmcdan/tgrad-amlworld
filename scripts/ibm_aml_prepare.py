"""IBM synthetic AML (Altman et al., NeurIPS 2023 D&B): parse <NAME>_Trans.csv and
<NAME>_Patterns.txt into compact integer arrays.

Run with an isolated interpreter (the data folder is untrusted input):

    python scripts/ibm_aml_prepare.py --name HI-Small

Output: data/ibm_aml_proc/<NAME>.npz (data/ is git-ignored)
  src, dst      int32 account ids (account = bank + account number)
  t             int32 minutes since the first transaction (sorted ascending)
  usd           float64 amount paid, converted to USD with rates implied by the data
  fmt, cur      int8 payment format / payment-currency codes
  xcur          bool, paid and received currencies differ
  y             int8 Is Laundering
  p_src, p_dst, p_t, p_type, p_attempt   transactions listed in the patterns file
  accounts, fmt_names, cur_names, pattern_names
"""
import argparse
import os

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PATTERNS = ["CYCLE", "FAN-IN", "FAN-OUT", "GATHER-SCATTER", "SCATTER-GATHER",
            "STACK", "BIPARTITE", "RANDOM"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="HI-Small")
    ap.add_argument("--data-dir", default=os.path.join(ROOT, "data", "ibm_aml"))
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "data", "ibm_aml_proc"))
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    dt = {"From Bank": str, "Account": str, "To Bank": str, "Account.1": str,
          "Receiving Currency": str, "Payment Currency": str, "Payment Format": str}
    df = pd.read_csv(os.path.join(a.data_dir, f"{a.name}_Trans.csv"), dtype=dt)
    ts = pd.to_datetime(df["Timestamp"], format="%Y/%m/%d %H:%M")
    t0 = ts.min()
    df["t"] = ((ts - t0).dt.total_seconds() // 60).astype(np.int64)
    df = df.sort_values("t", kind="stable").reset_index(drop=True)
    s = df["From Bank"] + "_" + df["Account"]
    d = df["To Bank"] + "_" + df["Account.1"]
    acc, inv = np.unique(np.concatenate([s.values, d.values]), return_inverse=True)
    n = len(df)
    src, dst = inv[:n].astype(np.int32), inv[n:].astype(np.int32)

    # USD rates implied by the data: median of (USD amount / other amount) over
    # transactions with US Dollar on exactly one side
    rate = {"US Dollar": 1.0}
    for c in pd.unique(pd.concat([df["Payment Currency"], df["Receiving Currency"]])):
        if c == "US Dollar":
            continue
        m1 = (df["Payment Currency"] == c) & (df["Receiving Currency"] == "US Dollar")
        m2 = (df["Receiving Currency"] == c) & (df["Payment Currency"] == "US Dollar")
        r = np.concatenate([(df.loc[m1, "Amount Received"] / df.loc[m1, "Amount Paid"]).values,
                            (df.loc[m2, "Amount Paid"] / df.loc[m2, "Amount Received"]).values])
        rate[c] = float(np.median(r)) if len(r) else np.nan
    usd = df["Amount Paid"].values * df["Payment Currency"].map(rate).values

    fmt_names = sorted(df["Payment Format"].unique())
    cur_names = sorted(rate)
    fmt = df["Payment Format"].map({k: i for i, k in enumerate(fmt_names)}).values.astype(np.int8)
    cur = df["Payment Currency"].map({k: i for i, k in enumerate(cur_names)}).values.astype(np.int8)
    xcur = (df["Payment Currency"] != df["Receiving Currency"]).values

    # patterns file: blocks "BEGIN LAUNDERING ATTEMPT - <TYPE>[: detail]" ... "END ..."
    idx = {k: i for i, k in enumerate(acc)}
    p_src, p_dst, p_t, p_type, p_att = [], [], [], [], []
    cur_type, att = None, -1
    miss = 0
    with open(os.path.join(a.data_dir, f"{a.name}_Patterns.txt")) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith("BEGIN LAUNDERING ATTEMPT"):
                ty = line.split("-", 1)[1].split(":")[0].strip()
                cur_type = PATTERNS.index(ty); att += 1
                continue
            if line.startswith("END LAUNDERING ATTEMPT"):
                cur_type = None
                continue
            if cur_type is None:
                continue
            f = line.split(",")
            ks, kd = f[1] + "_" + f[2], f[3] + "_" + f[4]
            if ks not in idx or kd not in idx:
                miss += 1
                continue
            tt = pd.Timestamp(f[0].replace("/", "-"))
            p_src.append(idx[ks]); p_dst.append(idx[kd])
            p_t.append(int((tt - t0).total_seconds() // 60)); p_type.append(cur_type); p_att.append(att)

    out = os.path.join(a.out_dir, f"{a.name}.npz")
    np.savez_compressed(out, src=src, dst=dst, t=df["t"].values.astype(np.int32), usd=usd,
                        fmt=fmt, cur=cur, xcur=xcur, y=df["Is Laundering"].values.astype(np.int8),
                        p_src=np.array(p_src, np.int32), p_dst=np.array(p_dst, np.int32),
                        p_t=np.array(p_t, np.int32), p_type=np.array(p_type, np.int8),
                        p_attempt=np.array(p_att, np.int32),
                        accounts=acc.astype(str), fmt_names=np.array(fmt_names), cur_names=np.array(cur_names),
                        pattern_names=np.array(PATTERNS), t0=str(t0))
    print(f"{a.name}: {n} transactions, {len(acc)} accounts, {int(df['Is Laundering'].sum())} laundering, "
          f"{att + 1} attempts / {len(p_src)} pattern transactions ({miss} unmatched) -> {out}")
    print("USD rates:", {k: round(v, 4) for k, v in rate.items()})


if __name__ == "__main__":
    main()
