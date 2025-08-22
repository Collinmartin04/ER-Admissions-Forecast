from pathlib import Path
import pandas as pd

def _find_header_index(df: pd.DataFrame, max_search_rows: int = 50) -> int:
    for i in range(min(max_search_rows, len(df))):
        first_cell = str(df.iloc[i, 0]).strip().lower()
        if first_cell == "period":
            return i
    raise ValueError("Could not find header row with 'Period' in first column.")

def _parse_period(x: str):
    x = str(x).strip()
    for fmt in ("%y-%b", "%Y-%b", "%b-%y", "%b-%Y"):
        try:
            return pd.to_datetime(x, format=fmt)
        except Exception:
            pass
    return pd.NaT

def _to_num(s):
    if pd.isna(s):
        return None
    s = str(s).replace(",", "").replace("%", "").strip()
    return pd.to_numeric(s, errors="coerce")

def load_national_ae_timeseries(csv_path: str | Path) -> pd.DataFrame:
    """
    Returns a tidy monthly dataframe with columns:
      - Date (monthly period as Timestamp at month start)
      - y (Total A&E attendances)
    Sorted by Date; non-positive or NaN removed.
    """
    raw = pd.read_csv(csv_path, header=None)
    hdr_idx = _find_header_index(raw)
    headers = raw.iloc[hdr_idx].tolist()
    df = raw.iloc[hdr_idx + 1:].copy()
    df.columns = headers

    # Identify the total attendances column
    cand_total_cols = [c for c in df.columns if isinstance(c, str)
                       and "total" in c.lower() and "attend" in c.lower()]
    if not cand_total_cols:
        raise ValueError(f"No 'Total Attendances' column found. Got: {df.columns.tolist()}")
    total_col = cand_total_cols[0]

    df["Date"] = df["Period"].map(_parse_period)
    df["y"] = df[total_col].map(_to_num)
    ts = df[["Date", "y"]].dropna().copy()
    ts = ts.sort_values("Date")
    ts = ts[ts["y"] > 0]
    # normalize to month start
    ts["Date"] = ts["Date"].values.astype("datetime64[M]")
    ts = ts.drop_duplicates(subset=["Date"])
    return ts.reset_index(drop=True)
