#!/usr/bin/env python3
"""
Skrip: fetch_and_save_evaluasi.py
Fungsi:
 - Memanggil /api/evaluasi-massal
 - Menyimpan response JSON ke evaluasi_full.json
 - Jika ada field detail_konsumen, simpan ke CSV/XLSX
Usage:
  python fetch_and_save_evaluasi.py --n 1000 --topk 10 --seed 42 --url http://127.0.0.1:5000/api/evaluasi-massal
"""
import argparse
import requests
import json
import pandas as pd
import sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--n", type=int, default=1000, help="Jumlah konsumen sintetik (n_konsumen)")
parser.add_argument("--topk", type=int, default=10, help="Top-K")
parser.add_argument("--seed", type=int, default=42, help="Random seed")
parser.add_argument("--url", type=str, default="http://127.0.0.1:5000/api/evaluasi-massal", help="URL endpoint")
parser.add_argument("--timeout", type=int, default=1800, help="Timeout in seconds")
args = parser.parse_args()

payload = {
    "n_konsumen": args.n,
    "top_k": args.topk,
    "seed": args.seed
}

out_dir = Path(".")
json_path = out_dir / "evaluasi_full.json"

print(f"Posting to {args.url} with payload: {payload} (timeout={args.timeout}s)")
try:
    r = requests.post(args.url, json=payload, timeout=args.timeout)
except Exception as e:
    print("Request failed:", e, file=sys.stderr)
    sys.exit(2)

if not r.ok:
    print("Server returned error:", r.status_code, r.text, file=sys.stderr)
    sys.exit(3)

data = r.json()
# Save raw JSON
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)
print(f"Saved JSON -> {json_path}")

# Save overall summary (if exists)
if "rata_rata_keseluruhan" in data:
    summary = data["rata_rata_keseluruhan"]
    pd.DataFrame([summary]).to_csv(out_dir / "evaluasi_summary.csv", index=False)
    pd.DataFrame([summary]).to_excel(out_dir / "evaluasi_summary.xlsx", index=False)
    print("Saved evaluasi_summary.csv/.xlsx")

# Save per_pekerjaan
if "per_pekerjaan" in data and data["per_pekerjaan"]:
    rows = []
    for k, v in data["per_pekerjaan"].items():
        row = {"pekerjaan": k}
        row.update(v)
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_dir / "evaluasi_by_pekerjaan.csv", index=False)
    pd.DataFrame(rows).to_excel(out_dir / "evaluasi_by_pekerjaan.xlsx", index=False)
    print("Saved evaluasi_by_pekerjaan.csv/.xlsx")

# Save detail_konsumen if present
if "detail_konsumen" in data and data["detail_konsumen"]:
    df = pd.DataFrame(data["detail_konsumen"])
    df.to_csv(out_dir / "evaluasi_detail_konsumen_sample.csv", index=False)
    df.to_excel(out_dir / "evaluasi_detail_konsumen_sample.xlsx", index=False)
    print("Saved evaluasi_detail_konsumen_sample.csv/.xlsx (this is the detail block returned by the server)")
else:
    print("No detail_konsumen field present in response or it's empty.")