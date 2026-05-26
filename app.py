"""
Backend API - Sistem Rekomendasi Properti
Menggunakan Flask + AHP + Profile Matching
+ Penyesuaian Wilayah: Provinsi → Kota → Kecamatan
+ Threshold Adaptif per Profesi
+ Dataset dinamis dari data_original.xlsx
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
import numpy as np
import os

app = Flask(__name__)
CORS(app)

# ===================== LOAD DATASET =====================

DATASET_PATH = "data_original.xlsx"

def load_dataset(filepath=DATASET_PATH):
    """Load dan preprocess dataset properti dari Excel."""
    if not os.path.exists(filepath):
        print(f"[ERROR] File '{filepath}' tidak ditemukan!")
        return pd.DataFrame()

    df = pd.read_excel(filepath)
    df.columns = df.columns.str.strip()

    # ✅ DEBUG: Cetak nama kolom asli agar mudah diagnosa
    print(f"[DEBUG] Kolom asli di Excel: {list(df.columns)}")

    rename_map = {}
    # Kolom yang harus dibuang (pre-computed score, tidak dipakai)
    drop_cols = [c for c in df.columns if c.lower().startswith("score_")]
    df = df.drop(columns=drop_cols, errors="ignore")
    print(f"[DEBUG] Kolom score di-drop: {drop_cols}")

    for col in df.columns:
        # Normalize: lowercase, hapus spasi/strip, ganti spasi dengan _
        cl = col.lower().strip().replace(" ", "_").replace("/", "_").replace("-", "_")

        # Harga
        if cl in ["price", "harga", "price_clean", "harga_clean", "harga_jual",
                  "harga_(rp)", "harga(rp)", "price_(rp)"]:
            rename_map[col] = "harga"

        # Luas bangunan
        elif any(x in cl for x in ["luas_bangunan", "building_size", "lb", "luas_b"]):
            rename_map[col] = "luas_bangunan"

        # Luas tanah
        elif any(x in cl for x in ["luas_tanah", "land_size", "lt", "luas_t"]):
            rename_map[col] = "luas_tanah"

        # Luas generik (fallback)
        elif cl in ["luas", "size", "luas_m2", "luas_(m2)"]:
            rename_map[col] = "luas_bangunan"

        # Kamar tidur
        elif any(x in cl for x in ["kamar_tidur", "bedroom", "bedrooms",
                                    "kamar_tidur_clean", "jml_kamar_tidur", "kt"]):
            rename_map[col] = "kamar_tidur"

        # Kamar mandi
        elif any(x in cl for x in ["kamar_mandi", "bathroom", "bathrooms",
                                    "kamar_mandi_clean", "jml_kamar_mandi", "km"]):
            rename_map[col] = "kamar_mandi"

        # Jenis properti
        elif any(x in cl for x in ["jenis_properti", "jenis", "tipe", "type",
                                    "kategori", "property_type", "tipe_properti"]):
            rename_map[col] = "jenis_properti"

        # Provinsi — paling kritis!
        elif any(x in cl for x in ["provinsi", "province", "prov"]):
            rename_map[col] = "provinsi"

        # Kota — juga kritis!
        elif any(x in cl for x in ["kota", "kota_kab", "city", "kabupaten",
                                    "kota_kabupaten", "kab", "kota_kab."]):
            rename_map[col] = "kota"

        # Kecamatan
        elif any(x in cl for x in ["kecamatan", "district", "kec", "sub_district"]):
            rename_map[col] = "kecamatan"

        # Nama / judul listing
        elif any(x in cl for x in ["nama", "judul", "title", "name",
                                    "listing", "nama_properti", "judul_iklan"]):
            rename_map[col] = "nama"

    print(f"[DEBUG] Rename map: {rename_map}")
    df = df.rename(columns=rename_map)

    for col in ["harga", "luas_bangunan", "luas_tanah", "kamar_tidur",
                "kamar_mandi", "jenis_properti", "provinsi", "kota", "kecamatan", "nama"]:
        if col not in df.columns:
            df[col] = None

    df["harga"]        = pd.to_numeric(df["harga"],        errors="coerce")
    df["luas_bangunan"]= pd.to_numeric(df["luas_bangunan"],errors="coerce")
    df["luas_tanah"]   = pd.to_numeric(df["luas_tanah"],   errors="coerce")
    df["kamar_tidur"]  = pd.to_numeric(df["kamar_tidur"],  errors="coerce")
    df["kamar_mandi"]  = pd.to_numeric(df["kamar_mandi"],  errors="coerce")

    df["provinsi"]      = df["provinsi"].astype(str).str.strip()
    df["kota"]          = df["kota"].astype(str).str.strip()
    df["kecamatan"]     = df["kecamatan"].astype(str).str.strip()
    df["jenis_properti"]= df["jenis_properti"].astype(str).str.strip()

    df = df.dropna(subset=["harga"])
    df = df[df["harga"] > 0]
    df = df[df["provinsi"].notna() & (df["provinsi"] != "nan") & (df["provinsi"] != "")]

    df["nama"] = df["nama"].fillna(
        df.apply(lambda r: f"{r['jenis_properti']} - {r['kota']}", axis=1)
    )

    df = df.reset_index(drop=True)
    df["id"] = df.index + 1

    print(f"[INFO] Dataset loaded: {len(df)} properti")
    print(f"[INFO] Provinsi: {sorted(df['provinsi'].unique())}")
    return df

df_properti = load_dataset()

# ===================== DYNAMIC MAPPING =====================

def build_map(df):
    """Bangun mapping provinsi → kota → kecamatan dari data asli."""
    peta = {}
    for _, row in df[["provinsi", "kota", "kecamatan"]].drop_duplicates().iterrows():
        prov = row["provinsi"]
        kota = row["kota"]
        kec  = row["kecamatan"]
        if not prov or prov == "nan":
            continue
        if prov not in peta:
            peta[prov] = {}
        if kota and kota != "nan":
            if kota not in peta[prov]:
                peta[prov][kota] = set()
            if kec and kec != "nan":
                peta[prov][kota].add(kec)
    for prov in peta:
        for kota in peta[prov]:
            peta[prov][kota] = sorted(peta[prov][kota])
    return peta

WILAYAH_MAP = build_map(df_properti) if not df_properti.empty else {}

# ===================== THRESHOLD ADAPTIF =====================

THRESHOLD_PROFESI = {
    "buruh":      2.5,
    "asn":        3.0,
    "pengusaha":  3.2,
    "umum":       2.8,
    "pns":        3.0,
    "swasta":     2.8,
    "wiraswasta": 3.0,
    "profesional":3.2,
    "freelancer": 2.7,
    "pensiunan":  2.6,
}

def get_threshold(pekerjaan):
    return THRESHOLD_PROFESI.get(str(pekerjaan).strip().lower(), 2.8)

# ===================== AHP WEIGHTS =====================

def get_weights_by_pekerjaan(pekerjaan):
    p = str(pekerjaan).strip().lower()
    presets = {
        "buruh":      [0.45, 0.10, 0.10, 0.05, 0.30],
        "asn":        [0.25, 0.20, 0.15, 0.10, 0.30],
        "pns":        [0.25, 0.20, 0.15, 0.10, 0.30],
        "pengusaha":  [0.10, 0.35, 0.25, 0.15, 0.15],
        "profesional":[0.15, 0.30, 0.25, 0.15, 0.15],
        "swasta":     [0.30, 0.20, 0.20, 0.10, 0.20],
        "wiraswasta": [0.20, 0.25, 0.20, 0.10, 0.25],
        "freelancer": [0.35, 0.20, 0.15, 0.10, 0.20],
        "pensiunan":  [0.40, 0.15, 0.15, 0.10, 0.20],
        "umum":       [0.35, 0.20, 0.15, 0.10, 0.20],
    }
    return presets.get(p, presets["umum"])

# ===================== SCORING / PROFILE MATCHING =====================

def score_harga(harga, budget_min, budget_max):
    mid = (budget_min + budget_max) / 2
    if mid <= 0:
        return 3.0
    r = harga / mid
    if r <= 0.7:
        return 5.0
    elif r <= 1.0:
        return 5.0 - (r - 0.7) * (2.0 / 0.3)
    elif r <= 1.3:
        return 3.0 - (r - 1.0) * (2.0 / 0.3)
    else:
        return max(1.0, 1.0 - (r - 1.3))

def score_luas(luas, pref_luas):
    if pref_luas <= 0 or pd.isna(luas):
        return 3.0
    r = luas / pref_luas
    if r >= 1.0:
        return min(5.0, 3.0 + (r - 1.0) * 2.0)
    return max(1.0, r * 3.0)

def score_gap(nilai, pref):
    gap = int(nilai or 0) - int(pref or 0)
    tabel = {0: 5.0, 1: 4.5, -1: 4.0, 2: 3.5, -2: 3.0, 3: 2.5, -3: 2.0}
    return tabel.get(gap, max(1.0, 2.0 - abs(gap) * 0.5))

def score_lokasi(row, pref_kota, pref_provinsi):
    if pref_kota and str(row.get("kota","")).strip().lower() == pref_kota.strip().lower():
        return 5.0
    if pref_provinsi and str(row.get("provinsi","")).strip().lower() == pref_provinsi.strip().lower():
        return 3.5
    return 2.0

def profile_matching(df_f, pref):
    weights = get_weights_by_pekerjaan(pref.get("pekerjaan", "umum"))
    results = []
    for _, row in df_f.iterrows():
        sh = score_harga(row["harga"], pref["budget_min"], pref["budget_max"])
        sl = score_luas(row["luas_bangunan"] or row["luas_tanah"], pref.get("luas_min", 0))
        sk = score_gap(row["kamar_tidur"], pref.get("kamar_tidur", 0))
        sm = score_gap(row["kamar_mandi"], pref.get("kamar_mandi", 0))
        so = score_lokasi(row, pref.get("kota",""), pref.get("provinsi",""))
        total = (weights[0]*sh + weights[1]*sl + weights[2]*sk +
                 weights[3]*sm + weights[4]*so)
        results.append({
            "id":           int(row["id"]),
            "nama":         str(row["nama"]),
            "harga":        int(row["harga"]),
            "luas_bangunan":float(row["luas_bangunan"] or 0),
            "luas_tanah":   float(row["luas_tanah"] or 0),
            "kamar_tidur":  int(row["kamar_tidur"] or 0),
            "kamar_mandi":  int(row["kamar_mandi"] or 0),
            "jenis_properti":str(row["jenis_properti"]),
            "provinsi":     str(row["provinsi"]),
            "kota":         str(row["kota"]),
            "kecamatan":    str(row["kecamatan"]),
            "skor_total":   round(total, 4),
            "skor":         round(total, 4),  # ✅ FIX: alias untuk index.html yang baca item.skor
            "detail_skor": {
                "harga":      round(sh, 2),
                "luas":       round(sl, 2),
                "kamar_tidur":round(sk, 2),
                "kamar_mandi":round(sm, 2),
                "lokasi":     round(so, 2)
            }
        })
    return sorted(results, key=lambda x: x["skor_total"], reverse=True)

# ===================== EVALUASI METRICS =====================

def hitung_metrik(rec_ids, rel_ids, k):
    rec = rec_ids[:k]
    rel_set = set(rel_ids)
    tp = sum(1 for i in rec if i in rel_set)
    precision = tp / k if k else 0
    recall    = tp / len(rel_set) if rel_set else 0
    f1        = (2*precision*recall/(precision+recall)) if (precision+recall) > 0 else 0
    dcg       = sum((1 if rec[i] in rel_set else 0)/np.log2(i+2) for i in range(len(rec)))
    idcg      = sum(1/np.log2(i+2) for i in range(min(len(rel_set), k)))
    ndcg      = dcg/idcg if idcg else 0
    mrr       = next((1/(i+1) for i, idx in enumerate(rec) if idx in rel_set), 0)
    return {"precision": round(precision,4), "recall": round(recall,4),
            "f1": round(f1,4), "ndcg": round(ndcg,4), "mrr": round(mrr,4)}

# ===================== API ENDPOINTS =====================

@app.route("/")
def index():
    return send_from_directory(".", "index.html")

# ✅ DEBUG ENDPOINT: Cek nama kolom asli di Excel via browser
@app.route("/api/debug-kolom", methods=["GET"])
def debug_kolom():
    if df_properti.empty:
        # Baca ulang untuk lihat kolom mentah
        if os.path.exists(DATASET_PATH):
            df_raw = pd.read_excel(DATASET_PATH)
            return jsonify({
                "status": "dataset_gagal_load",
                "kolom_asli": list(df_raw.columns),
                "total_baris": len(df_raw)
            })
        return jsonify({"status": "file_tidak_ditemukan", "path": DATASET_PATH})
    return jsonify({
        "status": "ok",
        "total_properti": len(df_properti),
        "kolom_setelah_rename": list(df_properti.columns),
        "provinsi_tersedia": sorted(df_properti["provinsi"].dropna().unique().tolist()),
        "sample_provinsi_5": df_properti["provinsi"].dropna().head(5).tolist()
    })

# ✅ FIX 1: Nama endpoint diubah dari /api/data-info → /api/dataset-info
# agar sesuai dengan fetch() di index.html
@app.route("/api/dataset-info", methods=["GET"])
def data_info():
    if df_properti.empty:
        return jsonify({"error": "Dataset tidak tersedia"}), 500
    return jsonify({
        "total_properti":  len(df_properti),
        "provinsi":        sorted(df_properti["provinsi"].dropna().unique().tolist()),
        "jenis_properti":  sorted(df_properti["jenis_properti"].dropna().unique().tolist()),
        # ✅ FIX: nama field disesuaikan dengan yang dibaca index.html
        "min_harga":       int(df_properti["harga"].min()),
        "max_harga":       int(df_properti["harga"].max()),
        "rata_rata_harga": int(df_properti["harga"].mean()),
    })

# ✅ FIX 2: Endpoint /api/wilayah — kembalikan daftar provinsi sebagai list
# agar index.html bisa populate dropdown <select id="provinsi">
@app.route("/api/wilayah", methods=["GET"])
def wilayah():
    return jsonify({
        "provinsi": sorted(WILAYAH_MAP.keys()),
        "map":      WILAYAH_MAP
    })

# ✅ FIX 3: Route /api/kota/<provinsi> — URL parameter di-decode agar
# nama provinsi dengan spasi (misal "Jawa Barat") tidak menjadi 404
@app.route("/api/kota/<path:provinsi>", methods=["GET"])
def get_kota(provinsi):
    prov_key = next((k for k in WILAYAH_MAP if k.lower() == provinsi.lower()), None)
    if not prov_key:
        return jsonify({"kota": []})
    return jsonify({"kota": sorted(WILAYAH_MAP[prov_key].keys())})

@app.route("/api/kecamatan/<path:provinsi>/<path:kota>", methods=["GET"])
def get_kecamatan(provinsi, kota):
    prov_key = next((k for k in WILAYAH_MAP if k.lower() == provinsi.lower()), None)
    if not prov_key:
        return jsonify({"kecamatan": []})
    kota_key = next((k for k in WILAYAH_MAP[prov_key] if k.lower() == kota.lower()), None)
    if not kota_key:
        return jsonify({"kecamatan": []})
    return jsonify({"kecamatan": WILAYAH_MAP[prov_key][kota_key]})

@app.route("/api/rekomendasi", methods=["POST"])
def rekomendasi():
    if df_properti.empty:
        return jsonify({"error": "Dataset tidak tersedia"}), 500

    data   = request.get_json() or {}
    errors = []

    if not data.get("budget_max"):
        errors.append("'budget_max' wajib diisi.")
    if not data.get("provinsi"):
        errors.append("'provinsi' wajib diisi.")
    if errors:
        return jsonify({"error": "Validasi gagal", "details": errors}), 400

    pref = {
        "budget_min":    int(data.get("budget_min", 0)),
        "budget_max":    int(data["budget_max"]),
        "luas_min":      float(data.get("luas_min", 0)),
        "kamar_tidur":   int(data.get("kamar_tidur", 0)),
        "kamar_mandi":   int(data.get("kamar_mandi", 0)),
        "jenis_properti":str(data.get("jenis_properti", "")),
        "provinsi":      str(data.get("provinsi", "")),
        "kota":          str(data.get("kota", "")),
        "kecamatan":     str(data.get("kecamatan", "")),
        "pekerjaan":     str(data.get("pekerjaan", "umum")),
    }
    top_k = int(data.get("top_k", 10))

    df_f = df_properti.copy()

    toleransi = {"buruh": 0.05, "asn": 0.15, "pengusaha": 0.30}.get(
        pref["pekerjaan"].lower(), 0.10)
    df_f = df_f[df_f["harga"] <= pref["budget_max"] * (1 + toleransi)]
    if pref["budget_min"] > 0:
        df_f = df_f[df_f["harga"] >= pref["budget_min"] * (1 - toleransi)]

    if pref["provinsi"]:
        df_f = df_f[df_f["provinsi"].str.lower() == pref["provinsi"].lower()]
    if pref["kota"]:
        df_kota = df_f[df_f["kota"].str.lower() == pref["kota"].lower()]
        if len(df_kota) >= 3:
            df_f = df_kota
    if pref["kecamatan"]:
        df_kec = df_f[df_f["kecamatan"].str.lower() == pref["kecamatan"].lower()]
        if len(df_kec) >= 3:
            df_f = df_kec

    if pref["luas_min"] > 0:
        df_f = df_f[
            (df_f["luas_bangunan"] >= pref["luas_min"]) |
            (df_f["luas_tanah"]    >= pref["luas_min"])
        ]
    if pref["kamar_tidur"] > 0:
        df_f = df_f[df_f["kamar_tidur"] >= pref["kamar_tidur"]]
    if pref["jenis_properti"]:
        df_jenis = df_f[df_f["jenis_properti"].str.lower() == pref["jenis_properti"].lower()]
        if len(df_jenis) >= 3:
            df_f = df_jenis

    if df_f.empty:
        return jsonify({
            "message": "Tidak ada properti yang sesuai kriteria.",
            "rekomendasi": [], "total_kandidat": 0
        })

    results   = profile_matching(df_f, pref)
    threshold = get_threshold(pref["pekerjaan"])
    top_results = [r for r in results if r["skor_total"] >= threshold][:top_k]
    if not top_results:
        top_results = results[:top_k]

    bobot = get_weights_by_pekerjaan(pref["pekerjaan"])
    return jsonify({
        "pekerjaan":      pref["pekerjaan"],
        "threshold_used": threshold,
        "bobot_ahp":      dict(zip(["harga","luas","kamar_tidur","kamar_mandi","lokasi"], bobot)),
        "total_kandidat": len(df_f),
        "total_rekomendasi": len(top_results),
        # ✅ FIX: tambah field message agar ditampilkan di index.html
        "message": f"Ditemukan {len(top_results)} rekomendasi dari {len(df_f)} kandidat properti.",
        "rekomendasi":    top_results
    })

@app.route("/api/evaluasi-massal", methods=["POST"])
def evaluasi_massal():
    if df_properti.empty:
        return jsonify({"error": "Dataset tidak tersedia"}), 500

    data          = request.get_json() or {}
    n             = int(data.get("jumlah_konsumen", 1000))
    top_k         = int(data.get("top_k", 10))
    pekerjaan_list= ["buruh", "asn", "pengusaha", "umum"]
    provinsi_list = df_properti["provinsi"].dropna().unique().tolist()

    all_metrics  = {p: [] for p in pekerjaan_list}
    harga_vals   = df_properti["harga"].dropna()
    harga_min_g  = harga_vals.min()
    harga_max_g  = harga_vals.max()

    for _ in range(n):
        pek      = np.random.choice(pekerjaan_list)
        provinsi = np.random.choice(provinsi_list)
        budget_max = int(np.random.uniform(harga_min_g, harga_max_g * 0.8))
        budget_min = int(budget_max * np.random.uniform(0.3, 0.7))
        luas_min   = int(np.random.uniform(20, 150))
        kamar      = int(np.random.choice([1, 2, 3, 4]))

        pref = {
            "budget_min": budget_min, "budget_max": budget_max,
            "luas_min": luas_min, "kamar_tidur": kamar, "kamar_mandi": 1,
            "jenis_properti": "", "provinsi": provinsi,
            "kota": "", "kecamatan": "", "pekerjaan": pek
        }

        toleransi = {"buruh": 0.05, "asn": 0.15, "pengusaha": 0.30}.get(pek, 0.10)
        df_f = df_properti[
            (df_properti["provinsi"].str.lower() == provinsi.lower()) &
            (df_properti["harga"] <= budget_max * (1 + toleransi)) &
            (df_properti["harga"] >= budget_min * (1 - toleransi))
        ]
        if len(df_f) < 2:
            continue

        results  = profile_matching(df_f, pref)
        threshold= get_threshold(pek)
        rec_ids  = [r["id"] for r in results[:top_k]]
        rel_ids  = [r["id"] for r in results if r["skor_total"] >= threshold]
        if not rel_ids:
            rel_ids = [r["id"] for r in results[:3]]

        all_metrics[pek].append(hitung_metrik(rec_ids, rel_ids, top_k))

    summary = {}
    for pek, ml in all_metrics.items():
        if not ml:
            summary[pek] = {"count": 0}
            continue
        summary[pek] = {
            "count":         len(ml),
            "avg_precision": round(np.mean([m["precision"] for m in ml]), 4),
            "avg_recall":    round(np.mean([m["recall"]    for m in ml]), 4),
            "avg_f1":        round(np.mean([m["f1"]        for m in ml]), 4),
            "avg_ndcg":      round(np.mean([m["ndcg"]      for m in ml]), 4),
            "avg_mrr":       round(np.mean([m["mrr"]       for m in ml]), 4),
        }

    all_m   = [m for ml in all_metrics.values() for m in ml]
    overall = {}
    if all_m:
        overall = {
            "total_evaluasi": len(all_m),
            "avg_precision":  round(np.mean([m["precision"] for m in all_m]), 4),
            "avg_recall":     round(np.mean([m["recall"]    for m in all_m]), 4),
            "avg_f1":         round(np.mean([m["f1"]        for m in all_m]), 4),
            "avg_ndcg":       round(np.mean([m["ndcg"]      for m in all_m]), 4),
            "avg_mrr":        round(np.mean([m["mrr"]       for m in all_m]), 4),
        }

    return jsonify({
        "jumlah_konsumen_target": n,
        "threshold_per_profesi":  THRESHOLD_PROFESI,
        "per_profesi": summary,
        "overall":     overall
    })

# ===================== RUN =====================
if __name__ == "__main__":
    print(f"[INFO] Total properti: {len(df_properti)}")
    print(f"[INFO] Threshold: {THRESHOLD_PROFESI}")
    app.run(debug=True, port=5000)