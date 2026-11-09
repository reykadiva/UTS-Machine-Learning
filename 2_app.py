"""LANGKAH 2: Mesin pencari review Steam.
Jalankan: python -m streamlit run 2_app.py"""
import html
import os
import re
from collections import Counter

import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import linear_kernel

st.set_page_config(page_title="Steam Review Search", page_icon="🔎", layout="wide")

# Topik cepat: (label tombol, keyword)
TOPIK = [("🐛 Bug", "bug"), ("💥 Crash", "crash"), ("🐌 Lag / FPS", "stutter fps"),
         ("💰 Harga", "price"), ("🎮 Gameplay", "gameplay"), ("📖 Cerita", "story"),
         ("🎨 Grafis", "graphics"), ("🔊 Audio", "sound"), ("⚙️ Optimisasi", "optimization"),
         ("🤝 Multiplayer", "multiplayer")]

# Kata yang dianggap "mirip" dengan keyword (opsional)
SINONIM = {
    "bug": ["glitch", "crash", "broken", "freeze", "stutter"],
    "crash": ["freeze", "crashing", "bug"],
    "lag": ["stutter", "fps", "framerate"],
    "price": ["expensive", "overpriced", "money"],
}


# ---------- Preprocessing teks ----------
def kata(teks):
    """Pecah teks jadi kata: huruf kecil, buang simbol & stopword (the, is, and, ...)."""
    return [w for w in re.findall(r"[a-z]+", str(teks).lower())
            if w not in ENGLISH_STOP_WORDS and len(w) > 2]


def stem(w):
    """Stemming sederhana: bugs->bug, crashes->crash, glitched->glitch."""
    for suf in ("ing", "es", "ed", "s"):
        if w.endswith(suf) and len(w) - len(suf) >= 3 and not w.endswith("ss"):
            return w[:-len(suf)]
    return w


def analisis(teks):
    """Kata (di-stem) + pasangan 2 kata berurutan, mis. 'wall bug'."""
    t = [stem(w) for w in kata(teks)]
    return t + [a + " " + b for a, b in zip(t, t[1:])]


# ---------- Data & index pencarian ----------
@st.cache_data
def muat():
    d = pd.read_csv("steam_reviews.csv", dtype={"steamid": str}).dropna(subset=["review"])
    if "timestamp" in d.columns:
        d["tanggal"] = pd.to_datetime(d["timestamp"], unit="s")
    else:
        d["tanggal"] = pd.NaT
    return d


@st.cache_resource
def bangun_index(game):
    d = muat()
    d = d[d["game"] == game].reset_index(drop=True)
    vec = TfidfVectorizer(analyzer=analisis, sublinear_tf=True)  # teks -> angka (bobot kata)
    X = vec.fit_transform(d["review"])
    return d, vec, X


# ---------- Fungsi tampilan ----------
def sorot(teks, qstems):
    """Beri stabilo (<mark>) pada kata yang cocok dengan keyword."""
    out = []
    for bagian in re.split(r"([A-Za-z]+)", teks):
        aman = html.escape(bagian)
        if bagian.isalpha() and stem(bagian.lower()) in qstems:
            aman = f"<mark>{aman}</mark>"
        out.append(aman)
    return "".join(out).replace("\n", "<br>")


def cuplikan(teks, qstems, lebar=350):
    """Potong review panjang di sekitar kata keyword pertama."""
    if len(teks) <= lebar * 2:
        return teks
    for m in re.finditer(r"[A-Za-z]+", teks):
        if stem(m.group().lower()) in qstems:
            awal = max(0, m.start() - lebar // 2)
            akhir = min(len(teks), awal + lebar)
            return ("..." if awal > 0 else "") + teks[awal:akhir] + ("..." if akhir < len(teks) else "")
    return teks[:lebar] + "..."


def set_q(nilai):
    st.session_state.q = nilai


# ---------- Cek data ----------
if not os.path.exists("steam_reviews.csv"):
    st.error("File `steam_reviews.csv` belum ada. Jalankan dulu: `python 1_crawl.py`")
    st.stop()
data = muat()
jumlah = data.groupby("game").size()

# ---------- Sidebar: pengaturan ----------
with st.sidebar:
    st.header("⚙️ Pengaturan")
    game = st.selectbox("🎮 Game", sorted(jumlah.index),
                        format_func=lambda g: f"{g}  ({jumlah[g]:,} review)")
    sent = st.radio("Jenis review", ["Semua", "👍 Recommended", "👎 Not Recommended"])
    min_jam = st.slider("Minimal jam main saat review", 0, 500, 0,
                        help="Naikkan untuk hanya melihat review dari pemain yang sudah lama main.")
    opsi_urut = ["Paling relevan", "Jam main terbanyak"]
    if data["tanggal"].notna().any():
        opsi_urut.append("Terbaru")
    urut = st.selectbox("Urutkan hasil", opsi_urut)
    topn = st.slider("Jumlah review ditampilkan", 5, 50, 10)
    expand = st.checkbox("Cari juga kata yang mirip", True,
                         help="Contoh: 'bug' juga mencari glitch, crash, broken, freeze, stutter.")

# ---------- Halaman utama ----------
st.title("🔎 Cari Isi Review Game Steam")
st.caption("Cari apa yang dibicarakan pemain: bug, performa, cerita, harga, dan lain-lain. "
           "Ketik keyword **bahasa Inggris** atau klik topik cepat.")

if "q" not in st.session_state:
    st.session_state.q = "bug"

kolom = st.columns(len(TOPIK))
for k, (label, kw) in zip(kolom, TOPIK):
    k.button(label, on_click=set_q, args=(kw,), use_container_width=True)

q = st.text_input("Keyword", key="q", placeholder="contoh: wall bug, crash on startup, boss fight")

if not q.strip():
    st.info("👆 Ketik keyword atau klik salah satu topik cepat untuk mulai mencari.")
    st.stop()

d, vec, X = bangun_index(game)

q_kata = kata(q)
if expand:
    for w in list(q_kata):
        q_kata += SINONIM.get(stem(w), [])
qstems = {stem(w) for w in q_kata}

qv = vec.transform([" ".join(q_kata)])
if qv.nnz == 0:
    st.warning(f"Keyword **{q}** tidak ditemukan di review {game}. Coba kata lain, atau pakai bahasa Inggris.")
    st.stop()

skor = linear_kernel(qv, X).ravel()     # kemiripan keyword dengan tiap review (cosine similarity)

mask = (d["jam_main"] >= min_jam).to_numpy()
if sent.startswith("👍"):
    mask = mask & (d["recommended"] == 1).to_numpy()
elif sent.startswith("👎"):
    mask = mask & (d["recommended"] == 0).to_numpy()
cocok = (skor > 0) & mask

total, n = int(mask.sum()), int(cocok.sum())
if total == 0:
    st.info("Tidak ada review yang lolos filter di sidebar. Longgarkan filternya.")
    st.stop()
if n == 0:
    st.info(f"Tidak ada review {game} yang membahas **{q}** dengan filter ini.")
    st.stop()

hasil = d[cocok].assign(skor=skor[cocok])
if urut == "Jam main terbanyak":
    hasil = hasil.sort_values("jam_main", ascending=False)
elif urut == "Terbaru":
    hasil = hasil.sort_values("tanggal", ascending=False)
else:
    hasil = hasil.sort_values("skor", ascending=False)

# ---------- Ringkasan ----------
persen = n / total
persen_neg = (hasil["recommended"] == 0).mean()
dasar_neg = (d.loc[mask, "recommended"] == 0).mean()

st.subheader(f"Hasil untuk “{q}” di {game}")
m1, m2, m3 = st.columns(3)
m1.metric("Review yang membahas", f"{n:,} dari {total:,}")
m2.metric("Porsi dari semua review", f"{persen:.1%}")
m3.metric("Yang 👎 Not Recommended", f"{persen_neg:.0%}",
          delta=f"{(persen_neg - dasar_neg) * 100:+.0f} poin dari rata-rata game",
          delta_color="inverse",
          help="Dibandingkan dengan persentase 👎 di semua review game ini.")

if persen >= 0.15:
    st.error(f"Topik ini **sangat sering dibahas** ({persen:.0%} dari review).")
elif persen >= 0.05:
    st.warning(f"Topik ini **cukup sering dibahas** ({persen:.0%} dari review).")
else:
    st.success(f"Topik ini **jarang dibahas** ({persen:.0%} dari review).")
st.caption("Kategori kasar: ≥15% sangat sering, 5–15% cukup sering, <5% jarang.")

# ---------- Tab ----------
tab1, tab2, tab3 = st.tabs(["📋 Daftar review", "📊 Kata yang sering muncul bersamanya", "ℹ️ Cara kerja"])

with tab1:
    tampil = hasil.head(topn)
    a, b = st.columns([3, 1])
    a.write(f"Menampilkan **{len(tampil)}** dari {n:,} review yang cocok.")
    unduh = hasil[["steamid", "recommended", "jam_main", "jam_total", "review", "skor"]]
    b.download_button("⬇️ Unduh semua hasil (CSV)", unduh.to_csv(index=False).encode("utf-8"),
                      "hasil_pencarian.csv", "text/csv", use_container_width=True)

    for r in tampil.itertuples():
        label = ":green[**👍 Recommended**]" if r.recommended else ":red[**👎 Not Recommended**]"
        tgl = r.tanggal.strftime("%d %b %Y") if pd.notna(r.tanggal) else "-"
        ringkas = cuplikan(r.review, qstems)
        with st.container(border=True):
            st.markdown(f"{label} &nbsp;·&nbsp; ⏱ **{r.jam_main:.0f} jam** saat review "
                        f"(total {r.jam_total:.0f} jam) &nbsp;·&nbsp; 📅 {tgl}")
            st.markdown(sorot(ringkas, qstems), unsafe_allow_html=True)
            if ringkas != r.review:
                with st.expander("Baca selengkapnya"):
                    st.markdown(sorot(r.review, qstems), unsafe_allow_html=True)
            st.caption(f"Reviewer: [{r.steamid}](https://steamcommunity.com/profiles/{r.steamid})"
                       f" · skor relevansi {r.skor:.2f}")

with tab2:
    st.write("Kata-kata yang paling sering ikut muncul di review yang membahas keyword-mu. "
             "Berguna untuk tahu **detail masalahnya** (misal: bug-nya soal apa saja).")
    hitung = Counter()
    for t in hasil["review"]:
        hitung.update({w for w in kata(t) if stem(w) not in qstems})
    top = hitung.most_common(15)
    grafik = pd.DataFrame({"kata": [w for w, _ in top],
                           "% review": [c / n * 100 for _, c in top]}).set_index("kata")
    st.bar_chart(grafik)

with tab3:
    st.markdown("""
1. **Crawling**: review diambil dari API resmi Steam, lengkap dengan jam main dan 👍/👎.
2. **Preprocessing**: huruf kecil, buang simbol dan kata umum (the, is, and), lalu stemming sederhana
   (*bugs → bug*, *crashes → crash*).
3. **TF-IDF**: tiap review diubah menjadi angka. Kata yang jarang tapi khas diberi bobot lebih besar.
4. **Cosine similarity**: keyword yang kamu ketik diubah ke angka dengan cara yang sama, lalu dihitung
   kemiripannya dengan setiap review. Skor 0–1; makin tinggi makin relevan.
5. Review yang skornya > 0 ditampilkan, diurutkan sesuai pilihanmu.
""")
