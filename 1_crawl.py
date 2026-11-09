"""LANGKAH 1: CRAWLING review Steam (API resmi, tanpa login/API key).
- Mengambil 20 game. Hasil disimpan per game, jadi kalau terhenti di tengah
  jalan, jalankan lagi saja: game yang sudah selesai akan dilewati."""
import json
import os
import time
import requests
import pandas as pd

URL = "https://api.steampowered.com/IUserReviewsService/GetAppReviews/v1/"
FILE = "steam_reviews.csv"
HALAMAN = 10   # 10 halaman x 100 review = maks 1000 review per game

# appid : nama game  (appid ada di URL halaman Steam: store.steampowered.com/app/<appid>/)
GAMES = {
    1245620: "Elden Ring",
    1091500: "Cyberpunk 2077",
    1086940: "Baldur's Gate 3",
    271590: "GTA V",
    730: "Counter-Strike 2",
    570: "Dota 2",
    578080: "PUBG: Battlegrounds",
    1172470: "Apex Legends",
    1174180: "Red Dead Redemption 2",
    292030: "The Witcher 3",
    990080: "Hogwarts Legacy",
    1623730: "Palworld",
    553850: "Helldivers 2",
    2358720: "Black Myth: Wukong",
    2246340: "Monster Hunter Wilds",
    413150: "Stardew Valley",
    367520: "Hollow Knight",
    1966720: "Lethal Company",
    489830: "Skyrim Special Edition",
    1716740: "Starfield",
}


def ambil_halaman(appid, cursor):
    """Ambil 1 halaman review. Return None kalau gagal."""
    param = {"appid": appid, "filter": 1, "languages": ["english"],
             "num_per_page": 100, "cursor": cursor, "purchase_type": 1}
    for _ in range(3):
        try:
            resp = requests.get(URL, params={"input_json": json.dumps(param)}, timeout=20)
            if resp.status_code == 429:
                print("  Terlalu cepat (429), tunggu 30 detik...")
                time.sleep(30)
                continue
            if resp.status_code != 200:
                print(f"  Gagal (status {resp.status_code}):", resp.text[:150])
                return None
            return resp.json().get("response", {})
        except ValueError:
            print("  Balasan bukan JSON:", resp.text[:150])
            return None
        except requests.RequestException as e:
            print("  Masalah koneksi:", e)
            time.sleep(5)
    return None


def main():
    # lanjutkan dari hasil sebelumnya (kalau ada dan formatnya sesuai)
    lama = pd.DataFrame()
    if os.path.exists(FILE):
        lama = pd.read_csv(FILE, dtype={"steamid": str})
        if "timestamp" not in lama.columns:
            print("File lama formatnya beda, diabaikan (akan ditimpa).")
            lama = pd.DataFrame()
    selesai = set(lama["game"]) if len(lama) else set()

    hasil_semua = [lama] if len(lama) else []
    for no, (appid, nama) in enumerate(GAMES.items(), 1):
        if nama in selesai:
            print(f"[{no}/{len(GAMES)}] {nama}: sudah ada, dilewati")
            continue
        print(f"[{no}/{len(GAMES)}] {nama} ...")
        baris, cursor = [], "*"
        for _ in range(HALAMAN):
            r = ambil_halaman(appid, cursor)
            if not r or not r.get("reviews"):
                break
            for rev in r["reviews"]:
                a = rev["author"]
                baris.append({
                    "game": nama,
                    "steamid": str(a.get("steamid", "")),            # identitas reviewer
                    "review": rev["review"],                          # teks review
                    "jam_main": a.get("playtime_at_review", 0) / 60,  # jam saat menulis review
                    "jam_total": a.get("playtime_forever", 0) / 60,   # jam total sampai sekarang
                    "recommended": int(rev["voted_up"]),              # 1=👍, 0=👎
                    "timestamp": rev.get("timestamp_created", 0),     # waktu review dibuat
                })
            cursor = r.get("cursor")
            if not cursor:
                break
            time.sleep(1.5)  # sopan ke server
        print(f"    -> {len(baris)} review")
        if baris:
            hasil_semua.append(pd.DataFrame(baris))
            df = pd.concat(hasil_semua, ignore_index=True)
            df = df[df["review"].str.strip() != ""].drop_duplicates(subset="review")
            df.to_csv(FILE, index=False)   # simpan tiap selesai 1 game

    if not hasil_semua:
        print("Tidak ada data yang terambil. Cek pesan error di atas.")
        return
    df = pd.read_csv(FILE)
    print("\nSelesai. Total review:", len(df))
    print(df.groupby("game").size().sort_values(ascending=False))


if __name__ == "__main__":
    main()
