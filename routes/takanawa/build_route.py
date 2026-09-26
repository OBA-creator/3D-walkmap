r"""経由地CSV(waypoints.csv)から、ビューア用の route.geojson を作る。

使い方(PowerShell):
    python build_route.py routes\takanawa --hint 東京都

流れ(2回に分けて実行します):
  1回目: 座標が空の地点を、名前からOSMの検索(Nominatim)で埋め、CSVに書き戻して止まる。
         表示される地図リンクで位置を確認し、違えばCSVの lat/lon を直接書き換える。
  2回目: 全地点に座標があれば、ルートを作って route.geojson に書き出す。
         徒歩の区間 = OSMの歩行者向け道路網で最短経路  /  電車の区間 = 経由点を結ぶ直線

CSVの列: order, name, category, mode, lat, lon, query, note
  mode     その地点に「どうやって着いたか」 walk(徒歩・既定) / train(電車)。最初の行は空欄でよい
  category via にすると、ラベルを出さない経由点になる(電車の通過駅など)
  query    名前で検索しても当たらないときの検索語(空なら name を使う)
"""
import argparse
import csv
import json
import math
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
from pyproj import Transformer
from shapely.geometry import LineString, box

USER_AGENT = "machiaruki-3d-route/0.1 (personal hobby project)"
MODES = ("walk", "train")


# ---------- CSV ----------
def read_waypoints(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    for need in ("order", "name", "category", "mode", "lat", "lon", "query", "note"):
        if need not in fields:
            fields.append(need)
    for r in rows:
        for k in fields:
            r[k] = (r.get(k) or "").strip()
    rows.sort(key=lambda r: int(r["order"]))
    return fields, rows


def write_waypoints(path, fields, rows):
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def has_coord(r):
    try:
        float(r["lat"]), float(r["lon"])
        return True
    except (ValueError, TypeError):
        return False


# ---------- 地名 → 座標 ----------
def geocode_one(query, hint=""):
    """OSM Nominatim で1件検索。(lat, lon, 表示名) か None を返す。"""
    q = f"{query} {hint}".strip()
    params = {"q": q, "format": "jsonv2", "limit": 1, "countrycodes": "jp", "accept-language": "ja"}
    url = "https://nominatim.openstreetmap.org/search?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=30) as res:
        data = json.load(res)
    if not data:
        return None
    d = data[0]
    return float(d["lat"]), float(d["lon"]), d.get("display_name", "")


def fill_coordinates(rows, hint, geocoder):
    filled, failed = [], []
    first = True
    for r in rows:
        if has_coord(r):
            continue
        if not first:
            time.sleep(1.1)  # Nominatim の利用ルール: 1秒に1回まで
        first = False
        q = r["query"] or r["name"]
        try:
            res = geocoder(q, hint)
        except Exception as e:  # 通信エラーなど
            print(f"  ! {r['name']}: 検索に失敗しました ({e})")
            res = None
        if res is None:
            failed.append(r)
            continue
        r["lat"], r["lon"] = f"{res[0]:.6f}", f"{res[1]:.6f}"
        filled.append((r, res[2]))
    return filled, failed


def print_table(rows):
    print("\n順  名前 / 座標 / 地図で確認")
    for r in rows:
        if has_coord(r):
            print(f"{r['order']:>2}  {r['name']}  ({r['lat']}, {r['lon']})")
            print(f"     https://www.google.com/maps?q={r['lat']},{r['lon']}")
        else:
            print(f"{r['order']:>2}  {r['name']}  (座標なし)")


# ---------- 歩行者ルート ----------
def build_walk_graph(points, margin_deg=0.006):
    """points=[(lon,lat),...] を囲む範囲の歩行者向け道路網をOSMから取得する。"""
    try:
        import osmnx as ox
    except ImportError:
        raise SystemExit("osmnx が入っていません。 pip install osmnx を実行してから、もう一度実行してください。")
    xs, ys = [p[0] for p in points], [p[1] for p in points]
    area = box(min(xs) - margin_deg, min(ys) - margin_deg, max(xs) + margin_deg, max(ys) + margin_deg)
    print("道路網をOSMから取得中... (少し時間がかかります)")
    return ox.graph_from_polygon(area, network_type="walk")


class NodeIndex:
    """グラフの節点から、最も近いものを探す(総当たり。数万点なら十分速い)。"""

    def __init__(self, G):
        self.ids = list(G.nodes)
        self.xs = np.array([G.nodes[n]["x"] for n in self.ids])
        self.ys = np.array([G.nodes[n]["y"] for n in self.ids])

    def nearest(self, lon, lat):
        k = math.cos(math.radians(lat))
        d = ((self.xs - lon) * k) ** 2 + (self.ys - lat) ** 2
        return self.ids[int(np.argmin(d))]


def _d2(a, b):
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


def _dedupe(coords):
    out = [coords[0]]
    for c in coords[1:]:
        if c != out[-1]:
            out.append(c)
    return out


def walk_line(G, idx, a, b):
    """a, b = (lon, lat)。道路網の最短経路に沿った座標列を返す(見つからなければ None)。"""
    import networkx as nx

    u, v = idx.nearest(*a), idx.nearest(*b)
    try:
        nodes = nx.shortest_path(G, u, v, weight="length")
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None
    coords = [a]
    for n1, n2 in zip(nodes, nodes[1:]):
        edge = min(G.get_edge_data(n1, n2).values(), key=lambda d: d.get("length", 0))
        p1 = (G.nodes[n1]["x"], G.nodes[n1]["y"])
        p2 = (G.nodes[n2]["x"], G.nodes[n2]["y"])
        geom = edge.get("geometry")
        pts = [tuple(c) for c in geom.coords] if geom is not None else [p1, p2]
        if _d2(pts[0], p1) > _d2(pts[-1], p1):  # 向きをそろえる
            pts = pts[::-1]
        coords.extend(pts)
    coords.append(b)
    return _dedupe(coords)


# ---------- 距離 ----------
def make_meter_transformer(lon, lat):
    zone = int((lon + 180) // 6) + 1
    return Transformer.from_crs(4326, (32600 if lat >= 0 else 32700) + zone, always_xy=True)


def length_m(coords, tf):
    xs, ys = tf.transform([c[0] for c in coords], [c[1] for c in coords])
    return float(LineString(list(zip(xs, ys))).length)


# ---------- 出力 ----------
def write_route(path, name, legs, spots, total_m, by_mode):
    lons = [c[0] for leg in legs for c in leg["coords"]]
    lats = [c[1] for leg in legs for c in leg["coords"]]
    meta = {"name": name, "total_m": round(total_m), "by_mode_m": {k: round(v) for k, v in by_mode.items()},
            "bounds": [round(min(lons), 6), round(min(lats), 6), round(max(lons), 6), round(max(lats), 6)]}
    feats = []
    for leg in legs:
        feats.append({"type": "Feature",
                      "properties": {"kind": "leg", "mode": leg["mode"], "from": leg["from"], "to": leg["to"],
                                     "d0": round(leg["d0"]), "d1": round(leg["d1"]), "approx": leg["approx"]},
                      "geometry": {"type": "LineString",
                                   "coordinates": [[round(x, 6), round(y, 6)] for x, y in leg["coords"]]}})
    for s in spots:
        props = {"kind": "spot", "name": s["name"], "category": s["category"], "note": s["note"],
                 "d": round(s["d"]), "show": s["category"] != "via"}
        for k in ("desc", "photo", "credit", "dir"):   # 詳細シートやラベルの向き用(あれば)
            if s.get(k):
                props[k] = s[k]
        feats.append({"type": "Feature", "properties": props,
                      "geometry": {"type": "Point", "coordinates": [round(s["lon"], 6), round(s["lat"], 6)]}})
    with open(path, "w", encoding="utf-8") as f:
        f.write('{"type":"FeatureCollection","meta":' + json.dumps(meta, ensure_ascii=False, separators=(",", ":")))
        f.write(',"features":[\n')
        f.write(",\n".join(json.dumps(x, ensure_ascii=False, separators=(",", ":")) for x in feats))
        f.write("\n]}\n")


# ---------- 本体 ----------
def main(argv=None, geocoder=geocode_one, graph_builder=build_walk_graph):
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", type=Path, help="ルートのフォルダ(waypoints.csv が入っているところ)")
    ap.add_argument("--hint", default="", help="検索のときに名前へ足す語(例: 東京都)")
    ap.add_argument("--out", type=Path, default=None, help="出力先(既定: <folder>/route.geojson)")
    args = ap.parse_args(argv)

    csv_path = args.folder / "waypoints.csv"
    out_path = args.out or args.folder / "route.geojson"
    if not csv_path.exists():
        print("waypoints.csv が見つかりません:", csv_path)
        return 1

    fields, rows = read_waypoints(csv_path)
    if len(rows) < 2:
        print("経由地が2つ以上必要です。")
        return 1
    for r in rows[1:]:
        if r["mode"] == "":
            r["mode"] = "walk"
        if r["mode"] not in MODES:
            print(f"{r['order']} {r['name']}: mode は {MODES} のどれかにしてください(今は '{r['mode']}')")
            return 1

    # --- 1回目: 座標を埋めて止まる ---
    if any(not has_coord(r) for r in rows):
        filled, failed = fill_coordinates(rows, args.hint, geocoder)
        write_waypoints(csv_path, fields, rows)
        for r, disp in filled:
            print(f"  {r['name']} → {r['lat']}, {r['lon']}   [{disp[:60]}]")
        print_table(rows)
        if failed:
            print("\n見つからなかった地点:", "、".join(r["name"] for r in failed))
            print("CSVの query 列を変えて再実行するか、lat/lon を直接入力してください。")
            return 1
        print("\n座標を CSV に書き込みました。地図リンクで位置を確認し、違う地点は lat/lon を直してから、")
        print("もう一度同じコマンドを実行してください。")
        return 0

    # --- 2回目: ルートを作る ---
    pts = [(float(r["lon"]), float(r["lat"])) for r in rows]
    walk_pts = []
    for i in range(1, len(rows)):
        if rows[i]["mode"] == "walk":
            walk_pts += [pts[i - 1], pts[i]]
    G = idx = None
    if walk_pts:
        G = graph_builder(walk_pts)
        idx = NodeIndex(G)

    tf = make_meter_transformer(*pts[0])
    legs, spots, d = [], [], 0.0
    spots.append({**_spot(rows[0], pts[0]), "d": 0.0})
    for i in range(1, len(rows)):
        a, b, mode = pts[i - 1], pts[i], rows[i]["mode"]
        approx = False
        if mode == "walk":
            coords = walk_line(G, idx, a, b)
            if coords is None:
                print(f"  ! {rows[i - 1]['name']} → {rows[i]['name']}: 徒歩の経路が見つからず、直線にしました。経由地を足してください。")
                coords, approx = [a, b], True
        else:
            coords = [a, b]
        L = length_m(coords, tf)
        legs.append({"mode": mode, "from": rows[i - 1]["name"], "to": rows[i]["name"],
                     "d0": d, "d1": d + L, "coords": coords, "approx": approx})
        d += L
        spots.append({**_spot(rows[i], b), "d": d})

    by_mode = {m: sum(l["d1"] - l["d0"] for l in legs if l["mode"] == m) for m in MODES}
    args.folder.mkdir(parents=True, exist_ok=True)
    write_route(out_path, args.folder.name, legs, spots, d, by_mode)

    print("\n区間                                   種別   距離(m)")
    for l in legs:
        flag = " (直線)" if l["approx"] else ""
        print(f"{l['from']} → {l['to']}".ljust(36), l["mode"].ljust(6), f"{l['d1'] - l['d0']:>7.0f}{flag}")
    print(f"\n合計 {d:.0f} m  (徒歩 {by_mode['walk']:.0f} m / 電車 {by_mode['train']:.0f} m)")
    print("書き出し:", out_path)
    return 0


def _spot(r, p):
    return {"name": r["name"], "category": r["category"] or "spot", "note": r["note"], "lon": p[0], "lat": p[1]}


if __name__ == "__main__":
    raise SystemExit(main())
