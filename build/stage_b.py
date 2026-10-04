"""Stage B of the web asset build (heavy layers). Imports helpers from build_web_assets.py.
    python stage_b.py B1 B2 B3 B4 B5 B6
Display-only generalisation (simplify / fill small holes) is applied here; the research data and calculations are untouched."""
import sys, json, math, time, sqlite3
import numpy as np, pandas as pd, geopandas as gpd, shapely
from shapely.geometry import mapping, box
from shapely.ops import unary_union
from build_web_assets import *

DEPTH_ORDER = ["minder dan 0,5 m", "tussen 0,5 en 1,0 m", "tussen 1,0 en 1,5 m", "tussen 1,5 en 2,0 m", "tussen 2,0 en 5,0 m", "meer dan 5,0 m"]
TOL = {10: 8, 100: 10, 1000: 15, 10000: 20}


def _study():
    h = gpd.read_file(PROC / "buurt_analysis_env.gpkg")
    return h, h.geometry.union_all()


def stage_B1():
    """ROR modelled flooded areas (T=10..10000): display geometry (holes < 3,000 m2 filled, simplified)."""
    say("[B1] modelled flooded areas")
    for T in (10, 100, 1000, 10000):
        t0 = time.time()
        z = gpd.read_file(PROC / f"ldo_ror3_t{T}_overstroomde_gebieden.gpkg")
        tol = TOL[T]
        geoms = [fill_holes(g, 3000).simplify(tol, preserve_topology=True) for g in z.geometry]
        u = shapely.make_valid(unary_union([g for g in geoms if not g.is_empty])).simplify(tol, preserve_topology=True)
        write(f"ror_t{T}.json", fc(gpd.GeoDataFrame({"T": [T]}, geometry=[u], crs=RD), ["T"]))
        say(f"   T={T}: {time.time()-t0:.0f}s; display area {u.area/1e6:.1f} km2 (raw polygons {z.area.sum()/1e6:.1f} km2)")


def stage_B2():
    """ROR modelled maximum water depth classes, dissolved by class (display)."""
    say("[B2] modelled depth classes")
    for T in (10, 100, 1000, 10000):
        t0 = time.time()
        z = gpd.read_file(PROC / f"ldo_ror3_t{T}_maximale_waterdiepte.gpkg")
        col = "legenda" if "legenda" in z else "LEGENDA"
        z["k"] = z[col].map({c: i for i, c in enumerate(DEPTH_ORDER)})
        tol = TOL[T]
        z["geometry"] = [fill_holes(g, 3000).simplify(tol, preserve_topology=True) for g in z.geometry]
        rows = []
        for k, g in z.groupby("k"):
            u = shapely.make_valid(unary_union(g.geometry.values)).simplify(tol, preserve_topology=True)
            rows.append({"k": int(k), "geometry": u})
        write(f"ror_depth_t{T}.json", fc(gpd.GeoDataFrame(rows, crs=RD), ["k"]))
        say(f"   T={T}: {time.time()-t0:.0f}s")


def cap_flags(res):
    """Same rule as pipeline script 05b: footprint buffered 2 m intersects the modelled flooded-area polygons."""
    out = {}
    rb = res.copy(); rb["geometry"] = rb.geometry.buffer(2)
    for T in (10, 100, 1000, 10000):
        z = gpd.read_file(PROC / f"ldo_ror3_t{T}_overstroomde_gebieden.gpkg")[["geometry"]]
        hit = set(gpd.sjoin(rb[["uuid", "geometry"]], z, how="inner", predicate="intersects").uuid)
        out[T] = res.uuid.isin(hit).astype(int).values
    return out


def stage_B3():
    """BAG buildings within 150 m of the observed extent, classified; capture flags for the flooded residential buildings."""
    say("[B3] buildings")
    pand = gpd.read_file(PROC / "bag_pand_2021snap.gpkg")
    ext = gpd.read_file(PROC / "flood_obs_2021_all_parts.gpkg")
    zone = gpd.GeoDataFrame(geometry=ext.geometry.buffer(150).values, crs=RD)
    near = gpd.sjoin(pand[["uuid", "geometry"]], zone, how="inner", predicate="intersects").uuid.unique()
    p = pand[pand.uuid.isin(near)].copy().reset_index(drop=True)
    p["k"] = np.select([p.flood_all & p.residential, p.flood_all & ~p.residential, ~p.flood_all & p.residential], [2, 3, 1], 0)
    fl = p[p.k == 2].copy()
    caps = cap_flags(fl)
    for T in caps:
        p[f"t{T}"] = 0
        p.loc[p.uuid.isin(fl.uuid), f"t{T}"] = pd.Series(caps[T], index=fl.uuid.values).reindex(p.loc[p.uuid.isin(fl.uuid), "uuid"].values).values
    say(f"   buildings near flood: {len(p):,}; by class {p.k.value_counts().to_dict()}")
    p["by"] = p.bouwjaar.where(p.bouwjaar < 2100)
    p["nd"] = p.n_dwell.astype(int)
    p["geometry"] = p.geometry.simplify(0.4)
    write("bag_near.json", fc(p, ["k", "t10", "t100", "t1000", "t10000", "by", "nd"]))
    say("   flooded residential captured: " + str({T: int(caps[T].sum()) for T in caps}) + f" of {len(fl)}")


def stage_B4():
    """AHN4 DTM -> hillshade PNG (display raster) in WGS84 with bounds."""
    say("[B4] relief")
    import rasterio
    from rasterio.warp import calculate_default_transform, reproject, Resampling
    from PIL import Image
    src = rasterio.open(PROC / "ahn_dtm_10m.tif")
    a = src.read(1).astype("float32"); a[a == src.nodata] = np.nan
    dy, dx = np.gradient(a, 10.0, 10.0)
    slope = np.arctan(3.0 * np.hypot(dx, dy)); aspect = np.arctan2(-dx, dy)
    az, al = math.radians(315), math.radians(45)
    hs = np.clip(255 * (np.sin(al) * np.cos(slope) + np.cos(al) * np.sin(slope) * np.cos(az - aspect)), 0, 255)
    hs[np.isnan(a)] = np.nan
    tr, w, hgt = calculate_default_transform(src.crs, "EPSG:4326", src.width, src.height, *src.bounds, resolution=0.0003)
    out = np.full((hgt, w), np.nan, dtype="float32")
    reproject(hs, out, src_transform=src.transform, src_crs=src.crs, dst_transform=tr, dst_crs="EPSG:4326", resampling=Resampling.bilinear, src_nodata=np.nan, dst_nodata=np.nan)
    alpha = np.where(np.isnan(out), 0, 255).astype("uint8"); g = np.where(np.isnan(out), 0, out).astype("uint8")
    Image.merge("LA", (Image.fromarray(g), Image.fromarray(alpha))).save(SITE / "assets/relief.png", optimize=True)
    west, north = tr * (0, 0); east, south = tr * (w, hgt)
    (DATA / "relief.json").write_text(json.dumps({"bounds": [west, south, east, north], "size": [w, hgt]}), encoding="utf8")
    say(f"   relief.png {(SITE/'assets/relief.png').stat().st_size/1e6:.2f} MB, {w}x{hgt}")


GRP = {"Woongebied": "Residential", "Bedrijfsterrein": "Industry, retail, services", "Detailhandel en horeca": "Industry, retail, services", "Openbare voorziening": "Industry, retail, services",
       "Sociaal-culturele voorziening": "Industry, retail, services", "Hoofdweg": "Roads and rail", "Spoorterrein": "Roads and rail", "Semi-verhard overig terrein": "Roads and rail",
       "Overig agrarisch terrein": "Agriculture", "Glastuinbouw": "Agriculture", "Volkstuin": "Agriculture", "Bos": "Woods and nature", "Open droog natuurlijk terrein": "Woods and nature",
       "Open nat natuurlijk terrein": "Woods and nature", "Rijn & Maas": "Water", "Overig binnenwater": "Water", "Water met recreatieve functie": "Water", "Water met delfstofwinningsfunctie": "Water",
       "Park en plantsoen": "Parks, sport, recreation", "Sportterrein": "Parks, sport, recreation", "Dagrecreatief terrein": "Parks, sport, recreation", "Verblijfsrecreatief terrein": "Parks, sport, recreation",
       "Begraafplaats": "Parks, sport, recreation"}


def stage_B5():
    """BBG 2017 land-use groups, BRO soil classes, NWB roads, Top10NL water: clipped 300 m around the study buurten, simplified (display)."""
    say("[B5] land use / soil / roads / water")
    h, area = _study()
    clipg = area.buffer(300)
    clip = gpd.GeoDataFrame(geometry=[clipg], crs=RD)
    bb = tuple(clipg.bounds)
    t0 = time.time()
    bbg = gpd.read_file(PROC / "bbg2017.gpkg", bbox=bb)[["categorie", "geometry"]]
    bbg["g"] = bbg.categorie.map(GRP).fillna("Other")
    bbg["geometry"] = bbg.geometry.simplify(12, preserve_topology=True)
    b2 = gpd.overlay(bbg[["g", "geometry"]], clip, how="intersection").dissolve(by="g").reset_index()
    b2["geometry"] = b2.geometry.simplify(10, preserve_topology=True)
    write("bbg.json", fc(b2, ["g"])); say(f"   bbg {time.time()-t0:.0f}s; groups {b2.g.tolist()}")
    P = RAW / "bodem/BRO_DownloadBodemkaart.gpkg"
    soil = gpd.read_file(P, layer="soilarea", bbox=bb)
    con = sqlite3.connect(P)
    su = pd.read_sql("select * from soilarea_soilunit", con); cls = pd.read_sql("select code, mainsoilclassification from soil_units", con)
    su1 = su[su.soilunit_sequencenumber == su.groupby("maparea_id").soilunit_sequencenumber.transform("min")].drop_duplicates("maparea_id")
    soil = soil.merge(su1[["maparea_id", "soilunit_code"]], on="maparea_id", how="left").merge(cls, left_on="soilunit_code", right_on="code", how="left")
    soil["geometry"] = soil.geometry.simplify(15, preserve_topology=True)
    s2 = gpd.overlay(soil[["mainsoilclassification", "geometry"]].rename(columns={"mainsoilclassification": "c"}), clip, how="intersection").dissolve(by="c").reset_index()
    s2["geometry"] = s2.geometry.simplify(12, preserve_topology=True)
    write("soil.json", fc(s2, ["c"])); say(f"   soil classes {s2.c.tolist()}")
    nwb = gpd.read_file(PROC / "nwb_wegvakken.gpkg", bbox=bb)[["wegbehsrt", "geometry"]]
    nwb = nwb[nwb.intersects(clipg)].copy(); nwb["geometry"] = nwb.geometry.simplify(6)
    write("roads.json", fc(nwb.rename(columns={"wegbehsrt": "t"}), ["t"])); say(f"   roads {len(nwb):,}; types {nwb.wegbehsrt.value_counts().to_dict()}")
    w = gpd.read_file(PROC / "top10nl_water.gpkg", bbox=bb)[["typewater", "geometry"]]
    w = w[w.intersects(clipg)].copy(); w["geometry"] = w.geometry.simplify(4, preserve_topology=True)
    write("water.json", fc(w.rename(columns={"typewater": "t"}), ["t"]))


def stage_B6():
    """Small-multiple window around the most exposed buurt, in RD metres (the site draws it with a planar projection)."""
    say("[B6] analysis window")
    h = gpd.read_file(PROC / "buurt_analysis_env.gpkg")
    top = h.sort_values("popC", ascending=False).iloc[0]
    c = top.geometry.centroid; R = 1800
    wb = box(c.x - R, c.y - R, c.x + R, c.y + R)
    ext = gpd.read_file(PROC / "flood_obs_2021_all_parts.gpkg"); ext = ext[ext.intersects(wb)].copy()
    ext["geometry"] = ext.geometry.intersection(wb).simplify(2)
    pand = gpd.read_file(PROC / "bag_pand_2021snap.gpkg", bbox=wb.bounds)
    pand = pand[pand.intersects(wb)].reset_index(drop=True)
    fl = pand[pand.flood_all & pand.residential]
    caps = cap_flags(fl)
    capd = {T: set(fl.uuid[caps[T] == 1]) for T in caps}

    def poly(g, nd=1):
        m = mapping(g); return {"t": m["type"], "c": rnd(m["coordinates"], nd)}
    bld = []
    for r in pand.itertuples():
        k = 2 if (r.flood_all and r.residential) else 3 if r.flood_all else 1 if r.residential else 0
        b = {"k": k, "g": poly(r.geometry.simplify(0.5))}
        if k == 2:
            b["cap"] = [int(r.uuid in capd[T]) for T in (10, 100, 1000, 10000)]
        bld.append(b)
    win = {"center": [c.x, c.y], "R": R, "name": top.buurtnaam, "gem": top.gemeentenaam, "obs": [poly(g) for g in ext.dissolve().geometry], "bld": bld}
    for T in (10, 100, 1000, 10000):
        z = gpd.read_file(PROC / f"ldo_ror3_t{T}_overstroomde_gebieden.gpkg", bbox=wb.bounds)
        zz = [fill_holes(g.intersection(wb), 3000).simplify(2) for g in z.geometry if g.intersects(wb)]
        u = unary_union(zz)
        win[f"ror{T}"] = [] if u.is_empty else [poly(u)]
    (DATA / "window.json").write_text(json.dumps(win, separators=(",", ":")), encoding="utf8")
    say(f"   window.json {(DATA/'window.json').stat().st_size/1e6:.2f} MB; buildings {len(bld)}; flooded residential {len(fl)}; captured {({T: len(capd[T]) for T in capd})}")


if __name__ == "__main__":
    for st in sys.argv[1:]:
        globals()["stage_" + st]()
