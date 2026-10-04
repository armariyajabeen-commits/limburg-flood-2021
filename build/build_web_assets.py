"""Derived WEB assets for the Netherlands research site. Reads the verified project outputs in ../../nl_verification and writes
browser-sized files to ../data. It does not change any research data or calculation: every value is copied from, or recomputed with the same
rule as, the pipeline scripts 01-10. Simplification/hole-filling here is for DISPLAY only and is stated on the site.

    python build_web_assets.py A        # buurten, cells, grid, observed extent, facts
    python build_web_assets.py B        # modelled hazard, buildings, relief, land use, soil, roads, water, analysis window
"""
import sys, json, re, math, pathlib, time
import numpy as np, pandas as pd, geopandas as gpd, shapely
from shapely.geometry import mapping, box
from shapely.ops import unary_union

HERE = pathlib.Path(__file__).resolve().parent
SITE = HERE.parent
PROJ = SITE.parent / "nl_verification"
PROC, OUT, LOG, RAW = PROJ / "data/processed", PROJ / "outputs", PROJ / "logs", PROJ / "data/raw"
DATA = SITE / "data"; DATA.mkdir(exist_ok=True)
RD = "EPSG:28992"


def say(m):
    print(m, flush=True)


def rnd(o, nd=5):
    if isinstance(o, (list, tuple)):
        return [rnd(x, nd) for x in o]
    return round(o, nd)


def fc(gdf, props=None, nd=5):
    """GeoJSON FeatureCollection (WGS84) with rounded coordinates and only the listed properties."""
    g = gdf.to_crs(4326)
    feats = []
    for i, (geom, row) in enumerate(zip(g.geometry, gdf.drop(columns="geometry").to_dict("records"))):
        if geom is None or geom.is_empty:
            continue
        p = {k: row[k] for k in (props or row.keys())}
        p = {k: (None if (isinstance(v, float) and (math.isnan(v) or math.isinf(v))) else (v.item() if hasattr(v, "item") else v)) for k, v in p.items()}
        m = mapping(geom)
        feats.append({"type": "Feature", "id": i, "properties": p, "geometry": {"type": m["type"], "coordinates": rnd(m["coordinates"], nd)}})
    return {"type": "FeatureCollection", "features": feats}


def write(name, obj):
    f = DATA / name
    f.write_text(json.dumps(obj, separators=(",", ":"), allow_nan=False), encoding="utf8")
    say(f"  {name}: {f.stat().st_size/1e6:.2f} MB")


def fill_holes(geom, min_area):
    """Display only: drop interior rings smaller than min_area m2 (the modelled polygons have holes at building footprints)."""
    def one(p):
        return shapely.Polygon(p.exterior, [r for r in p.interiors if shapely.Polygon(r).area >= min_area])
    if geom.geom_type == "Polygon":
        return one(geom)
    if geom.geom_type == "MultiPolygon":
        return shapely.MultiPolygon([one(p) for p in geom.geoms])
    return geom


def parse_logs():
    """Numbers that only exist in the pipeline logs are parsed from them (so the site can cite the log)."""
    t = (LOG / "run.log").read_text(encoding="utf8") + "\n" + (LOG / "07.out").read_text(encoding="utf8") + "\n" + (LOG / "08.out").read_text(encoding="utf8") + "\n" + (LOG / "06.out").read_text(encoding="utf8")
    f = {}
    def grab(key, pat, cast=float, grp=1, src="logs"):
        m = re.search(pat, t)
        if m:
            f[key] = {"v": cast(m.group(grp).replace(",", "")), "src": src}
        else:
            say(f"  !! facts: no match for {key}")
    grab("gwr_bw", r"GWR \(Gaussian[^)]*\): bw=(\d+)", int, src="logs/08.out")
    grab("gwr_r2", r"GWR \(Gaussian[^)]*\): bw=[\d.]+ neighbours of n=\d+; R2=([\d.]+)", src="logs/08.out")
    grab("gwr_adjr2", r"; adjR2=([\d.]+); AICc=([\d.]+); max\|coef", src="logs/08.out")
    grab("gwr_aicc", r"GWR \(Gaussian[^)]*\).*?AICc=([\d.]+)", src="logs/08.out")
    grab("gwr_skew", r"GWR residual skewness ([\d.\-]+)", src="logs/08.out")
    grab("gwr_kurt", r"GWR residual skewness [\d.\-]+, kurtosis ([\d.\-]+)", src="logs/08.out")
    grab("ols_r2", r"OLS R2=([\d.]+), adjR2=([\d.]+)", src="logs/07.out")
    grab("ols_adjr2", r"OLS R2=[\d.]+, adjR2=([\d.]+)", src="logs/07.out")
    grab("mgwr_r2", r"MGWR: bandwidths \[[^\]]*\]; R2=([\d.]+)", src="logs/08.out")
    grab("mgwr_adjr2", r"MGWR: bandwidths \[[^\]]*\]; R2=[\d.]+; adjR2=([\d.]+)", src="logs/08.out")
    grab("mgwr_aicc", r"MGWR: bandwidths \[[^\]]*\]; R2=[\d.]+; adjR2=[\d.]+; AICc=([\d.]+)", src="logs/08.out")
    m = re.search(r"MGWR: bandwidths \[([^\]]*)\]", t)
    f["mgwr_bws"] = {"v": [int(x) for x in m.group(1).split(",")], "src": "logs/08.out"}
    grab("moran_resp", r"Moran's I of response: I=([\-\d.]+), p=([\d.]+)", src="logs/07.out")
    grab("moran_resp_p", r"Moran's I of response: I=[\-\d.]+, p=([\d.]+)", src="logs/07.out")
    grab("moran_resid", r"Moran's I of OLS residuals=([\-\d.]+)", src="logs/07.out")
    grab("fraclogit_dev", r"fractional-logit deviance explained: ([\d.]+)", src="logs/07.out")
    grab("n_complete", r"complete cases for response \+ \d+ predictors: (\d+) of (\d+)", int, src="logs/07.out")
    grab("n_response", r"complete cases for response \+ \d+ predictors: \d+ of (\d+)", int, src="logs/07.out")
    grab("logit_pseudo_r2", r"logit\(any exposure\): pseudo-R2 ([\d.]+)", src="logs/08.out")
    grab("logit_events", r"events (\d+) of (\d+)", int, src="logs/08.out")
    grab("zero_share_complete", r"modelling sample: n=\d+; response zero ([\d.]+)%", src="logs/08.out")
    grab("nonzero_complete", r"; nonzero (\d+);", int, src="logs/08.out")
    grab("bag_cbs_corr", r"corr ([\d.]+)\n", src="logs/06.out")
    grab("bag_cbs_exact", r"exact match ([\d.]+)%", src="logs/06.out")
    grab("bag_cbs_pm1", r"within \+-1 ([\d.]+)%", src="logs/06.out")
    grab("bag_sum", r"sum BAG ([\d,]+) vs sum CBS ([\d,]+)", lambda s: int(s), src="logs/06.out")
    grab("cbs_sum", r"sum BAG [\d,]+ vs sum CBS ([\d,]+)", lambda s: int(s), src="logs/06.out")
    grab("pand_total", r"pand: ([\d,]+) total -> ([\d,]+) in approx", int, src="logs/run.log")
    grab("pand_snap", r"pand: [\d,]+ total -> ([\d,]+) in approx", int, src="logs/run.log")
    grab("pand_demol_kept", r"demolished-after-event kept: ([\d,]+)", int, src="logs/run.log")
    grab("pand_future", r"bouwjaar>2021 dropped: ([\d,]+)", int, src="logs/run.log")
    grab("vbo_total", r"vbo: ([\d,]+) total -> ([\d,]+) in approx", int, src="logs/run.log")
    grab("vbo_snap", r"vbo: [\d,]+ total -> ([\d,]+) in approx", int, src="logs/run.log")
    grab("vbo_res", r"vbo residential \(woonfunctie incl\. mixed\): ([\d,]+)", int, src="logs/run.log")
    grab("vbo_res_pure", r"pure woonfunctie: ([\d,]+)", int, src="logs/run.log")
    grab("pand_with_dwelling", r"panden with >=1 dwelling: ([\d,]+)", int, src="logs/run.log")
    grab("pand_no_vbo", r"without any address object: ([\d,]+)", int, src="logs/run.log")
    grab("res_in_extent", r"\[ALL Geul polygons\] buildings in observed extent: ([\d,]+) of ([\d,]+); residential buildings ([\d,]+)", lambda s: int(s), grp=3, src="logs/run.log")
    grab("bldg_in_extent", r"\[ALL Geul polygons\] buildings in observed extent: ([\d,]+)", int, src="logs/run.log")
    grab("dw_pts_in_extent", r"dwelling address objects ([\d,]+) of ([\d,]+); non-residential address objects ([\d,]+)", int, src="logs/run.log")
    grab("dw_total_snap", r"dwelling address objects [\d,]+ of ([\d,]+);", int, src="logs/run.log")
    grab("nonres_in_extent", r"non-residential address objects ([\d,]+)", int, src="logs/run.log")
    grab("dn1_res", r"\[Geul DN==1 only\] buildings in observed extent: [\d,]+ of [\d,]+; residential buildings ([\d,]+); dwelling address objects ([\d,]+)", int, src="logs/run.log")
    grab("dn1_dw", r"\[Geul DN==1 only\].*?dwelling address objects ([\d,]+)", int, src="logs/run.log")
    grab("dw_bldg_touch", r"dwellings whose address POINT is in extent vs whose BUILDING touches extent: [\d,]+ vs ([\d,]+)", int, src="logs/run.log")
    grab("n_cells", r"cells touching the 243 study buurten: ([\d,]+)", int, src="logs/06.out")
    grab("cell_pop", r"cell population sum ([\d,]+)", int, src="logs/06.out")
    grab("pop_missing_pct", r"aantal_inwoners\s+([\d.]+)%", src="logs/06.out")
    for key, lab in (("men", "aantal_mannen"), ("women", "aantal_vrouwen"), ("a65", "aantal_inwoners_65_jaar_en_ouder"), ("a014", "aantal_inwoners_0_tot_15_jaar"), ("hh", "aantal_part_huishoudens"), ("single", "aantal_eenpersoonshuishoudens"), ("dw_cbs", "aantal_woningen"), ("benefit", "aantal_personen_met_uitkering_onder_aowlft")):
        grab("miss_" + key, lab + r"\s+([\d.]+)%", src="logs/06.out")
    grab("grid_a", r"A_area_weighted\s+([\d,]+)", int, src="logs/06.out")
    grab("grid_b", r"B_centroid\s+([\d,]+)", int, src="logs/06.out")
    grab("grid_c", r"C_bag_allocated\s+([\d,]+)", int, src="logs/06.out")
    grab("study_pop_grid", r"total_pop_in_study_cells\s+([\d,]+)", int, src="logs/06.out")
    HH = r"\n  households\s+: known in [\d.]+% of cells / "
    grab("hh_known_pop", HH + r"([\d.]+)% of pop", src="logs/06.out")
    grab("hh_A", HH + r"[\d.]+% of pop; exposed \(A\)\s+([\d,]+)", int, src="logs/06.out")
    grab("hh_C", HH + r"[\d.]+% of pop; exposed \(A\)\s+[\d,]+\s+exposed \(C\)\s+([\d,]+)", int, src="logs/06.out")
    grab("hh_total", HH + r"[\d.]+% of pop; exposed \(A\)\s+[\d,]+\s+exposed \(C\)\s+[\d,]+\s+of\s+([\d,]+)", int, src="logs/06.out")
    grab("buurt_pop_grid", r"Sum of CBS grid pop over study buurten ([\d,]+)", int, src="logs/07.out")
    grab("buurt_ratio", r"ratio ([\d.]+)\)", src="logs/07.out")
    grab("cells_assigned", r"cells assigned to a study buurt: ([\d.]+)%", src="logs/07.out")
    grab("top5", r"top 5 / 10 / 20 buurten hold ([\d.]+)% / ([\d.]+)% / ([\d.]+)%", src="logs/08.out")
    grab("top10share", r"top 5 / 10 / 20 buurten hold [\d.]+% / ([\d.]+)% / ", src="logs/08.out")
    grab("top20", r"top 5 / 10 / 20 buurten hold [\d.]+% / [\d.]+% / ([\d.]+)%", src="logs/08.out")
    grab("n_resp", r"RESPONSE share of buurt population exposed \(BAG-allocated\): n=(\d+)", int, src="logs/07.out")
    grab("resp_zero", r"RESPONSE share[^\n]*zero: (\d+) \(", int, src="logs/07.out")
    grab("resp_zero_pct", r"RESPONSE share[^\n]*zero: \d+ \(([\d.]+)%\)", src="logs/07.out")
    grab("resp_ge50", r">=0\.5: (\d+)", int, src="logs/07.out")
    grab("resp_eq1", r">=0\.5: \d+; ==1: (\d+)", int, src="logs/07.out")
    grab("alt_area_zero", r"alt response area-weighted: zero ([\d.]+)%", src="logs/07.out")
    grab("corr_AC", r"corr\(A,C\) = ([\d.]+)", src="logs/07.out")
    grab("corr_dwC", r"corr\(dwelling share, C\) = ([\d.]+)", src="logs/07.out")
    grab("ahn_cells", r"AHN mosaic \(\d+, \d+\), valid cells ([\d,]+)", int, src="logs/run.log")
    grab("ahn_min", r"elevation min/median/max = ([\d.]+)/", src="logs/run.log")
    grab("ahn_max", r"elevation min/median/max = [\d.]+/[\d.]+/([\d.]+)", src="logs/run.log")
    grab("bbg_n", r"BBG2017: ([\d,]+) polygons", int, src="logs/run.log")
    grab("nwb_n", r"NWB wegvakken: ([\d,]+)", int, src="logs/run.log")
    grab("water_n", r"Top10NL waterdeel_vlak: ([\d,]+)", int, src="logs/run.log")
    grab("road_density_med", r"buurt road density median ([\d.]+)", src="logs/08.out")
    grab("n_study_buurten", r"(\d+) buurten; \d+ tiles of", int, src="logs/02_fetch_bag.out")
    grab("bag_tiles", r"\d+ buurten; (\d+) tiles of", int, src="logs/02_fetch_bag.out")
    grab("geul_dn1_km2", r"Geul area km2: DN==1: ([\d.]+)", src="logs/run.log")
    grab("geul_dn0_km2", r"DN==0: ([\d.]+);", src="logs/run.log")
    grab("union_km2", r"union of all parts: ([\d.]+) km2", src="logs/run.log")
    grab("maas_invalid", r"maas: n=\d+ invalid_before=(\d+)", int, src="logs/run.log")
    return f


def stage_A():
    say("[A] buurten / cells / grid / observed / facts")
    h = gpd.read_file(PROC / "buurt_analysis_env.gpkg").reset_index(drop=True)
    h["gemeente"] = h.gemeentenaam
    keep = {"buurtcode": "code", "buurtnaam": "name", "gemeente": "gem", "pop_kf": "pop", "pop_grid": "popgrid", "popA": "popA", "popC": "popC", "share_C": "shC", "share_A": "shA",
            "dw": "dw", "dw_flood": "dwf", "pct_65p": "p65", "pct_0_15": "p014", "pct_single_hh": "psing", "pct_hh_kids": "pkids", "pct_low_inc_pers": "plow", "benefits_per100": "ben",
            "cars_per_hh": "cars", "pct_owner": "pown", "pct_multifam": "pmulti", "density": "dens", "dist_gp_km": "gp", "elev_mean": "elev", "road_km_per_km2": "road", "water_share": "wat",
            "soil_main": "soil", "hh_size": "hhs", "percentagePersonen15Tot25Jaar": "p1525", "percentagePersonen25Tot45Jaar": "p2545", "percentagePersonen45Tot65Jaar": "p4565"}
    d = h[list(keep)].rename(columns=keep)
    for c in ("popA", "popC", "popgrid"):
        d[c] = d[c].round(1)
    for c in ("shC", "shA", "wat"):
        d[c] = d[c].round(4)
    for c in ("elev", "road", "ben", "cars", "gp", "hhs"):
        d[c] = d[c].round(2)
    # CBS sentinel -99999999 -> null for the 5-group shares not already cleaned
    for c in ("p1525", "p2545", "p4565"):
        d[c] = d[c].where(d[c] > -99990)
    gh = gpd.GeoDataFrame(d, geometry=h.geometry.simplify(6, preserve_topology=True), crs=RD)
    write("buurten.json", fc(gh))

    cells = gpd.read_file(PROC / "grid_cells_study.gpkg")
    cb = cells.buurtcode if "buurtcode" in cells else None
    bidx = {c: i for i, c in enumerate(h.buurtcode)}
    # cells carry no buurtcode in grid_cells_study.gpkg (it was added in memory in script 07): reassign by centroid exactly as 07 did
    cc = cells.copy(); cc["geometry"] = cc.geometry.centroid
    j = gpd.sjoin(cc[["crs28992res100m", "geometry"]], h[["buurtcode", "geometry"]], how="left", predicate="within").drop_duplicates("crs28992res100m").set_index("crs28992res100m").buurtcode
    cells["bi"] = cells.crs28992res100m.map(j).map(bidx).fillna(-1).astype(int)
    cen = gpd.GeoSeries(cells.geometry.centroid, crs=RD).to_crs(4326)
    rows = []
    nn = lambda v: None if pd.isna(v) else int(round(v))
    for k, (x, y) in enumerate(zip(cen.x, cen.y)):
        r = cells.iloc[k]
        f1 = lambda v: None if pd.isna(v) else round(float(v), 3)
        rows.append([round(x, 5), round(y, 5), nn(r["pop"]), f1(r.pop_A), f1(r.pop_C), int(r.n_dw_bag), int(r.n_dw_flood), int(r.bi),
                     nn(r.aantal_part_huishoudens), nn(r.aantal_inwoners_65_jaar_en_ouder), nn(r.aantal_inwoners_0_tot_15_jaar), round(float(r.flood_frac), 3)])
    write("cells.json", {"cols": ["lon", "lat", "pop", "popA", "popC", "dw", "dwf", "b", "hh", "a65", "a014", "ff"], "rows": rows})

    gg = gpd.GeoDataFrame({"pop": cells["pop"].round(0), "popC": cells.pop_C.round(1), "ff": cells.flood_frac.round(3), "dw": cells.n_dw_bag.astype(int), "dwf": cells.n_dw_flood.astype(int)}, geometry=cells.geometry, crs=RD)
    write("grid.json", fc(gg, nd=5))

    ext = gpd.read_file(PROC / "flood_obs_2021_all_parts.gpkg")
    raw = gpd.read_file(RAW / "limburg2021/Geul_v1/Geul_FloodExtent_v1.shp").set_crs(RD)
    parts = []
    for src, g in ext.groupby("src"):
        if src == "Geul_all":
            for _, r in raw.iterrows():
                parts.append({"src": "Geul", "dn": int(r.DN), "geometry": r.geometry.buffer(0)})
        else:
            for geom in g.geometry:
                parts.append({"src": src, "dn": None, "geometry": geom})
    og = gpd.GeoDataFrame(parts, crs=RD)
    og["geometry"] = og.geometry.simplify(4, preserve_topology=True)
    og = og[~og.geometry.is_empty]
    # merge Maas fragments (2,887 polygons) into one multipolygon per source/dn for a smaller file
    og["dn"] = og.dn.fillna(-1).astype(int)
    og = og.dissolve(by=["src", "dn"]).reset_index()
    og["dn"] = og.dn.replace(-1, None)
    write("observed.json", fc(og, ["src", "dn"]))

    facts = parse_logs()
    # numbers computed directly from the data
    facts["n_buurten"] = {"v": int(len(h)), "src": "outputs/buurt_analysis_env.csv"}
    facts["pop_buurten"] = {"v": int(h.pop_kf.sum()), "src": "outputs/buurt_analysis_env.csv"}
    facts["bounds"] = {"v": [float(x) for x in gpd.GeoSeries(h.geometry, crs=RD).to_crs(4326).total_bounds], "src": "buurt_analysis_env.gpkg"}
    facts["n_zero_C"] = {"v": int((h.popC == 0).sum()), "src": "outputs/summary_table.csv"}
    inv = pd.read_csv(OUT / "verified_data_inventory.csv"); facts["inventory"] = {"v": json.loads(inv.to_json(orient="records")), "src": "outputs/verified_data_inventory.csv"}
    facts["suppression"] = {"v": pd.read_csv(OUT / "grid_suppression_rates.csv").rename(columns={"Unnamed: 0": "field"}).round(4).to_dict("records"), "src": "outputs/grid_suppression_rates.csv"}
    top = pd.read_csv(OUT / "top10_exposed_buurten.csv"); facts["top10"] = {"v": json.loads(top.round(3).to_json(orient="records")), "src": "outputs/top10_exposed_buurten.csv"}
    facts["age"] = {"v": pd.read_csv(OUT / "exposed_by_age_group.csv").to_dict("records"), "src": "outputs/exposed_by_age_group.csv"}
    facts["cap"] = {"v": pd.read_csv(OUT / "modelled_vs_observed_corrected.csv").round(4).to_dict("records"), "src": "outputs/modelled_vs_observed_corrected.csv"}
    facts["cap_river"] = {"v": pd.read_csv(OUT / "modelled_catch_by_river_corrected.csv").round(4).to_dict("records"), "src": "outputs/modelled_catch_by_river_corrected.csv"}
    facts["model_area"] = {"v": pd.read_csv(OUT / "modelled_vs_observed.csv").round(4).to_dict("records"), "src": "outputs/modelled_vs_observed.csv (area columns only; its building columns are superseded)"}
    facts["depth"] = {"v": pd.read_csv(OUT / "modelled_depth_at_flooded_residential_buildings.csv").to_dict("records"), "src": "outputs/modelled_depth_at_flooded_residential_buildings.csv"}
    facts["summary"] = {"v": pd.read_csv(OUT / "summary_table.csv").to_dict("records"), "src": "outputs/summary_table.csv"}
    facts["vif"] = {"v": pd.read_csv(OUT / "vif.csv").round(2).to_dict("records"), "src": "outputs/vif.csv"}
    facts["logit"] = {"v": pd.read_csv(OUT / "logit_any_exposure.csv").round(4).to_dict("records"), "src": "outputs/logit_any_exposure.csv"}
    # Spearman correlation with the response, same predictors and complete cases as script 07
    PRED = ["pct_65p", "pct_0_15", "pct_single_hh", "pct_low_inc_pers", "benefits_per100", "cars_per_hh", "pct_owner", "pct_multifam", "density", "dist_gp_km", "elev_mean"]
    cc2 = h.dropna(subset=PRED + ["share_C"])
    facts["spearman"] = {"v": {p: round(float(cc2[p].corr(cc2.share_C, method="spearman")), 3) for p in PRED}, "src": "recomputed from outputs/buurt_analysis_env.csv (same sample as 07)"}
    facts["hist"] = {"v": [round(float(x), 4) for x in cc2.share_C.tolist()], "src": "outputs/buurt_analysis_env.csv (share_C, complete cases)"}
    miss = h[PRED].isna().mean().round(3).to_dict(); facts["pred_missing"] = {"v": miss, "src": "outputs/buurt_analysis_env.csv"}
    (DATA / "facts.json").write_text(json.dumps(facts, separators=(",", ":")), encoding="utf8")
    say(f"  facts.json: {len(facts)} keys")


if __name__ == "__main__":
    for st in sys.argv[1:] or ["A"]:
        globals()["stage_" + st]()
