# Netherlands flood research site (Limburg, July 2021)

A static research site (full-screen MapLibre map, top control bar, floating legend/layer cards, 430 px side panel, light/dark theme, phone layout), carrying the verified Netherlands data chain in `../nl_verification`.

## Run it

Browsers block `fetch` from `file://`, so serve the folder:

```bash
cd "D:/flood anaylsis/nl_site"
python -m http.server 8791
```

Open http://localhost:8791/ (map) and http://localhost:8791/analysis/ (evidence and method). Needs internet for MapLibre, D3, Public Sans and the CARTO basemap (the map falls back to a plain background if the basemap cannot be reached in 6 s).

## Pages

| Page | Content |
|---|---|
| `index.html` | Map explorer. Buurt choropleth (16 variables), municipality/buurt filters, A/B circles on 100 m cells, observed extent, modelled ROR scenarios (hatched, separate legend group), scenario depth classes, BAG buildings (coloured by flooded in 2021, or by whether the scenario catches them), 100 m grid, AHN4 relief, land use, soil, water, roads. |
| `analysis/index.html` | Headline numbers, observed vs modelled evidence (Valkenburg small multiples, capture tables, area overlap, scenario depth), the 11-step provenance chain, methods A/B/C, top-10 buurten, age and suppression, BAG 2021 reconstruction, regression feasibility, limits, reproducibility frictions, source table with URLs. |

## Where the numbers come from

`data/facts.json` is parsed from the project logs and output tables by `build/build_web_assets.py` (stage A); every value has a `src` field. The pages read it at load time. Nothing is typed in by hand except the text of the limits list, which restates the verification report.

Checks run on the finished site: study-area totals in the side panel equal the project's (282,760 residents in cells; exposed residents A 6,535, B 5,490, C 3,877); building capture counts per scenario equal `outputs/modelled_vs_observed_corrected.csv` (572 / 873 / 1,579 / 1,594 of 2,033 at a 2 m buffer).

## Rebuild the web assets

```bash
cd build
../../nl_verification/.venv/Scripts/python build_web_assets.py A
../../nl_verification/.venv/Scripts/python stage_b.py B1 B2 B3 B4 B5 B6   # about 8 minutes
```

The browser files in `data/` and `assets/` are derived, display-only copies (holes under 3,000 m² in the scenario polygons filled, geometries simplified, relief as a PNG). The research data and calculations are not changed, and no figure is computed from the simplified copies. Heavy layers load on first use (largest: scenario depth 6–11 MB, buildings 15 MB, roads 8 MB).

## Known limits of the site

- The 4TU extent is preliminary and partly modelled; the Geul file has no CRS (EPSG:28992 assumed).
- The scenario layers have an unclear licence and are served from a third-party host.
- BAG is today's snapshot; the 2021 state is approximated.
- Age on the map is a buurt-level proxy; the 100 m grid suppresses most subgroup counts.
