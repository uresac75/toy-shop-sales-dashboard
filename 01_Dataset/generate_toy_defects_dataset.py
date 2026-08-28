"""
Toys Manufacturing Defect Analysis - Synthetic Dataset Generator
=================================================================
Generates a star-schema CSV set (5 dimensions + 1 fact) for QA/defect
analytics across an India-based toy manufacturing operation.

Date range : Jan 2024 - Aug 2026
Fact rows  : >= 250,000 inspection records
Built-in   : seasonality (Q4 export ramp, Deepavali, Chinese New Year dip),
             equipment-age -> structural defects, night-shift -> cosmetic
             defects, bad-supplier -> electronic defects, critical safety
             defects -> 100% batch scrap, Pareto product popularity.
"""

import numpy as np
import pandas as pd
from datetime import date, datetime, timedelta
import zipfile, os, sys

rng = np.random.default_rng(42)
OUT = os.path.dirname(os.path.abspath(__file__))

START = date(2024, 1, 1)
END = date(2026, 8, 31)
DAYS = (END - START).days + 1
SHIFTS = ["Morning", "Afternoon", "Night"]

# ---------------------------------------------------------------------------
# DIM 1: PRODUCTS (100 SKUs, 5 categories, long-tail popularity)
# ---------------------------------------------------------------------------
CATEGORIES = {
    "Action Figures":  {"prefix": "AF", "material": "Plastic",     "cost": (120, 450),  "complexity": (3, 6)},
    "Electronic Toys": {"prefix": "ET", "material": "Electronics", "cost": (450, 2500), "complexity": (6, 10)},
    "Wooden Blocks":   {"prefix": "WB", "material": "Wood",        "cost": (150, 600),  "complexity": (1, 3)},
    "Plush Toys":      {"prefix": "PT", "material": "Fabric",      "cost": (100, 550),  "complexity": (2, 4)},
    "Board Games":     {"prefix": "BG", "material": "Cardboard",   "cost": (200, 900),  "complexity": (3, 7)},
}
NAME_POOL = {
    "Action Figures": ["Galaxy Ranger", "Jungle Commando", "Robo Warrior", "Ninja Strike", "Dino Hunter",
                       "Space Trooper", "Cricket Legend", "Super Rakshak", "Mecha Titan", "Desert Scout"],
    "Electronic Toys": ["Talking Parrot", "RC Racer", "Learning Tablet", "Dance Robot", "Laser Blaster",
                        "Musical Keyboard", "Smart Puppy", "Drone Explorer", "Karaoke Mic", "Coding Bot"],
    "Wooden Blocks": ["Alphabet Tower", "Rainbow Stacker", "City Builder", "Temple Blocks", "Number Train",
                      "Shape Sorter", "Animal Puzzle", "Bridge Kit", "Castle Set", "Geo Blocks"],
    "Plush Toys": ["Cuddly Elephant", "Bengal Tiger Cub", "Sleepy Panda", "Peacock Pal", "Teddy Classic",
                   "Unicorn Dream", "Baby Dino", "Monkey Mischief", "Polar Bear Hug", "Bunny Bliss"],
    "Board Games": ["Snakes & Ladders Deluxe", "Ludo Champion", "Trade Tycoon", "Word Wizard", "Chess Master",
                    "Carrom Junior", "Quiz Quest", "Memory Match", "Strategy Siege", "Treasure Trail"],
}
AGE_GROUPS = ["0-2 yrs", "3-5 yrs", "6-8 yrs", "9-12 yrs", "13+ yrs"]

products = []
pid = 1
for cat, meta in CATEGORIES.items():
    for i in range(20):
        base = NAME_POOL[cat][i % 10]
        variant = ["", " Pro", " Mini", " XL", " Junior", " Classic", " Neo", " Max", " Lite", " 2.0"][i // 2]
        products.append({
            "ProductID": f"{meta['prefix']}-{pid:03d}",
            "ProductName": (base + variant).strip(),
            "Category": cat,
            "TargetAgeGroup": rng.choice(AGE_GROUPS),
            "ComplexityScore": int(rng.integers(meta["complexity"][0], meta["complexity"][1] + 1)),
            "StandardCost": round(float(rng.uniform(*meta["cost"])), 2),
            "PrimaryMaterial": meta["material"],
        })
        pid += 1
dim_products = pd.DataFrame(products)

# Pareto popularity: top ~20% of SKUs get ~70% of volume
pop = rng.pareto(1.3, size=len(dim_products)) + 0.05
prod_weights = pop / pop.sum()

# ---------------------------------------------------------------------------
# DIM 2: FACTORIES & ASSEMBLY LINES
# ---------------------------------------------------------------------------
FACTORIES = [
    ("Sriperumbudur, Tamil Nadu", 8),
    ("Hosur, Tamil Nadu", 7),
    ("Coimbatore, Tamil Nadu", 6),
    ("Noida, Uttar Pradesh", 5),
    ("Sri City, Andhra Pradesh", 4),
]
MANAGERS = ["R. Karthik", "S. Priya", "V. Aravind", "M. Divya", "K. Saravanan", "P. Keerthana",
            "A. Vignesh", "D. Harini", "N. Praveen", "G. Gayathri", "S. Dinesh", "L. Swetha",
            "T. Hariharan", "B. Nithya", "C. Naveen", "J. Deepika", "R. Mohan", "S. Anand",
            "V. Lakshmi", "K. Ramesh", "P. Suresh", "M. Kavya", "A. Bala", "D. Meena",
            "N. Ganesh", "G. Revathi", "S. Kumar", "L. Janani", "T. Rajesh", "B. Sandhya"]
MAINT = ["Monthly", "Quarterly", "Bi-Annual"]

lines = []
lid = 1
for floc, nlines in FACTORIES:
    for _ in range(nlines):
        lines.append({
            "LineID": f"LINE-{lid:02d}",
            "FactoryLocation": floc,
            "LineManager": MANAGERS[lid - 1],
            "EquipmentAgeYears": round(float(rng.uniform(0.5, 15.0)), 1),
            "MaintenanceSchedule": rng.choice(MAINT, p=[0.3, 0.45, 0.25]),
        })
        lid += 1
dim_lines = pd.DataFrame(lines)
N_LINES = len(dim_lines)

# per-line daily capacity (units) - uneven, bigger factories run faster lines
line_capacity = rng.integers(900, 2600, size=N_LINES).astype(float)
eq_age = dim_lines["EquipmentAgeYears"].to_numpy()
# older equipment runs a bit slower
line_capacity *= (1.0 - 0.012 * eq_age)

# ---------------------------------------------------------------------------
# DIM 3: SUPPLIERS
# ---------------------------------------------------------------------------
SUPPLIER_SPECS = [
    # (name, material, quality rating 1-5)  -- deliberately unequal quality
    ("Chennai Polymers Pvt Ltd", "Plastic", 4.6), ("Ambattur Plastics Co", "Plastic", 4.1),
    ("Hosur Moulding Works", "Plastic", 3.4), ("Gujarat Petrochem Toys", "Plastic", 4.3),
    ("Shenzhen MicroCircuits Ltd", "Electronics", 4.5), ("Dragon Electronics Co", "Electronics", 2.4),
    ("Bengaluru ChipWorks", "Electronics", 4.2), ("Taiwan ToyTronics", "Electronics", 3.9),
    ("Tirupur Textiles & Fabrics", "Fabric", 4.4), ("Erode Cotton Mills", "Fabric", 4.0),
    ("Surat Synthetic Fabrics", "Fabric", 3.1), ("Karur Plush Materials", "Fabric", 4.5),
    ("Kerala Timber Traders", "Wood", 4.3), ("Assam Plywood Industries", "Wood", 3.6),
    ("Mysore Wood Crafts", "Wood", 4.7), ("Nilgiri Forest Products", "Wood", 3.8),
    ("Sivakasi Print & Pack", "Cardboard", 4.6), ("Chennai Corrugators", "Cardboard", 3.9),
    ("Delhi Paper Boards", "Cardboard", 3.2), ("Madurai Packaging Co", "Cardboard", 4.1),
]
dim_suppliers = pd.DataFrame([
    {"SupplierID": f"SUP-{i+1:02d}", "SupplierName": n, "MaterialType": m, "QualityRating": q}
    for i, (n, m, q) in enumerate(SUPPLIER_SPECS)
])
BAD_ELEC_SUPPLIER = "SUP-06"  # Dragon Electronics Co - the notorious one

# supplier pools per material with volume weights biased to a few big vendors
supplier_pool = {}
for mat in ["Plastic", "Electronics", "Fabric", "Wood", "Cardboard"]:
    idx = dim_suppliers.index[dim_suppliers["MaterialType"] == mat].to_numpy()
    w = rng.dirichlet(np.ones(len(idx)) * 0.8)
    supplier_pool[mat] = (dim_suppliers.loc[idx, "SupplierID"].to_numpy(),
                          dim_suppliers.loc[idx, "QualityRating"].to_numpy(), w)

# ---------------------------------------------------------------------------
# DIM 4: DEFECT TYPES
# ---------------------------------------------------------------------------
DEFECTS = [
    ("No Defect Found", "None", "None", "No"),
    ("Paint smudge / colour bleed", "Cosmetic", "Low", "No"),
    ("Surface scratch", "Cosmetic", "Low", "No"),
    ("Misaligned decal or sticker", "Cosmetic", "Low", "No"),
    ("Uneven paint finish", "Cosmetic", "Medium", "No"),
    ("Discoloured plastic moulding", "Cosmetic", "Medium", "No"),
    ("Cracked body shell", "Structural", "High", "Yes"),
    ("Loose joint / limb detachment", "Structural", "Medium", "No"),
    ("Warped plastic part", "Structural", "Medium", "No"),
    ("Broken hinge mechanism", "Structural", "High", "Yes"),
    ("Splintered wooden edge", "Structural", "High", "Yes"),
    ("Seam stitching failure", "Structural", "Medium", "No"),
    ("Battery compartment failure", "Electronic", "High", "Yes"),
    ("Speaker no sound output", "Electronic", "Medium", "No"),
    ("LED not functioning", "Electronic", "Low", "No"),
    ("Circuit board short", "Electronic", "High", "Yes"),
    ("Button unresponsive", "Electronic", "Medium", "No"),
    ("Charging port defect", "Electronic", "High", "Yes"),
    ("Box crush damage", "Packaging", "Low", "No"),
    ("Missing instruction manual", "Packaging", "Low", "No"),
    ("Incorrect label / barcode", "Packaging", "Medium", "No"),
    ("Shrink-wrap tear", "Packaging", "Low", "No"),
    ("Small parts choking hazard", "Safety Hazard", "Critical", "Yes"),
    ("Sharp edge exposure", "Safety Hazard", "Critical", "Yes"),
    ("Toxic paint lead content", "Safety Hazard", "Critical", "Yes"),
    ("Battery overheating risk", "Safety Hazard", "Critical", "Yes"),
]
dim_defects = pd.DataFrame([
    {"DefectID": f"DEF-{i:03d}", "DefectDescription": d, "DefectCategory": c,
     "Severity": s, "RequiresScrap": r}
    for i, (d, c, s, r) in enumerate(DEFECTS)
])
DEF_IDS = dim_defects["DefectID"].to_numpy()
DEF_CAT = dim_defects["DefectCategory"].to_numpy()
DEF_SEV = dim_defects["Severity"].to_numpy()
DEF_SCRAP = (dim_defects["RequiresScrap"] == "Yes").to_numpy()

def def_idx(category):
    return np.where(DEF_CAT == category)[0]

IDX_COSMETIC = def_idx("Cosmetic")
IDX_STRUCT = def_idx("Structural")
IDX_ELEC = def_idx("Electronic")
IDX_PACK = def_idx("Packaging")
IDX_SAFETY = def_idx("Safety Hazard")

# ---------------------------------------------------------------------------
# DIM 5: CALENDAR & SHIFTS (with seasonal production multiplier)
# ---------------------------------------------------------------------------
def deepavali(year):
    return {2024: date(2024, 10, 31), 2025: date(2025, 10, 20), 2026: date(2026, 11, 8)}[year]

def season_mult(d: date) -> float:
    """Production volume multiplier - export toy business."""
    m = 1.0
    # Q4 holiday export ramp: production peaks Aug-Oct to ship for Christmas
    if d.month == 7:   m *= 1.15
    elif d.month == 8: m *= 1.45
    elif d.month == 9: m *= 1.55
    elif d.month == 10: m *= 1.40
    elif d.month == 11: m *= 1.10
    # Domestic Deepavali pre-build (30-45 days before)
    dv = deepavali(d.year)
    delta = (dv - d).days
    if 0 <= delta <= 40:
        m *= 1.12
    # Pongal week slowdown (TN factories, worker leave)
    if d.month == 1 and 13 <= d.day <= 18:
        m *= 0.55
    # Post-holiday lull
    if d.month in (1, 2):
        m *= 0.85
    # YoY growth: ~9% per year
    m *= 1.0 + 0.09 * (d.year - 2024) + 0.09 * (d.timetuple().tm_yday / 365.0) * 0.5
    # Sundays: skeleton crew
    if d.weekday() == 6:
        m *= 0.45
    return m

SHIFT_MULT = {"Morning": 1.05, "Afternoon": 1.00, "Night": 0.82}

cal_rows = []
all_dates = [START + timedelta(days=i) for i in range(DAYS)]
for d in all_dates:
    sm = season_mult(d)
    for sh in SHIFTS:
        cal_rows.append({
            "Date": d.isoformat(), "Year": d.year, "Quarter": f"Q{(d.month-1)//3+1}",
            "Month": d.month, "MonthName": d.strftime("%B"), "DayOfWeek": d.strftime("%A"),
            "IsWeekend": "Yes" if d.weekday() >= 5 else "No", "Shift": sh,
            "ShiftStart": {"Morning": "06:00", "Afternoon": "14:00", "Night": "22:00"}[sh],
            "ShiftEnd": {"Morning": "14:00", "Afternoon": "22:00", "Night": "06:00"}[sh],
            "SeasonalIndex": round(sm, 3),
        })
dim_calendar = pd.DataFrame(cal_rows)

# ---------------------------------------------------------------------------
# FACT TABLE: QUALITY INSPECTIONS
# ---------------------------------------------------------------------------
# Each batch -> 1..n inspection rows (one per defect type found; clean
# batches get a single DEF-000 "No Defect Found" row).

prod_ids = dim_products["ProductID"].to_numpy()
prod_cost = dim_products["StandardCost"].to_numpy()
prod_cat = dim_products["Category"].to_numpy()
prod_mat = dim_products["PrimaryMaterial"].to_numpy()
prod_cplx = dim_products["ComplexityScore"].to_numpy()

# line downtime events (maintenance / breakdowns) - unequal reliability
line_reliability = np.clip(0.985 - 0.004 * eq_age + rng.normal(0, 0.004, N_LINES), 0.90, 0.995)

fact_chunks = []
insp_counter = 1
batch_counter = 1

for d in all_dates:
    sm = season_mult(d)
    date_str = d.isoformat()
    for sh in SHIFTS:
        shm = SHIFT_MULT[sh]
        # which lines are running this shift
        running = rng.random(N_LINES) < line_reliability
        # Sunday: only ~40% of lines staffed
        if d.weekday() == 6:
            running &= rng.random(N_LINES) < 0.45
        run_idx = np.where(running)[0]
        if len(run_idx) == 0:
            continue

        n = len(run_idx)
        # product per line-shift (a line runs one SKU per batch)
        p_idx = rng.choice(len(prod_ids), size=n, p=prod_weights)
        # 1-2 batches per line-shift
        n_batches = rng.integers(1, 3, size=n)

        for j, li in enumerate(run_idx):
            for _b in range(n_batches[j]):
                pi = p_idx[j] if _b == 0 else rng.choice(len(prod_ids), p=prod_weights)
                cat = prod_cat[pi]
                mat = prod_mat[pi]
                cost = prod_cost[pi]
                cplx = prod_cplx[pi]

                # supplier for the primary material
                sids, squal, sw = supplier_pool[mat]
                s_sel = rng.choice(len(sids), p=sw)
                sid, srating = sids[s_sel], squal[s_sel]

                units = int(line_capacity[li] * sm * shm / n_batches[j]
                            * rng.uniform(0.85, 1.15))
                units = max(units, 120)
                # sampling inspection 8-18%; electronics inspected more
                samp = rng.uniform(0.12, 0.20) if cat == "Electronic Toys" else rng.uniform(0.08, 0.15)
                inspected = max(int(units * samp), 30)

                # ---- base defect probability engine -------------------
                base = 0.020 + 0.0035 * cplx                 # complexity
                base += (4.5 - srating) * 0.008              # supplier quality
                if sh == "Night":
                    base += 0.006                            # fatigue
                base += eq_age[li] * 0.0016                  # equipment wear
                # ramp-season pressure: rushing raises defects slightly
                if sm > 1.3:
                    base += 0.004
                base = min(base, 0.14)

                # expected defective units in the sample
                exp_def = inspected * base
                total_def_units = int(rng.poisson(max(exp_def, 0.01)))
                total_def_units = min(total_def_units, inspected)

                batch_id = f"BATCH-{batch_counter:07d}"
                batch_counter += 1

                rows_here = []
                if total_def_units == 0:
                    rows_here.append((0, 0))  # (defect_index, count) -> DEF-000
                else:
                    # category mix weights for this batch
                    w_cos = 1.0 + (0.55 if sh == "Night" else 0.0)          # Correlation 2
                    w_str = 0.7 + eq_age[li] * 0.11                          # Correlation 1
                    w_ele = 0.0
                    if cat == "Electronic Toys":
                        w_ele = 1.1 + (2.8 if sid == BAD_ELEC_SUPPLIER else 0.0)  # bad supplier
                    w_pak = 0.5
                    w_saf = 0.015 + (0.02 if srating < 3.3 else 0.0)
                    if cat == "Wooden Blocks":
                        w_str *= 1.25
                    if cat == "Plush Toys":
                        w_cos *= 1.2
                    cat_w = np.array([w_cos, w_str, w_ele, w_pak, w_saf])
                    cat_w = cat_w / cat_w.sum()
                    cat_pick = rng.choice(5, size=total_def_units, p=cat_w)
                    idx_map = [IDX_COSMETIC, IDX_STRUCT, IDX_ELEC, IDX_PACK, IDX_SAFETY]
                    picked = {}
                    for cp in cat_pick:
                        di = int(rng.choice(idx_map[cp]))
                        picked[di] = picked.get(di, 0) + 1
                    rows_here = sorted(picked.items())

                # safety-critical -> 100% scrap of whole batch (Correlation 3)
                has_critical = any(DEF_SEV[di] == "Critical" for di, _c in rows_here)

                for di, cnt in rows_here:
                    if has_critical and DEF_SEV[di] == "Critical":
                        scrap_units = units          # entire batch condemned
                        rework_h = 0.0
                    elif DEF_SCRAP[di]:
                        scrap_units = cnt
                        rework_h = 0.0
                    else:
                        scrap_units = 0
                        rework_h = round(cnt * rng.uniform(0.05, 0.25), 2)
                    scrap_cost = round(scrap_units * cost, 2)

                    fact_chunks.append((
                        f"INSP-{insp_counter:07d}", batch_id, date_str, sh,
                        prod_ids[pi], dim_lines.iloc[li]["LineID"], sid,
                        units, inspected, DEF_IDS[di], cnt,
                        scrap_cost, rework_h,
                    ))
                    insp_counter += 1

fact = pd.DataFrame(fact_chunks, columns=[
    "InspectionID", "BatchID", "Date", "Shift", "ProductID", "LineID", "SupplierID",
    "TotalUnitsProduced", "UnitsInspected", "DefectID", "DefectCount",
    "ScrapCost", "ReworkHours",
])

# enterprise ETL metadata
fact["SourceSystem"] = "MES-QMS-v3"
fact["InsertTimestamp"] = (pd.to_datetime(fact["Date"]) + pd.Timedelta(days=1)
                           + pd.to_timedelta(rng.integers(6*3600, 10*3600, len(fact)), unit="s")
                           ).dt.strftime("%Y-%m-%d %H:%M:%S")
fact["RecordVersion"] = 1

# ---------------------------------------------------------------------------
# VALIDATION
# ---------------------------------------------------------------------------
print("=" * 64)
print("VALIDATION REPORT")
print("=" * 64)
ok = True

def check(label, cond):
    global ok
    status = "PASS" if cond else "FAIL"
    if not cond:
        ok = False
    print(f"[{status}] {label}")

check(f"Fact row count >= 250,000  (actual {len(fact):,})", len(fact) >= 250_000)
check("No nulls in fact table", not fact.isnull().any().any())
for name, df in [("dim_products", dim_products), ("dim_lines", dim_lines),
                 ("dim_suppliers", dim_suppliers), ("dim_defects", dim_defects),
                 ("dim_calendar", dim_calendar)]:
    check(f"No nulls in {name}", not df.isnull().any().any())

check("DefectCount <= UnitsInspected", (fact["DefectCount"] <= fact["UnitsInspected"]).all())
check("UnitsInspected <= TotalUnitsProduced", (fact["UnitsInspected"] <= fact["TotalUnitsProduced"]).all())

# ScrapCost mathematically tied to StandardCost
m = fact.merge(dim_products[["ProductID", "StandardCost"]], on="ProductID")
m = m.merge(dim_defects[["DefectID", "Severity", "RequiresScrap"]], on="DefectID")
crit = m[m["Severity"] == "Critical"]
check("Critical defects: ScrapCost == TotalUnitsProduced x StandardCost (100% scrap)",
      np.allclose(crit["ScrapCost"], crit["TotalUnitsProduced"] * crit["StandardCost"], atol=0.02))
noncrit_scrap = m[(m["RequiresScrap"] == "Yes") & (m["Severity"] != "Critical")]
check("Non-critical scrap defects: ScrapCost == DefectCount x StandardCost",
      np.allclose(noncrit_scrap["ScrapCost"], noncrit_scrap["DefectCount"] * noncrit_scrap["StandardCost"], atol=0.02))

# Correlation 2: night shift cosmetic defect rate
cos = fact.merge(dim_defects[["DefectID", "DefectCategory"]], on="DefectID")
cos_rate = (cos[cos["DefectCategory"] == "Cosmetic"].groupby("Shift")["DefectCount"].sum()
            / cos.groupby("Shift")["UnitsInspected"].sum())
print(f"        Cosmetic defect rate by shift: "
      f"Morning={cos_rate['Morning']:.4f}  Afternoon={cos_rate['Afternoon']:.4f}  Night={cos_rate['Night']:.4f}")
check("Night cosmetic rate > Morning & Afternoon",
      cos_rate["Night"] > cos_rate["Morning"] and cos_rate["Night"] > cos_rate["Afternoon"])

# Correlation 1: equipment age vs structural defect rate
st = cos[cos["DefectCategory"] == "Structural"].groupby("LineID")["DefectCount"].sum()
insp_by_line = fact.groupby("LineID")["UnitsInspected"].sum()
st_rate = (st / insp_by_line).fillna(0)
age_map = dim_lines.set_index("LineID")["EquipmentAgeYears"]
corr = np.corrcoef(age_map[st_rate.index], st_rate)[0, 1]
print(f"        Corr(EquipmentAge, StructuralDefectRate) = {corr:.3f}")
check("Equipment age positively correlated with structural defects (r > 0.5)", corr > 0.5)

# Bad supplier check
el = cos[cos["DefectCategory"] == "Electronic"]
el_rate = el.groupby("SupplierID")["DefectCount"].sum() / cos.groupby("SupplierID")["UnitsInspected"].sum()
el_rate = el_rate.dropna()
if BAD_ELEC_SUPPLIER in el_rate.index:
    others = el_rate.drop(BAD_ELEC_SUPPLIER).max()
    print(f"        Electronic defect rate {BAD_ELEC_SUPPLIER} (Dragon Electronics)={el_rate[BAD_ELEC_SUPPLIER]:.4f} "
          f"vs next-worst={others:.4f}")
    check("Bad electronics supplier has highest electronic defect rate",
          el_rate.idxmax() == BAD_ELEC_SUPPLIER)

# Seasonality: Aug-Oct production peak
prod_by_month = (fact.drop_duplicates("BatchID")
                 .assign(Month=lambda x: pd.to_datetime(x["Date"]).dt.month)
                 .groupby("Month")["TotalUnitsProduced"].sum())
peak = prod_by_month.loc[[8, 9, 10]].mean()
offpeak = prod_by_month.drop([8, 9, 10]).mean()
print(f"        Avg monthly production Aug-Oct={peak:,.0f} vs rest={offpeak:,.0f} ({peak/offpeak:.2f}x)")
check("Production peaks Aug-Oct (>= 1.25x other months)", peak / offpeak >= 1.25)

print("=" * 64)
print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
print("=" * 64)

# ---------------------------------------------------------------------------
# WRITE CSVs + ZIP
# ---------------------------------------------------------------------------
files = {
    "dim_products.csv": dim_products,
    "dim_assembly_lines.csv": dim_lines,
    "dim_suppliers.csv": dim_suppliers,
    "dim_defect_types.csv": dim_defects,
    "dim_calendar_shifts.csv": dim_calendar,
    "fact_quality_inspections.csv": fact,
}
for fn, df in files.items():
    df.to_csv(os.path.join(OUT, fn), index=False)
    print(f"Wrote {fn:35s} {len(df):>9,} rows")

zip_path = os.path.join(OUT, "toy_manufacturing_defects_dataset.zip")
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
    for fn in files:
        z.write(os.path.join(OUT, fn), fn)
print(f"\nZipped -> {zip_path} ({os.path.getsize(zip_path)/1e6:.1f} MB)")
sys.exit(0 if ok else 1)
