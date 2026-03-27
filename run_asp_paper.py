"""
run_asp_paper.py  (Option A — matches paper exactly)
=====================================================
Replicates the paper's exact Clingo setup:
  - All 3 scenarios run simultaneously (scenario(w1/w2/w3) all active)
  - Demand overridden via Clingo #const (not by modifying the file)
  - --opt-mode=optN -n 0 to enumerate all optimal stable models

This should reproduce the paper's model counts exactly:
  scale=100:  232-267 models per demand level
  scale=500:  1199-1517 models
  scale=1000: 2486-3250 models

Requirements:
    pip install clingo
    or: conda install -c potassco clingo

Usage (run from folder containing all .lp files):
    py run_asp_paper.py

Output: asp_results_paper.json
"""

import json, time, re, sys, os
from collections import defaultdict

try:
    import clingo
except ImportError:
    print("ERROR: clingo not installed.  pip install clingo")
    sys.exit(1)

# ── Configuration ─────────────────────────────────────────────────────────────
# Paper's exact demand values per scale
SCALES = {
    100:  {"instance": "instance_100.lp",  "demands": [8000,  10000, 12000]},
    500:  {"instance": "instance_500.lp",  "demands": [40000, 45000, 50000]},
    1000: {"instance": "instance_1000.lp", "demands": [90000, 95000, 100000]},
}
MODEL_FILE = "model.lp"
SCENARIOS  = ["w1", "w2", "w3"]

# Paper's reported model counts for verification
PAPER_MODELS = {
    (100,8000):267, (100,10000):232, (100,12000):234,
    (500,40000):1517,(500,45000):1205,(500,50000):1199,
    (1000,90000):3250,(1000,95000):2492,(1000,100000):2486,
}

# ── Data extraction ───────────────────────────────────────────────────────────

def extract_instance_data(lp_text):
    return {
        'backup_costs':  {b:int(v) for b,v in re.findall(r'cost_backup\((\w+),\s*(\d+)\)', lp_text)},
        'backup_caps':   {b:int(v) for b,v in re.findall(r'cap_backup\((\w+),\s*(\d+)\)', lp_text)},
        'backup_soc':    {b:int(v) for b,v in re.findall(r'soc_backup\((\w+),\s*(\d+)\)', lp_text)},
        'backup_env':    {b:int(v) for b,v in re.findall(r'env_backup\((\w+),\s*(\d+)\)', lp_text)},
        'primary_costs': {s:int(v) for s,v in re.findall(r'cost_primary\((\w+),\s*(\d+)\)', lp_text)},
        'primary_caps':  {s:int(v) for s,v in re.findall(r'cap_primary\((\w+),\s*(\d+)\)', lp_text)},
        'primary_soc':   {s:int(v) for s,v in re.findall(r'soc_primary\((\w+),\s*(\d+)\)', lp_text)},
        'primary_env':   {s:int(v) for s,v in re.findall(r'env_primary\((\w+),\s*(\d+)\)', lp_text)},
        'soc_thresh':    int(re.search(r'soc_threshold\((\d+)\)', lp_text).group(1)),
        'env_thresh':    int(re.search(r'env_threshold\((\d+)\)', lp_text).group(1)),
        'disrupted':     re.findall(r'disrupted\((\w+),\s*(\w+)\)', lp_text),
    }

# ── Override demand in instance text ─────────────────────────────────────────

def override_demand(lp_text, demand):
    """
    Replace the demand fact in the instance file with the target demand.
    All other facts (scenarios, disrupted, capacities) remain unchanged.
    This matches the paper's approach of running the original instance
    with a different demand level.
    """
    return re.sub(r'demand\(\d+\)', f'demand({demand})', lp_text)

# ── Solver ────────────────────────────────────────────────────────────────────

def solve_instance(lp_text, idata, scale, demand, model_text):
    """
    Run Clingo with all 3 scenarios active simultaneously.
    This matches the paper's exact setup.
    """
    bc  = idata['backup_costs'];   bcp = idata['backup_caps']
    bs  = idata['backup_soc'];     be  = idata['backup_env']
    pc  = idata['primary_costs'];  pcp = idata['primary_caps']
    ps  = idata['primary_soc'];    pe  = idata['primary_env']
    soc_thresh = idata['soc_thresh']
    env_thresh = idata['env_thresh']
    total_prim_cap   = sum(pcp.values())
    total_backup_cap = sum(bcp.values())

    # Residual capacity per scenario
    scenario_info = {}
    for w in SCENARIOS:
        dis = {s for s,sc in idata['disrupted'] if sc == w}
        eff = sum(0 if s in dis else v for s,v in pcp.items())
        scenario_info[w] = {
            'disrupted': dis,
            'eff_cap': eff,
            'cap_lost': total_prim_cap - eff,
            'cap_lost_pct': round((total_prim_cap - eff) / total_prim_cap * 100, 1),
            'n_disrupted': len(dis),
            'backup_needed': eff < demand,
        }

    # Override demand only — keep all 3 scenarios
    patched = override_demand(lp_text, demand)

    all_models     = []
    time_first_opt = None
    t_start        = time.perf_counter()

    # --opt-mode=optN: enumerate all models at optimal cost level
    # --quiet=0,2: suppress harmless "No bound given" warning
    ctl = clingo.Control(["-n", "0", "--opt-mode=optN"])
    ctl.add("base", [], patched)
    ctl.add("base", [], model_text)
    ctl.ground([("base", [])])

    with ctl.solve(yield_=True) as handle:
        for model in handle:
            t_now = time.perf_counter() - t_start
            atoms = list(model.symbols(shown=True))

            # Group atoms by scenario
            prim_by_w = defaultdict(set)
            bk_by_w   = defaultdict(set)
            for a in atoms:
                args = a.arguments
                if a.name == 'select' and len(args) == 2:
                    prim_by_w[str(args[1])].add(str(args[0]))
                elif a.name == 'select_backup' and len(args) == 2:
                    bk_by_w[str(args[1])].add(str(args[0]))

            if time_first_opt is None:
                time_first_opt = round(t_now, 4)

            # Compute per-scenario metrics for this model
            per_scenario = {}
            for w in SCENARIOS:
                prim = sorted(prim_by_w[w])
                bk   = sorted(bk_by_w[w])
                dis  = scenario_info[w]['disrupted']
                eff_caps_w = {s: 0 if s in dis else pcp.get(s,0) for s in pcp}
                cost = (sum(pc.get(s,0)*pcp.get(s,0) for s in prim)
                      + sum(bc.get(b,0)*bcp.get(b,0)  for b in bk))
                soc  = (sum(ps.get(s,0) for s in prim)
                      + sum(bs.get(b,0) for b in bk))
                env  = (sum(pe.get(s,0) for s in prim)
                      + sum(be.get(b,0) for b in bk))
                supply = (sum(eff_caps_w.get(s,0) for s in prim)
                        + sum(bcp.get(b,0) for b in bk))
                per_scenario[w] = {
                    'primary': prim, 'backup': bk,
                    'n_primary': len(prim), 'n_backup': len(bk),
                    'asp_obj': sum(bc.get(b,0) for b in bk),
                    'total_cost': cost, 'soc': soc, 'env': env,
                    'fill_rate': min(100.0, supply/demand*100),
                    'bk_cap_util': sum(bcp.get(b,0) for b in bk)/total_backup_cap*100
                                   if total_backup_cap > 0 else 0,
                }

            all_models.append(per_scenario)

    total_time = round(time.perf_counter() - t_start, 4)
    n_models   = len(all_models)

    if n_models == 0:
        return {
            "scale": scale, "demand": demand,
            "feasible": False, "n_models": 0,
            "solve_time_total_s": total_time,
            "error": "No models found"
        }

    # ── Aggregate across all stable models, per scenario ─────────────────────
    result = {
        "scale": scale, "demand": demand,
        "feasible": True,
        "n_models": n_models,
        "paper_n_models": PAPER_MODELS.get((scale, demand), "N/A"),
        "solve_time_first_opt_s": time_first_opt,
        "solve_time_total_s": total_time,
        "soc_thresh": soc_thresh,
        "env_thresh":  env_thresh,
        "total_primary_cap": total_prim_cap,
        "total_backup_cap":  total_backup_cap,
        "scenarios": {},
    }

    for w in SCENARIOS:
        bk_counts   = [m[w]['n_backup']   for m in all_models]
        prim_counts = [m[w]['n_primary']   for m in all_models]
        obj_vals    = [m[w]['asp_obj']     for m in all_models]
        costs       = [m[w]['total_cost']  for m in all_models]
        socs        = [m[w]['soc']         for m in all_models]
        envs        = [m[w]['env']         for m in all_models]
        fills       = [m[w]['fill_rate']   for m in all_models]
        bk_utils    = [m[w]['bk_cap_util'] for m in all_models]

        # Supplier frequency
        bk_freq  = defaultdict(int)
        pri_freq = defaultdict(int)
        for m in all_models:
            for b in m[w]['backup']:  bk_freq[b]  += 1
            for p in m[w]['primary']: pri_freq[p] += 1

        si = scenario_info[w]
        result["scenarios"][w] = {
            # Capacity info
            "n_disrupted":      si['n_disrupted'],
            "cap_lost":         si['cap_lost'],
            "cap_lost_pct":     si['cap_lost_pct'],
            "eff_primary_cap":  si['eff_cap'],
            "backup_needed":    si['backup_needed'],

            # Backup activation
            "asp_backups_min":        min(bk_counts),
            "asp_backups_max":        max(bk_counts),
            "asp_backups_consistent": min(bk_counts)==max(bk_counts),

            # Primary selection
            "primary_selected_min":   min(prim_counts),
            "primary_selected_max":   max(prim_counts),

            # ASP objective
            "asp_obj_value":          obj_vals[0],
            "asp_obj_consistent":     len(set(obj_vals))==1,

            # Cost range
            "total_cost_min":         min(costs),
            "total_cost_max":         max(costs),
            "total_cost_range":       max(costs)-min(costs),

            # ESG
            "soc_min":                min(socs),
            "soc_max":                max(socs),
            "soc_headroom_pct":       round((min(socs)-soc_thresh)/soc_thresh*100,1),
            "env_min":                min(envs),
            "env_max":                max(envs),
            "env_headroom_pct":       round((min(envs)-env_thresh)/env_thresh*100,1),

            # Fill rate
            "fill_rate_min":          round(min(fills),2),
            "fill_rate_max":          round(max(fills),2),

            # Backup cap utilisation
            "backup_cap_util_min_pct": round(min(bk_utils),1),
            "backup_cap_util_max_pct": round(max(bk_utils),1),

            # Supplier frequency
            "top_backup_suppliers":  [{"supplier":s,"freq":f,"freq_pct":round(f/n_models*100,1)}
                                       for s,f in sorted(bk_freq.items(),  key=lambda x:-x[1])[:10]],
            "top_primary_suppliers": [{"supplier":s,"freq":f,"freq_pct":round(f/n_models*100,1)}
                                       for s,f in sorted(pri_freq.items(), key=lambda x:-x[1])[:10]],
            "low_primary_suppliers": [{"supplier":s,"freq":f,"freq_pct":round(f/n_models*100,1)}
                                       for s,f in sorted(pri_freq.items(), key=lambda x:x[1])[:5]],
        }

    return result

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    missing = [f for f in [MODEL_FILE]+[c["instance"] for c in SCALES.values()]
               if not os.path.exists(f)]
    if missing:
        print(f"ERROR: Missing files: {missing}")
        print(f"Current directory: {os.getcwd()}")
        sys.exit(1)

    model_text  = open(MODEL_FILE).read()
    all_results = []

    hdr = (f"{'Scale':>6} {'Demand':>8} | "
           f"{'Models':>7} {'Paper':>7} {'Match':>6} | "
           f"{'1stOpt(s)':>10} {'Total(s)':>9} | "
           f"{'w1_Bk':>6} {'w2_Bk':>6} {'w3_Bk':>6}")
    print(hdr)
    print("-"*len(hdr))

    for scale, cfg in SCALES.items():
        lp_text = open(cfg["instance"]).read()
        idata   = extract_instance_data(lp_text)
        for demand in cfg["demands"]:
            print(f"{scale:>6} {demand:>8} | solving...", end="\r", flush=True)
            r = solve_instance(lp_text, idata, scale, demand, model_text)
            all_results.append(r)

            if r["feasible"]:
                paper  = r["paper_n_models"]
                match  = "✓" if r["n_models"]==paper else f"≠{paper}"
                w1_bk  = r["scenarios"]["w1"]["asp_backups_min"]
                w2_bk  = r["scenarios"]["w2"]["asp_backups_min"]
                w3_bk  = r["scenarios"]["w3"]["asp_backups_min"]
                print(f"{scale:>6} {demand:>8} | "
                      f"{r['n_models']:>7} {paper:>7} {match:>6} | "
                      f"{r['solve_time_first_opt_s']:>10.3f} {r['solve_time_total_s']:>9.3f} | "
                      f"{w1_bk:>6} {w2_bk:>6} {w3_bk:>6}")
            else:
                print(f"{scale:>6} {demand:>8} | INFEASIBLE  {r.get('error')}")

    with open("asp_results_paper.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nDone. Saved asp_results_paper.json  ({len(all_results)} instances)")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    for scale in [100, 500, 1000]:
        rows = [r for r in all_results if r["scale"]==scale and r["feasible"]]
        if not rows: continue
        print(f"\nScale {scale}:")
        print(f"  Models:          {min(r['n_models'] for r in rows):,} – {max(r['n_models'] for r in rows):,}  "
              f"(paper: {min(PAPER_MODELS.get((scale,r['demand']),0) for r in rows):,} – "
              f"{max(PAPER_MODELS.get((scale,r['demand']),0) for r in rows):,})")
        print(f"  First opt (s):   {min(r['solve_time_first_opt_s'] for r in rows):.3f} – "
              f"{max(r['solve_time_first_opt_s'] for r in rows):.3f}")
        print(f"  Total time (s):  {min(r['solve_time_total_s'] for r in rows):.3f} – "
              f"{max(r['solve_time_total_s'] for r in rows):.3f}")
        for w in SCENARIOS:
            bks  = [r["scenarios"][w]["asp_backups_min"] for r in rows]
            objs = [r["scenarios"][w]["asp_obj_value"]   for r in rows]
            prims= [r["scenarios"][w]["primary_selected_min"] for r in rows]
            costs_min = [r["scenarios"][w]["total_cost_min"] for r in rows]
            costs_max = [r["scenarios"][w]["total_cost_max"] for r in rows]
            soc_h = [r["scenarios"][w]["soc_headroom_pct"] for r in rows]
            print(f"  {w}: backups={min(bks)}-{max(bks)}  obj={min(objs)}-{max(objs)}  "
                  f"prim={min(prims)}-{max(prims)}  "
                  f"cost={min(costs_min):,}-{max(costs_max):,}  "
                  f"soc_head={min(soc_h):.0f}%-{max(soc_h):.0f}%")

if __name__ == "__main__":
    main()
