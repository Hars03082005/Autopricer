"""
show_buy_price.py
=================
Prints the full buy-price waterfall breakdown for a vehicle.

Usage (from project root):
    python scripts/show_buy_price.py
    python scripts/show_buy_price.py --brand Honda --model City --year 2021 \
        --km 28000 --condition Good --owners 1 --segment economy \
        --market-value 735000

Outputs every deduction component so you can see exactly how
recommended_buy_price is derived.
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.decision_engine import (
    apply_market_sanity_clamp,
    compute_dynamic_recon_cost,
    compute_holding_cost,
    compute_doc_cost,
    compute_risk_buffer,
    compute_risk_score,
    compute_confidence_score,
    dynamic_target_margin,
    classify_vehicle_category,
    _PROFIT_LIMITS,
    _clamp,
    _round500,
)


RESET  = "\033[0m"
BOLD   = "\033[1m"
CYAN   = "\033[96m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
DIM    = "\033[2m"
LINE   = "-" * 64
DOUBLE = "=" * 64


def fmt_inr(amount: float) -> str:
    """Format a rupee amount as Rs X,XX,XXX."""
    return f"Rs {int(amount):,}"


def fmt_pct(v: float) -> str:
    return f"{v:.2f}%"


def print_section(title: str):
    print(f"\n{CYAN}{BOLD}{LINE}{RESET}")
    print(f"{CYAN}{BOLD}  {title}{RESET}")
    print(f"{CYAN}{LINE}{RESET}")


def print_row(label: str, value: str, note: str = ""):
    pad = 36
    note_str = f"  {DIM}({note}){RESET}" if note else ""
    print(f"  {label:<{pad}} {BOLD}{value}{RESET}{note_str}")


def print_deduction(label: str, amount: float, note: str = ""):
    pad = 36
    note_str = f"  {DIM}({note}){RESET}" if note else ""
    print(f"  {RED}−{RESET} {label:<{pad-2}} {RED}{BOLD}{fmt_inr(amount)}{RESET}{note_str}")


def parse_args():
    p = argparse.ArgumentParser(description="Buy-price waterfall breakdown")
    p.add_argument("--brand",        default="Honda",    help="Brand name")
    p.add_argument("--model",        default="City",     help="Model name")
    p.add_argument("--year",         type=int, default=2021)
    p.add_argument("--km",           type=float, default=28_000)
    p.add_argument("--condition",    default="Good",
                   choices=["Excellent", "Good", "Average", "Poor"])
    p.add_argument("--owners",       type=int, default=1)
    p.add_argument("--fuel",         default="Petrol")
    p.add_argument("--transmission", default="Manual")
    p.add_argument("--city",         default="Bangalore")
    p.add_argument("--inspected",    action="store_true", default=False)
    p.add_argument("--loan",         action="store_true", default=False,
                   help="Loan outstanding on vehicle")
    p.add_argument("--reg-state",    default="",  dest="reg_state")
    p.add_argument("--variant",      default="")
    p.add_argument("--color",        default="White")
    p.add_argument("--accident",     default="none",
                   choices=["none","minor","major","unknown"])
    p.add_argument("--market-value", type=float, default=735_000,
                   dest="market_value",
                   help="ML-predicted market value in ₹")
    p.add_argument("--target-margin", type=float, default=15.0,
                   dest="target_margin",
                   help="Dealer target margin %% (default 15)")
    p.add_argument("--repair-buffer", type=float, default=0,
                   dest="repair_buffer",
                   help="Dealer-entered repair estimate (0 = auto-compute)")
    return p.parse_args()


def main():
    a = parse_args()

    age          = max(0, 2026 - a.year)
    brand_lc     = a.brand.lower().strip()
    condition_lc = a.condition.lower().strip()
    fuel_lc      = a.fuel.lower().strip()
    city_lc      = a.city.lower().strip()

    variant_known       = a.variant.lower() not in {"", "unknown", "base"}
    color_known         = a.color.lower() not in {"", "unknown"}
    accident_hist_known = a.accident.lower() not in {"unknown", ""}
    reg_state_known     = bool(a.reg_state)
    service_hist_known  = a.inspected

    from backend.decision_engine import _INLINE_BRAND_SEGMENT
    segment = _INLINE_BRAND_SEGMENT.get(brand_lc, "economy")

    print(f"\n{BOLD}{GREEN}{DOUBLE}{RESET}")
    print(f"{BOLD}{GREEN}  PriceRef - Buy Price Waterfall Breakdown{RESET}")
    print(f"{BOLD}{GREEN}{DOUBLE}{RESET}")

    print_section("VEHICLE INPUTS")
    print_row("Brand / Model",      f"{a.brand} {a.model}")
    print_row("Year / Age",         f"{a.year}  ({age} yrs)")
    print_row("Odometer",           f"{int(a.km):,} km")
    annual_km = a.km / max(age, 0.5)
    print_row("Annual km (derived)",f"{int(annual_km):,} km/yr")
    print_row("Condition",          a.condition)
    print_row("Owners",             str(a.owners))
    print_row("Fuel / Transmission",f"{a.fuel} / {a.transmission}")
    print_row("City",               a.city)
    print_row("Inspected",          "Yes" if a.inspected else "No")
    print_row("Segment (auto)",     segment.upper())
    print_row("Variant known",      "Yes" if variant_known else "No")
    print_row("Accident history",   a.accident)

    print_section("STEP 0 — MARKET SANITY CLAMP")
    print_row("Raw ML market value", fmt_inr(a.market_value))

    clamped_val, was_clamped, clamp_note = apply_market_sanity_clamp(
        a.model, segment, age, a.market_value, city_lc, pre_clamp_confidence=70.0
    )
    print_row("After sanity clamp",  fmt_inr(clamped_val),
              clamp_note + (" ← CLAMPED" if was_clamped else ""))
    market_value = clamped_val

    risk_score, risk_level = compute_risk_score(
        age, a.km, a.owners, condition_lc, fuel_lc, a.inspected, was_clamped,
        variant_known=variant_known,
        color_known=color_known,
        accident_history=a.accident,
    )

    conf_score, model_conf, biz_conf = compute_confidence_score(
        age, a.km, a.owners, condition_lc, fuel_lc, a.variant, 0.0,
        risk_score, was_clamped, city_lc, a.inspected,
        owner_known=True,
        accident_hist=a.accident,
    )

    clamped_val, was_clamped, clamp_note = apply_market_sanity_clamp(
        a.model, segment, age, market_value, city_lc,
        pre_clamp_confidence=float(conf_score)
    )
    market_value = clamped_val

    print_section("SCORES")
    print_row("Risk score",       f"{risk_score}/95  ({risk_level})")
    print_row("Confidence score", f"{conf_score}/100  (model={model_conf}, biz={biz_conf})")

    print_section("STEP 1 — RECONDITIONING COST")
    if a.repair_buffer > 5_000:
        recon_cost = int(a.repair_buffer)
        print_row("Mode", "Dealer-entered repair estimate")
    else:
        recon_cost = compute_dynamic_recon_cost(
            segment, age, a.km, condition_lc, a.inspected, a.brand
        )
        print_row("Mode", "Auto-computed")

    from backend.decision_engine import _RECON_BASE, _BRAND_REPAIR_MULTIPLIER
    base = _RECON_BASE.get(segment, 18_000)
    brand_mult = _BRAND_REPAIR_MULTIPLIER.get(brand_lc, 1.0)
    print_row("Segment base",         fmt_inr(base))
    print_row("Brand repair mult",    f"×{brand_mult:.2f}  ({a.brand})")
    print_row("Inspected discount",   "×0.85" if a.inspected else "none")
    print_row("RECON COST",          f"{YELLOW}{fmt_inr(recon_cost)}{RESET}")

    print_section("STEP 2 — HOLDING COST")
    holding_cost, eff_days = compute_holding_cost(segment, market_value, a.brand)
    from backend.decision_engine import _HOLDING, _BRAND_POPULARITY
    h_rate = _HOLDING.get(segment, {}).get("rate_pct", 1.8)
    pop    = _BRAND_POPULARITY.get(brand_lc, 1.0)
    print_row("Segment rate",         f"{h_rate}%/month")
    print_row("Brand popularity mult",f"×{pop:.2f}  ({a.brand})")
    print_row("Effective days",       f"{eff_days} days")
    print_row("Formula",              f"MV × {h_rate}% × ({eff_days}/30)")
    print_row("HOLDING COST",         f"{YELLOW}{fmt_inr(holding_cost)}{RESET}")

    print_section("STEP 3 — DOCUMENTATION COST")
    doc_cost, doc_breakdown = compute_doc_cost(a.reg_state, city_lc, a.loan)
    for k, v in doc_breakdown.items():
        if v > 0:
            print_row(f"  {k.replace('_',' ').title()}", fmt_inr(v))
    print_row("DOC COST",             f"{YELLOW}{fmt_inr(doc_cost)}{RESET}")

    print_section("STEP 4 — RISK BUFFER")
    risk_buffer = compute_risk_buffer(
        market_value, risk_score, segment, age, a.km, a.owners, condition_lc, a.inspected,
        variant_known=variant_known,
        owner_known=True,
        service_hist_known=service_hist_known,
        accident_hist_known=accident_hist_known,
        reg_state_known=reg_state_known,
        color_known=color_known,
    )
    print_row("Risk score used",      f"{risk_score}/95")
    print_row("Base formula",         "MV × risk_score × 0.0001 × seg_factor")
    if not variant_known:   print_row("  + Variant unknown",    "₹1,500")
    if not color_known:     print_row("  + Color unknown",      "  ₹500")
    if not accident_hist_known: print_row("  + Accident hist unknown","₹3,000")
    if not service_hist_known:  print_row("  + Service hist unknown","₹1,500")
    if not reg_state_known: print_row("  + Reg state unknown",  "₹1,000")
    print_row("RISK BUFFER",          f"{YELLOW}{fmt_inr(risk_buffer)}{RESET}")

    print_section("STEP 5 — TARGET DEALER PROFIT")
    eff_margin = dynamic_target_margin(
        segment, age, a.km, a.owners, condition_lc, a.inspected, fuel_lc, a.target_margin
    )
    veh_cat     = classify_vehicle_category(a.brand, a.model)
    p_min, p_max = _PROFIT_LIMITS.get(veh_cat, (25_000, 100_000))
    raw_profit  = market_value * (eff_margin / 100)
    target_profit = int(_clamp(raw_profit, p_min, p_max))

    print_row("Vehicle category",     veh_cat)
    print_row("Effective margin %",   fmt_pct(eff_margin),
              f"blend of computed + user {fmt_pct(a.target_margin)}")
    print_row("Raw profit (MV × %)",  fmt_inr(raw_profit))
    print_row("Profit limits",        f"{fmt_inr(p_min)} – {fmt_inr(p_max)}")
    print_row("TARGET PROFIT",        f"{YELLOW}{fmt_inr(target_profit)}{RESET}")

    print_section("WATERFALL SUMMARY")
    total_deductions = recon_cost + holding_cost + doc_cost + risk_buffer + target_profit
    raw_buy          = market_value - total_deductions
    floored_buy      = max(market_value * 0.88, raw_buy)
    final_buy        = _round500(floored_buy)

    print_row("Market Value (clamped)",   fmt_inr(market_value))
    print_deduction("Reconditioning",        recon_cost)
    print_deduction("Holding cost",          holding_cost)
    print_deduction("Documentation",         doc_cost)
    print_deduction("Risk buffer",           risk_buffer)
    print_deduction("Target dealer profit",  target_profit)
    print(f"  {'-'*60}")
    print_row("Raw buy price",              fmt_inr(raw_buy))
    if floored_buy > raw_buy:
        print_row("Floor applied (88% of MV)", fmt_inr(floored_buy),
                  "offer too low — raised to stay competitive")
    print(f"\n  {GREEN}{BOLD}{'RECOMMENDED BUY PRICE':<36} {fmt_inr(final_buy)}{RESET}")
    print(f"  {DIM}(rounded to nearest Rs 500){RESET}")

    from backend.decision_engine import _CITY_DEMAND
    city_prem = _CITY_DEMAND.get(city_lc, 0.015)
    sell_price = _round500(market_value * (1 + city_prem * 0.5))
    exp_profit = max(
        int(sell_price - final_buy - recon_cost - holding_cost - doc_cost),
        target_profit
    )
    exp_margin = (exp_profit / max(final_buy, 1)) * 100

    print_section("DEAL SUMMARY")
    print_row("Recommended BUY price",  fmt_inr(final_buy))
    print_row("Recommended SELL price", fmt_inr(sell_price))
    print_row("Expected profit",        fmt_inr(exp_profit))
    print_row("Expected margin",        fmt_pct(exp_margin))

    print(f"\n{BOLD}{GREEN}{DOUBLE}{RESET}\n")


if __name__ == "__main__":
    main()
