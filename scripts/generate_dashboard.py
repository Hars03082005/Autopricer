"""
generate_dashboard.py
=====================
Generates a comprehensive, clean, and crystal-clear diagnostic dashboard
containing BOTH:
  1. Lakh-by-Lakh Price Segment Under/Over-prediction Analysis
  2. Brand-by-Brand Under/Over-prediction Analysis
"""

from __future__ import annotations
import json
import math
import sys
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "model_artifacts"
ANALYSIS_DIR = ROOT / "analysis"
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

val_csv = ARTIFACT / "validation_actual_vs_predicted_3750_cars.csv"
df = pd.read_csv(val_csv)
df.rename(columns={
    "Brand": "brand", "Model": "model", "Variant": "variant",
    "Age (Yrs)": "vehicle_age", "Odometer (KM)": "odometer_reading",
    "Fuel": "fuel_type", "Transmission": "transmission",
    "Actual Price (₹)": "actual_price",
    "Predicted Price (₹)": "predicted_price",
    "Difference (₹)": "difference", "Error (%)": "error_pct",
}, inplace=True)

# Calculation:
# Difference = Actual Price - Predicted Price
# > 0 -> UNDERPREDICTION (Actual price is higher than predicted)
# < 0 -> OVERPREDICTION (Predicted price is higher than actual)
df["difference"] = df["actual_price"] - df["predicted_price"]
df["abs_difference"] = df["difference"].abs()
df["is_underpredicted"] = df["difference"] > 0
df["is_overpredicted"] = df["difference"] < 0
df["pct_error"] = (df["abs_difference"] / df["actual_price"]) * 100

# ── 1. LAKH-BY-LAKH PRICE SEGMENT STATS ──────────────────────────────────────
bins = [0, 100000, 200000, 300000, 400000, 500000, 600000, 700000, 800000, 900000, 1000000, 1200000, 1500000, 2000000, 5000000]
labels = [
    "₹0–1L", "₹1–2L", "₹2–3L", "₹3–4L", "₹4–5L", "₹5–6L",
    "₹6–7L", "₹7–8L", "₹8–9L", "₹9–10L", "₹10–12L", "₹12–15L", "₹15–20L", "₹20L+"
]

df["price_segment"] = pd.cut(df["actual_price"], bins=bins, labels=labels, right=True)

segment_stats = []
for lbl in labels:
    sub = df[df["price_segment"] == lbl]
    if len(sub) > 0:
        n = len(sub)
        avg_act = float(sub["actual_price"].mean())
        avg_pred = float(sub["predicted_price"].mean())
        avg_diff = float(sub["difference"].mean())
        mae = float(sub["abs_difference"].mean())
        mape = float(sub["pct_error"].mean())
        
        n_under = int(sub["is_underpredicted"].sum())
        n_over = int(sub["is_overpredicted"].sum())
        pct_under = round((n_under / n) * 100, 1)
        pct_over = round((n_over / n) * 100, 1)
        
        if avg_diff > 15000:
            status = "UNDERPREDICTING"
        elif avg_diff < -15000:
            status = "OVERPREDICTING"
        else:
            status = "BALANCED / ACCURATE"

        segment_stats.append({
            "segment": lbl,
            "count": n,
            "avg_actual": round(avg_act),
            "avg_predicted": round(avg_pred),
            "avg_difference": round(avg_diff),
            "mae": round(mae),
            "mape": round(mape, 1),
            "pct_under": pct_under,
            "pct_over": pct_over,
            "status": status
        })

# ── 2. BRAND-BY-BRAND STATS (All Brands with N >= 10) ────────────────────────
brand_counts = df["brand"].value_counts()
eligible_brands = brand_counts[brand_counts >= 10].index.tolist()

brand_stats = []
for b in eligible_brands:
    sub = df[df["brand"] == b]
    n = len(sub)
    avg_act = float(sub["actual_price"].mean())
    avg_pred = float(sub["predicted_price"].mean())
    avg_diff = float(sub["difference"].mean())
    mae = float(sub["abs_difference"].mean())
    mape = float(sub["pct_error"].mean())
    
    n_under = int(sub["is_underpredicted"].sum())
    n_over = int(sub["is_overpredicted"].sum())
    pct_under = round((n_under / n) * 100, 1)
    pct_over = round((n_over / n) * 100, 1)
    
    if avg_diff > 15000:
        status = "UNDERPREDICTING"
    elif avg_diff < -15000:
        status = "OVERPREDICTING"
    else:
        status = "BALANCED / ACCURATE"

    brand_stats.append({
        "brand": b.title(),
        "count": n,
        "avg_actual": round(avg_act),
        "avg_predicted": round(avg_pred),
        "avg_difference": round(avg_diff),
        "mae": round(mae),
        "mape": round(mape, 1),
        "pct_under": pct_under,
        "pct_over": pct_over,
        "status": status
    })

# Sort brand stats by absolute bias / difference descending
df_brand_sorted = pd.DataFrame(brand_stats).sort_values("avg_difference", ascending=False)
brand_stats_sorted = df_brand_sorted.to_dict(orient="records")

# Global KPIs
total_cars = len(df)
total_under = int(df["is_underpredicted"].sum())
total_over = int(df["is_overpredicted"].sum())
pct_total_under = round((total_under / total_cars) * 100, 1)
pct_total_over = round((total_over / total_cars) * 100, 1)
global_mae = round(float(df["abs_difference"].mean()))
global_avg_diff = round(float(df["difference"].mean()))

dashboard_data = {
    "global": {
        "total_cars": total_cars,
        "global_mae": global_mae,
        "global_avg_diff": global_avg_diff,
        "total_under": total_under,
        "total_over": total_over,
        "pct_total_under": pct_total_under,
        "pct_total_over": pct_total_over,
    },
    "segments": segment_stats,
    "brands": brand_stats_sorted
}

html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Underprediction vs Overprediction — Price & Brand Breakdown — PriceRef</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap" rel="stylesheet">
  <style>
    :root {{
      --bg: #0b0f19;
      --card-bg: #131b2e;
      --card-border: #1f2d4d;
      --text: #f8fafc;
      --text-muted: #94a3b8;
      --under: #ef4444; /* Red: Model predicts lower than actual (Underpredicting) */
      --over: #3b82f6;  /* Blue: Model predicts higher than actual (Overpredicting) */
      --balanced: #10b981; /* Green */
      --accent: #38bdf8;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: 'Plus Jakarta Sans', sans-serif;
      background: var(--bg);
      color: var(--text);
      padding: 24px;
      line-height: 1.5;
    }}
    .header {{
      margin-bottom: 24px;
      padding-bottom: 16px;
      border-bottom: 1px solid var(--card-border);
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 12px;
    }}
    .header h1 {{
      font-size: 24px;
      font-weight: 800;
      color: #38bdf8;
    }}
    .header .subtitle {{
      color: var(--text-muted);
      font-size: 14px;
      margin-top: 4px;
    }}
    .nav-tabs {{
      display: flex;
      gap: 12px;
      margin-bottom: 24px;
      border-bottom: 1px solid var(--card-border);
      padding-bottom: 12px;
    }}
    .tab-btn {{
      background: rgba(30, 41, 59, 0.6);
      color: var(--text-muted);
      border: 1px solid var(--card-border);
      padding: 10px 20px;
      border-radius: 8px;
      font-weight: 700;
      font-size: 13px;
      cursor: pointer;
      transition: all 0.2s;
    }}
    .tab-btn.active {{
      background: #38bdf8;
      color: #0b0f19;
      border-color: #38bdf8;
    }}
    .summary-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 16px;
      margin-bottom: 24px;
    }}
    .summary-card {{
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 12px;
      padding: 16px;
    }}
    .summary-title {{
      font-size: 11px;
      font-weight: 700;
      color: var(--text-muted);
      text-transform: uppercase;
      letter-spacing: 0.5px;
    }}
    .summary-value {{
      font-size: 24px;
      font-weight: 800;
      margin-top: 4px;
    }}
    .text-under {{ color: var(--under); }}
    .text-over {{ color: var(--over); }}
    .text-green {{ color: var(--balanced); }}

    .section-title {{
      font-size: 18px;
      font-weight: 800;
      margin: 28px 0 16px 0;
      color: #e2e8f0;
      display: flex;
      align-items: center;
      gap: 10px;
    }}
    .section-title::before {{
      content: '';
      display: inline-block;
      width: 4px;
      height: 20px;
      background: var(--accent);
      border-radius: 2px;
    }}
    .charts-grid {{
      display: grid;
      grid-template-columns: 1fr;
      gap: 24px;
      margin-bottom: 32px;
    }}
    .chart-card {{
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 14px;
      padding: 20px;
    }}
    .chart-title {{
      font-size: 16px;
      font-weight: 700;
      color: #fff;
    }}
    .chart-desc {{
      font-size: 13px;
      color: var(--text-muted);
      margin-bottom: 16px;
    }}
    .chart-box {{
      position: relative;
      height: 380px;
      width: 100%;
    }}
    .table-card {{
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 14px;
      padding: 20px;
      overflow-x: auto;
      margin-bottom: 24px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
      text-align: left;
    }}
    th {{
      background: rgba(30, 41, 59, 0.7);
      color: #94a3b8;
      padding: 12px 14px;
      font-weight: 700;
      border-bottom: 1px solid var(--card-border);
      font-size: 11px;
      text-transform: uppercase;
    }}
    td {{
      padding: 12px 14px;
      border-bottom: 1px solid rgba(31, 41, 61, 0.7);
      color: #e2e8f0;
    }}
    tr:hover td {{
      background: rgba(56, 189, 248, 0.05);
    }}
    .badge {{
      display: inline-block;
      padding: 4px 8px;
      border-radius: 6px;
      font-size: 11px;
      font-weight: 700;
    }}
    .badge-under {{ background: rgba(239, 68, 68, 0.15); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.3); }}
    .badge-over {{ background: rgba(59, 130, 246, 0.15); color: #60a5fa; border: 1px solid rgba(59, 130, 246, 0.3); }}
    .badge-green {{ background: rgba(16, 185, 129, 0.15); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.3); }}
    .legend-box {{
      display: flex;
      gap: 20px;
      margin-top: 10px;
      font-size: 12px;
      color: var(--text-muted);
      flex-wrap: wrap;
    }}
    .legend-item {{
      display: flex;
      align-items: center;
      gap: 6px;
    }}
    .dot {{
      width: 10px;
      height: 10px;
      border-radius: 50%;
    }}
  </style>
</head>
<body>

  <div class="header">
    <div>
      <h1>PriceRef Error & Bias Diagnostic Dashboard</h1>
      <div class="subtitle">Detailed Underprediction vs. Overprediction Breakdown by Price Segment & Vehicle Brand (3,748 Cars)</div>
    </div>
  </div>

  <!-- SUMMARY CARDS -->
  <div class="summary-grid">
    <div class="summary-card">
      <div class="summary-title">Total Validation Cars</div>
      <div class="summary-value">{total_cars:,}</div>
    </div>
    <div class="summary-card">
      <div class="summary-title">Cars Underpredicted (Actual > Pred)</div>
      <div class="summary-value text-under">{total_under:,} <span style="font-size: 13px; font-weight: normal;">({pct_total_under}%)</span></div>
    </div>
    <div class="summary-card">
      <div class="summary-title">Cars Overpredicted (Pred > Actual)</div>
      <div class="summary-value text-over">{total_over:,} <span style="font-size: 13px; font-weight: normal;">({pct_total_over}%)</span></div>
    </div>
    <div class="summary-card">
      <div class="summary-title">Overall Average Rupee Bias</div>
      <div class="summary-value {'text-under' if global_avg_diff > 0 else 'text-over'}">{'+' if global_avg_diff > 0 else ''}₹{global_avg_diff:,}</div>
    </div>
  </div>

  <!-- ==================== SECTION 1: PRICE SEGMENT ANALYSIS ==================== -->
  <div class="section-title">1. Lakh-by-Lakh Price Segment Breakdown</div>

  <div class="charts-grid">

    <!-- CHART 1: Average Rupee Difference (Under vs Over by Price) -->
    <div class="chart-card">
      <div class="chart-title">Average Rupee Gap per Lakh Segment (Actual - Predicted)</div>
      <div class="chart-desc">Red Bars (Above 0) = Underpredicting (Model predicts TOO LOW) | Blue Bars (Below 0) = Overpredicting (Model predicts TOO HIGH)</div>
      <div class="chart-box">
        <canvas id="chartBiasLakh"></canvas>
      </div>
      <div class="legend-box">
        <div class="legend-item"><div class="dot" style="background: var(--under)"></div> <strong>Red Bars (Above 0):</strong> Underpredicting (Model is cheaper than market)</div>
        <div class="legend-item"><div class="dot" style="background: var(--over)"></div> <strong>Blue Bars (Below 0):</strong> Overpredicting (Model is higher than market)</div>
      </div>
    </div>

    <!-- CHART 2: Actual vs Predicted Price Comparison -->
    <div class="chart-card">
      <div class="chart-title">Average Actual Price vs. Average Predicted Price (Lakh to Lakh)</div>
      <div class="chart-desc">Side-by-side comparison of real market selling price vs model prediction</div>
      <div class="chart-box">
        <canvas id="chartActualVsPredLakh"></canvas>
      </div>
    </div>

    <!-- CHART 3: Percentage Underpredicted vs Overpredicted by Price -->
    <div class="chart-card">
      <div class="chart-title">Percentage of Cars Underpredicted vs. Overpredicted in Each Price Segment</div>
      <div class="chart-desc">Shows the proportion of cars receiving lower-than-actual vs higher-than-actual valuations</div>
      <div class="chart-box">
        <canvas id="chartPctSplit"></canvas>
      </div>
    </div>

  </div>

  <!-- TABLE 1: PRICE SEGMENT TABLE -->
  <div class="table-card">
    <div class="chart-title" style="margin-bottom: 12px;">Price Segment Breakdown Table (Lakh to Lakh)</div>
    <table>
      <thead>
        <tr>
          <th>Price Segment</th>
          <th>Car Count</th>
          <th>Avg Actual Price</th>
          <th>Avg Predicted Price</th>
          <th>Avg Error Gap (Actual - Pred)</th>
          <th>% Underpredicted</th>
          <th>% Overpredicted</th>
          <th>Avg MAE</th>
          <th>Verdict</th>
        </tr>
      </thead>
      <tbody>
        {"".join([f'''
        <tr>
          <td><strong>{s['segment']}</strong></td>
          <td>{s['count']:,}</td>
          <td>₹{s['avg_actual']:,}</td>
          <td>₹{s['avg_predicted']:,}</td>
          <td style="font-weight: 700; color: {'#f87171' if s['avg_difference'] > 0 else '#60a5fa'}">
            {'+' if s['avg_difference'] > 0 else ''}₹{s['avg_difference']:,}
          </td>
          <td><span style="color: #f87171; font-weight: 600;">{s['pct_under']}%</span></td>
          <td><span style="color: #60a5fa; font-weight: 600;">{s['pct_over']}%</span></td>
          <td>₹{s['mae']:,} ({s['mape']}%)</td>
          <td><span class="badge badge-{'under' if s['status']=='UNDERPREDICTING' else ('over' if s['status']=='OVERPREDICTING' else 'green')}">{s['status']}</span></td>
        </tr>
        ''' for s in dashboard_data['segments']])}
      </tbody>
    </table>
  </div>

  <!-- ==================== SECTION 2: BRAND ANALYSIS ==================== -->
  <div class="section-title">2. Brand-by-Brand Error & Bias Breakdown</div>

  <div class="charts-grid">

    <!-- CHART 4: Average Rupee Difference by Brand -->
    <div class="chart-card">
      <div class="chart-title">Average Rupee Gap by Brand (Actual - Predicted)</div>
      <div class="chart-desc">Which brands are we systematically underpredicting (Red) or overpredicting (Blue)? (Brands with N ≥ 10)</div>
      <div class="chart-box" style="height: 480px;">
        <canvas id="chartBrandBias"></canvas>
      </div>
      <div class="legend-box">
        <div class="legend-item"><div class="dot" style="background: var(--under)"></div> <strong>Red Bars (Above 0):</strong> Underpredicting brand (e.g. BMW, Toyota, Mercedes)</div>
        <div class="legend-item"><div class="dot" style="background: var(--over)"></div> <strong>Blue Bars (Below 0):</strong> Overpredicting brand (e.g. Chevrolet, Datsun, Ford)</div>
      </div>
    </div>

    <!-- CHART 5: Actual vs Predicted Price Comparison by Brand -->
    <div class="chart-card">
      <div class="chart-title">Average Actual Price vs. Average Predicted Price by Brand</div>
      <div class="chart-desc">Side-by-side comparison of actual selling prices vs model predictions for every brand</div>
      <div class="chart-box" style="height: 480px;">
        <canvas id="chartBrandActualPred"></canvas>
      </div>
    </div>

    <!-- CHART 6: Brand-wise Under vs Over Prediction Split (%) -->
    <div class="chart-card">
      <div class="chart-title">Percentage of Cars Underpredicted vs. Overpredicted by Brand</div>
      <div class="chart-desc">What percentage of cars within each brand are under-valued vs over-valued?</div>
      <div class="chart-box" style="height: 480px;">
        <canvas id="chartBrandPctSplit"></canvas>
      </div>
    </div>

  </div>

  <!-- TABLE 2: BRAND BREAKDOWN TABLE -->
  <div class="table-card">
    <div class="chart-title" style="margin-bottom: 12px;">Complete Brand Breakdown Table (Ranked from Most Underpredicted to Most Overpredicted)</div>
    <table>
      <thead>
        <tr>
          <th>Brand Name</th>
          <th>Car Count</th>
          <th>Avg Actual Price</th>
          <th>Avg Predicted Price</th>
          <th>Avg Error Gap (Actual - Pred)</th>
          <th>% Underpredicted</th>
          <th>% Overpredicted</th>
          <th>Avg MAE</th>
          <th>Verdict</th>
        </tr>
      </thead>
      <tbody>
        {"".join([f'''
        <tr>
          <td><strong>{b['brand']}</strong></td>
          <td>{b['count']:,}</td>
          <td>₹{b['avg_actual']:,}</td>
          <td>₹{b['avg_predicted']:,}</td>
          <td style="font-weight: 700; color: {'#f87171' if b['avg_difference'] > 0 else '#60a5fa'}">
            {'+' if b['avg_difference'] > 0 else ''}₹{b['avg_difference']:,}
          </td>
          <td><span style="color: #f87171; font-weight: 600;">{b['pct_under']}%</span></td>
          <td><span style="color: #60a5fa; font-weight: 600;">{b['pct_over']}%</span></td>
          <td>₹{b['mae']:,} ({b['mape']}%)</td>
          <td><span class="badge badge-{'under' if b['status']=='UNDERPREDICTING' else ('over' if b['status']=='OVERPREDICTING' else 'green')}">{b['status']}</span></td>
        </tr>
        ''' for b in dashboard_data['brands']])}
      </tbody>
    </table>
  </div>

  <script>
    const segs = {json.dumps(dashboard_data['segments'])};
    const segLabels = segs.map(s => s.segment);

    const brands = {json.dumps(dashboard_data['brands'])};
    const brandLabels = brands.map(b => b.brand);

    // ================== PRICE SEGMENT CHARTS ==================
    // Chart 1: Price Bias
    new Chart(document.getElementById('chartBiasLakh'), {{
      type: 'bar',
      data: {{
        labels: segLabels,
        datasets: [{{
          label: 'Average Rupee Difference (Actual - Predicted)',
          data: segs.map(s => s.avg_difference),
          backgroundColor: segs.map(s => s.avg_difference >= 0 ? '#ef4444' : '#3b82f6'),
          borderRadius: 6
        }}]
      }},
      options: {{
        responsive: true, maintainAspectRatio: false,
        plugins: {{
          legend: {{ display: false }},
          tooltip: {{
            callbacks: {{
              label: function(ctx) {{
                const val = ctx.raw;
                if (val > 0) return ` Underpredicting by ₹${{val.toLocaleString()}}`;
                if (val < 0) return ` Overpredicting by ₹${{Math.abs(val).toLocaleString()}}`;
                return ` Exact match`;
              }}
            }}
          }}
        }},
        scales: {{
          x: {{ grid: {{ color: '#1e293b' }}, ticks: {{ color: '#94a3b8' }} }},
          y: {{
            title: {{ display: true, text: 'Rupee Gap (₹) [+ Underpredict, - Overpredict]', color: '#94a3b8' }},
            grid: {{ color: '#1e293b' }}, ticks: {{ color: '#94a3b8' }}
          }}
        }}
      }}
    }});

    // Chart 2: Price Actual vs Pred
    new Chart(document.getElementById('chartActualVsPredLakh'), {{
      type: 'bar',
      data: {{
        labels: segLabels,
        datasets: [
          {{ label: 'Avg Actual Price (₹)', data: segs.map(s => s.avg_actual), backgroundColor: '#10b981', borderRadius: 6 }},
          {{ label: 'Avg Predicted Price (₹)', data: segs.map(s => s.avg_predicted), backgroundColor: '#f59e0b', borderRadius: 6 }}
        ]
      }},
      options: {{
        responsive: true, maintainAspectRatio: false,
        plugins: {{ legend: {{ labels: {{ color: '#94a3b8' }} }} }},
        scales: {{
          x: {{ grid: {{ color: '#1e293b' }}, ticks: {{ color: '#94a3b8' }} }},
          y: {{ title: {{ display: true, text: 'Price (₹)', color: '#94a3b8' }}, grid: {{ color: '#1e293b' }}, ticks: {{ color: '#94a3b8' }} }}
        }}
      }}
    }});

    // Chart 3: Price % Split
    new Chart(document.getElementById('chartPctSplit'), {{
      type: 'bar',
      data: {{
        labels: segLabels,
        datasets: [
          {{ label: '% Underpredicted (Model predicts lower)', data: segs.map(s => s.pct_under), backgroundColor: '#ef4444', borderRadius: 4 }},
          {{ label: '% Overpredicted (Model predicts higher)', data: segs.map(s => s.pct_over), backgroundColor: '#3b82f6', borderRadius: 4 }}
        ]
      }},
      options: {{
        responsive: true, maintainAspectRatio: false,
        plugins: {{ legend: {{ labels: {{ color: '#94a3b8' }} }} }},
        scales: {{
          x: {{ stacked: true, grid: {{ color: '#1e293b' }}, ticks: {{ color: '#94a3b8' }} }},
          y: {{ stacked: true, max: 100, title: {{ display: true, text: 'Percentage of Cars (%)', color: '#94a3b8' }}, grid: {{ color: '#1e293b' }}, ticks: {{ color: '#94a3b8' }} }}
        }}
      }}
    }});

    // ================== BRAND CHARTS ==================
    // Chart 4: Brand Bias
    new Chart(document.getElementById('chartBrandBias'), {{
      type: 'bar',
      data: {{
        labels: brandLabels,
        datasets: [{{
          label: 'Average Rupee Difference (Actual - Predicted)',
          data: brands.map(b => b.avg_difference),
          backgroundColor: brands.map(b => b.avg_difference >= 0 ? '#ef4444' : '#3b82f6'),
          borderRadius: 6
        }}]
      }},
      options: {{
        responsive: true, maintainAspectRatio: false,
        indexAxis: 'y',
        plugins: {{
          legend: {{ display: false }},
          tooltip: {{
            callbacks: {{
              label: function(ctx) {{
                const val = ctx.raw;
                if (val > 0) return ` Underpredicting by ₹${{val.toLocaleString()}}`;
                if (val < 0) return ` Overpredicting by ₹${{Math.abs(val).toLocaleString()}}`;
                return ` Exact match`;
              }}
            }}
          }}
        }},
        scales: {{
          x: {{ title: {{ display: true, text: 'Rupee Gap (₹) [+ Underpredict, - Overpredict]', color: '#94a3b8' }}, grid: {{ color: '#1e293b' }}, ticks: {{ color: '#94a3b8' }} }},
          y: {{ grid: {{ display: false }}, ticks: {{ color: '#cbd5e1', font: {{ weight: '600' }} }} }}
        }}
      }}
    }});

    // Chart 5: Brand Actual vs Pred
    new Chart(document.getElementById('chartBrandActualPred'), {{
      type: 'bar',
      data: {{
        labels: brandLabels,
        datasets: [
          {{ label: 'Avg Actual Price (₹)', data: brands.map(b => b.avg_actual), backgroundColor: '#10b981', borderRadius: 6 }},
          {{ label: 'Avg Predicted Price (₹)', data: brands.map(b => b.avg_predicted), backgroundColor: '#f59e0b', borderRadius: 6 }}
        ]
      }},
      options: {{
        responsive: true, maintainAspectRatio: false,
        indexAxis: 'y',
        plugins: {{ legend: {{ labels: {{ color: '#94a3b8' }} }} }},
        scales: {{
          x: {{ title: {{ display: true, text: 'Price (₹)', color: '#94a3b8' }}, grid: {{ color: '#1e293b' }}, ticks: {{ color: '#94a3b8' }} }},
          y: {{ grid: {{ display: false }}, ticks: {{ color: '#cbd5e1', font: {{ weight: '600' }} }} }}
        }}
      }}
    }});

    // Chart 6: Brand % Split
    new Chart(document.getElementById('chartBrandPctSplit'), {{
      type: 'bar',
      data: {{
        labels: brandLabels,
        datasets: [
          {{ label: '% Underpredicted (Model predicts lower)', data: brands.map(b => b.pct_under), backgroundColor: '#ef4444', borderRadius: 4 }},
          {{ label: '% Overpredicted (Model predicts higher)', data: brands.map(b => b.pct_over), backgroundColor: '#3b82f6', borderRadius: 4 }}
        ]
      }},
      options: {{
        responsive: true, maintainAspectRatio: false,
        indexAxis: 'y',
        plugins: {{ legend: {{ labels: {{ color: '#94a3b8' }} }} }},
        scales: {{
          x: {{ stacked: true, max: 100, title: {{ display: true, text: 'Percentage of Cars (%)', color: '#94a3b8' }}, grid: {{ color: '#1e293b' }}, ticks: {{ color: '#94a3b8' }} }},
          y: {{ stacked: true, grid: {{ display: false }}, ticks: {{ color: '#cbd5e1', font: {{ weight: '600' }} }} }}
        }}
      }}
    }});
  </script>

</body>
</html>
"""

dashboard_file = ANALYSIS_DIR / "diagnostic_dashboard.html"
with open(dashboard_file, "w", encoding="utf-8") as f:
    f.write(html_content)

print(f"Price + Brand Diagnostic Dashboard updated successfully at: {dashboard_file}")
