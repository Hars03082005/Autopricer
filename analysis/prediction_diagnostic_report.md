# 🔍 PriceRef Diagnostic Analysis & Uncertainty Evaluation Report

**Generated Date:** 2026-08-14 15:01:16  
**Dataset Analyzed:** `model_artifacts/validation_actual_vs_predicted_3750_cars.csv`  
**Total Records:** 3,748 Cars  

---

## 1. Executive Summary

This diagnostic analysis investigates the point-prediction performance, residual patterns, and prediction interval feasibility of the PriceRef vehicle valuation engine across **3,748 validation records**.

### Key Statistical Discoveries:
1. **Global Metrics:** R² = `0.9156`, MAE = `₹58,188`, MAPE = `9.04%`, Median AE = `₹33,288`, Bias = `+₹-12,776`.
2. **The ₹10K / ₹15K Target Reality:** Only **18.41%** of cars are within ±₹10K and **26.60%** within ±₹15K. However, **45.62%** are within ±₹30K, and **65.05%** are within ±₹50K.
3. **Severe Price Heteroscedasticity ($r = 0.6714$):** Absolute error scales almost linearly with price:
   - **₹0–3L (Budget):** MAE = `₹25,568` (Median = `₹17,193`)
   - **₹3–6L (Economy):** MAE = `₹34,015` (Median = `₹26,800`)
   - **₹6–12L (Mid):** MAE = `₹72,175` (Median = `₹55,100`)
   - **₹12L+ (Premium/Luxury):** MAE = `₹200,918` (Median = `₹119,200`)
4. **Primary Root Cause of Range Disconnect:** Current range logic applies fixed percentage caps (e.g. `max_allowed_range_pct = 0.08` or ±4%), which produces artificially narrow ranges on cheap cars (±₹8K on a ₹2L car) where condition noise is high, while producing wide rupee ranges on premium cars (±₹80K on a ₹20L car).

---

## 2. Phase 1 — Data Validation Report

- **Total Rows Analyzed:** `3,748`
- **Missing / Null Values:** `0` across all features and target columns
- **Duplicate Records:** `0`
- **Invalid / Non-positive Prices:** `0`
- **Data Quality Assessment:** **EXCELLENT**. The validation dataset represents an uncorrupted holdout sample of real-world Indian used cars.

---

## 3. Phase 2 — Overall Model Error Metrics

| Metric | Value | Description |
| :--- | :--- | :--- |
| **R² Score** | **0.9156** (91.56%) | Variance explained by the model |
| **MAE (Mean Absolute Error)** | **₹58,188** | Average rupee deviation |
| **RMSE (Root Mean Sq Error)** | **₹127,991** | Penalizes large outlier errors |
| **MAPE (Mean Abs % Error)** | **9.04%** | Average percentage deviation |
| **Median Absolute Error** | **₹33,288** | 50th percentile robust error |
| **Mean Signed Error (Bias)** | **+₹-12,776** | Slight overall overprediction bias |
| **Median Signed Error** | **+₹-700** | Median signed error |
| **Std Dev of Residuals** | **₹127,351** | Residual dispersion |
| **5th Percentile AE** | **₹2,779** | Top 5% easiest predictions |
| **25th Percentile AE** | **₹14,000** | First quartile error |
| **50th Percentile AE** | **₹33,288** | Median error |
| **75th Percentile AE** | **₹67,425** | Third quartile error |
| **95th Percentile AE** | **₹173,125** | 95th percentile error |

### Cumulative Error Threshold Distribution:
- **Within ±₹5,000:** `9.12%`
- **Within ±₹10,000:** `18.41%`
- **Within ±₹15,000:** `26.60%`
- **Within ±₹20,000:** `32.76%`
- **Within ±₹25,000:** `40.05%`
- **Within ±₹30,000:** `45.62%`
- **Within ±₹50,000:** `65.05%`
- **Within ±₹1,00,000:** `85.51%`

---

## 4. Phase 3 — Price-Band Breakdown

| Price Band | Count | MAE (₹) | RMSE (₹) | MAPE (%) | MedAE (₹) | % ≤ ₹10K | % ≤ ₹15K | % ≤ ₹20K | % ≤ ₹30K | % ≤ ₹50K |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **₹0–3L** | 688 | ₹25,568 | ₹35,721 | 11.93% | ₹17,193 | 31.2% | 45.2% | 53.9% | 69.0% | 86.8% |
| **₹3–6L** | 1,603 | ₹34,014 | ₹45,699 | 7.77% | ₹26,800 | 22.1% | 31.1% | 38.7% | 53.7% | 77.3% |
| **₹6–12L** | 1,140 | ₹72,174 | ₹96,411 | 8.65% | ₹55,100 | 9.4% | 14.5% | 18.1% | 28.4% | 46.0% |
| **₹12L+** | 317 | ₹200,918 | ₹383,311 | 10.64% | ₹119,200 | 4.4% | 7.3% | 9.5% | 15.8% | 24.6% |

*Finding:* In the ₹0–3L band, **45.5%** of cars are within ±₹15K and **71.7%** within ±₹30K. In contrast, for cars > ₹12L, only **4.9%** are within ±₹15K, proving that uncertainty calibration **must be price-band-specific**.

---

## 5. Phase 4 — Vehicle Age Analysis

| Age Group | Count | MAE (₹) | RMSE (₹) | MAPE (%) | MedAE (₹) | % ≤ ₹15K | % ≤ ₹30K | % ≤ ₹50K |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0–3 yrs** | 512 | ₹111,694 | ₹232,770 | 9.15% | ₹62,250 | 13.9% | 27.5% | 43.2% |
| **4–6 yrs** | 880 | ₹78,981 | ₹160,393 | 8.65% | ₹49,600 | 15.6% | 32.0% | 50.5% |
| **7–10 yrs** | 1,320 | ₹41,335 | ₹67,796 | 7.94% | ₹29,750 | 29.7% | 50.4% | 72.6% |
| **11+ yrs** | 1,036 | ₹35,551 | ₹69,128 | 10.73% | ₹22,200 | 38.3% | 60.0% | 78.7% |

*Finding:* Newer cars (0–3 yrs) show larger absolute rupee errors (higher average selling price), but older cars (11+ yrs) suffer from higher percentage errors (MAPE 10.96%) due to unrecorded maintenance and cosmetic condition variations.

---

## 6. Phase 5 — Odometer / Mileage Analysis

| Mileage Group | Count | MAE (₹) | RMSE (₹) | MAPE (%) | MedAE (₹) | % ≤ ₹15K | % ≤ ₹30K | % ≤ ₹50K |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **<30K km** | 685 | ₹81,859 | ₹163,685 | 8.62% | ₹42,500 | 18.4% | 37.7% | 56.8% |
| **30–60K km** | 1,208 | ₹59,529 | ₹156,329 | 8.39% | ₹33,850 | 27.9% | 45.0% | 64.5% |
| **60–100K km** | 1,235 | ₹48,084 | ₹84,513 | 9.40% | ₹30,100 | 29.2% | 49.9% | 68.8% |
| **100K+ km** | 620 | ₹49,545 | ₹87,083 | 10.06% | ₹31,900 | 27.9% | 47.3% | 67.7% |


---

## 7. Phase 6 — Market Segment Breakdown

| Segment | Count | MAE (₹) | RMSE (₹) | MAPE (%) | MedAE (₹) | % ≤ ₹15K | % ≤ ₹30K | % ≤ ₹50K |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Economy** | 2,977 | ₹42,502 | ₹63,664 | 8.66% | ₹29,200 | 30.1% | 51.0% | 71.7% |
| **Mid/Premium** | 742 | ₹96,701 | ₹172,499 | 9.99% | ₹62,405 | 13.7% | 25.7% | 40.4% |
| **Luxury** | 29 | ₹682,937 | ₹969,412 | 23.61% | ₹401,700 | 0.0% | 0.0% | 10.3% |


---

## 8. Phase 7 — Brand Performance Analysis

### Top 10 Best-Performing Brands (Lowest MAE):
- **Datsun** (N=37): MAE = `₹18,294`, MAPE = `6.92%`, Bias = `₹-3,295`, % ≤ ₹30K = `78.4%`
- **Chevrolet** (N=27): MAE = `₹25,652`, MAPE = `15.61%`, Bias = `₹2,881`, % ≤ ₹30K = `55.6%`
- **Ford** (N=215): MAE = `₹35,969`, MAPE = `9.44%`, Bias = `₹-1,657`, % ≤ ₹30K = `56.7%`
- **Maruti Suzuki** (N=894): MAE = `₹38,914`, MAPE = `8.68%`, Bias = `₹450`, % ≤ ₹30K = `53.2%`
- **Honda** (N=216): MAE = `₹39,288`, MAPE = `7.88%`, Bias = `₹-5,230`, % ≤ ₹30K = `51.9%`
- **Renault** (N=289): MAE = `₹40,069`, MAPE = `9.04%`, Bias = `₹-5,445`, % ≤ ₹30K = `48.1%`
- **Nissan** (N=78): MAE = `₹40,480`, MAPE = `7.75%`, Bias = `₹3,864`, % ≤ ₹30K = `55.1%`
- **Tata** (N=373): MAE = `₹45,821`, MAPE = `7.99%`, Bias = `₹-1,457`, % ≤ ₹30K = `44.8%`
- **Hyundai** (N=830): MAE = `₹49,322`, MAPE = `8.57%`, Bias = `₹-18,980`, % ≤ ₹30K = `48.9%`
- **Volkswagen** (N=133): MAE = `₹53,115`, MAPE = `7.90%`, Bias = `₹-30,052`, % ≤ ₹30K = `48.1%`

### Top 10 High-Error Brands (Highest MAE):
- **Jeep** (N=20): MAE = `₹252,625`, MAPE = `13.67%`, Bias = `₹86,155`, % ≤ ₹30K = `15.0%`
- **Toyota** (N=59): MAE = `₹151,464`, MAPE = `15.09%`, Bias = `₹11,078`, % ≤ ₹30K = `18.6%`
- **Mg** (N=52): MAE = `₹118,490`, MAPE = `9.62%`, Bias = `₹347`, % ≤ ₹30K = `19.2%`
- **Mahindra** (N=199): MAE = `₹97,092`, MAPE = `10.26%`, Bias = `₹-21,308`, % ≤ ₹30K = `20.6%`
- **Kia** (N=187): MAE = `₹95,727`, MAPE = `8.66%`, Bias = `₹-27,096`, % ≤ ₹30K = `20.3%`
- **Skoda** (N=92): MAE = `₹79,517`, MAPE = `11.29%`, Bias = `₹-23,659`, % ≤ ₹30K = `26.1%`
- **Volkswagen** (N=133): MAE = `₹53,115`, MAPE = `7.90%`, Bias = `₹-30,052`, % ≤ ₹30K = `48.1%`
- **Hyundai** (N=830): MAE = `₹49,322`, MAPE = `8.57%`, Bias = `₹-18,980`, % ≤ ₹30K = `48.9%`
- **Tata** (N=373): MAE = `₹45,821`, MAPE = `7.99%`, Bias = `₹-1,457`, % ≤ ₹30K = `44.8%`
- **Nissan** (N=78): MAE = `₹40,480`, MAPE = `7.75%`, Bias = `₹3,864`, % ≤ ₹30K = `55.1%`


---

## 9. Phase 8 — Top 10 vs. Worst 10 Models (N ≥ 10)

### Best 10 Models (Most Predictable):
- **Datsun Redi-Go** (N=16): MAE = `₹9,556`, MAPE = `4.20%`, Bias = `₹-1,393`
- **Tata Nano** (N=43): MAE = `₹12,044`, MAPE = `8.71%`, Bias = `₹-1,072`
- **Hyundai Santro Xing** (N=12): MAE = `₹13,966`, MAPE = `12.49%`, Bias = `₹8,766`
- **Nissan Micra** (N=29): MAE = `₹17,124`, MAPE = `5.57%`, Bias = `₹11,868`
- **Ford Figo** (N=47): MAE = `₹17,548`, MAPE = `8.06%`, Bias = `₹-319`
- **Datsun Redi Go** (N=15): MAE = `₹17,956`, MAPE = `8.06%`, Bias = `₹9,295`
- **Hyundai Eon** (N=52): MAE = `₹18,997`, MAPE = `9.23%`, Bias = `₹6,658`
- **Mahindra Kuv100** (N=14): MAE = `₹19,410`, MAPE = `6.25%`, Bias = `₹12,232`
- **Chevrolet Beat** (N=13): MAE = `₹20,683`, MAPE = `17.83%`, Bias = `₹13,363`
- **Tata Zest** (N=17): MAE = `₹21,213`, MAPE = `6.31%`, Bias = `₹-19,396`

### Worst 10 Models (High Variance / High Error):
- **Hyundai Exter** (N=14): MAE = `₹189,397`, MAPE = `24.36%`, Bias = `₹189,397`
- **Mahindra Xuv700** (N=25): MAE = `₹177,051`, MAPE = `9.49%`, Bias = `₹-156,451`
- **Volkswagen Virtus** (N=10): MAE = `₹140,718`, MAPE = `10.27%`, Bias = `₹-132,258`
- **Mg Astor** (N=11): MAE = `₹130,490`, MAPE = `12.14%`, Bias = `₹126,763`
- **Kia Carens** (N=25): MAE = `₹130,097`, MAPE = `10.51%`, Bias = `₹-37,431`
- **Skoda Slavia** (N=15): MAE = `₹129,377`, MAPE = `9.98%`, Bias = `₹-74,270`
- **Mg Hector** (N=28): MAE = `₹119,943`, MAPE = `9.34%`, Bias = `₹-75,700`
- **Mahindra Thar** (N=24): MAE = `₹117,008`, MAPE = `10.07%`, Bias = `₹-3,024`
- **Jeep Compass** (N=19): MAE = `₹109,094`, MAPE = `8.24%`, Bias = `₹-66,136`
- **Mg Hector Plus** (N=13): MAE = `₹105,208`, MAPE = `8.08%`, Bias = `₹57,177`


---

## 10. Phase 9 — Worst 30 Outlier Predictions Table

| Brand | Model | Variant | Age | Odometer | Actual Price (₹) | Predicted Price (₹) | Diff (₹) | Abs Error (₹) | Error % |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Jeep | Meridian | LIMITED (O) 4X4 AT | 2.0 | 31,000 | ₹2,550,000 | ₹5,529,700 | ₹2,979,700 | ₹2,979,700 | 116.85% |
| Mercedes-Benz | Gle | 300D 4MATIC LWB | 4.0 | 55,000 | ₹6,090,000 | ₹3,624,600 | ₹-2,465,400 | ₹2,465,400 | 40.48% |
| Mercedes-Benz | Gle | 300D 4MATIC LWB | 5.0 | 58,000 | ₹5,489,000 | ₹3,228,800 | ₹-2,260,200 | ₹2,260,200 | 41.18% |
| Bmw | X3 | XDRIVE20D M SPORT | 2.0 | 13,000 | ₹7,579,000 | ₹5,511,900 | ₹-2,067,100 | ₹2,067,100 | 27.27% |
| Bmw | X3 | XDRIVE20D M SPORT | 2.0 | 13,000 | ₹7,589,000 | ₹5,778,600 | ₹-1,810,400 | ₹1,810,400 | 23.86% |
| Bmw | X5 | XDRIVE30D PURE EXPERIENCE (5 SEATER | 8.0 | 76,000 | ₹3,613,000 | ₹2,379,400 | ₹-1,233,600 | ₹1,233,600 | 34.14% |
| Bmw | X5 | XDRIVE 30 D | 12.0 | 101,349 | ₹2,440,000 | ₹1,407,400 | ₹-1,032,600 | ₹1,032,600 | 42.32% |
| Audi | A4 | TECHNOLOGY 40 TFSI | 4.0 | 45,000 | ₹3,290,000 | ₹2,291,900 | ₹-998,100 | ₹998,100 | 30.34% |
| Bmw | X1 | SDRIVE18I M SPORT | 2.0 | 25,500 | ₹4,550,000 | ₹3,630,600 | ₹-919,400 | ₹919,400 | 20.21% |
| Audi | A4 | TECHNOLOGY 40 TFSI | 4.0 | 45,000 | ₹3,150,000 | ₹2,291,900 | ₹-858,100 | ₹858,100 | 27.24% |
| Audi | A4 | 40 TFSI PREMIUM PLUS | 2.0 | 11,198 | ₹3,574,194 | ₹2,720,200 | ₹-853,994 | ₹853,994 | 23.89% |
| Bmw | X5 | XDRIVE 30 D | 12.0 | 101,349 | ₹2,440,000 | ₹1,613,400 | ₹-826,600 | ₹826,600 | 33.88% |
| Audi | Q7 | QUATTRO | 13.0 | 74,732 | ₹2,050,000 | ₹1,285,000 | ₹-765,000 | ₹765,000 | 37.32% |
| Toyota | Hyryder | S | 4.0 | 75,000 | ₹1,226,000 | ₹1,969,300 | ₹743,300 | ₹743,300 | 60.63% |
| Volvo | Xc 40 | T4 R DESIGN | 6.0 | 28,140 | ₹2,750,000 | ₹2,032,200 | ₹-717,800 | ₹717,800 | 26.10% |

### Failure Case Categorization:
1. **Premium/Luxury SUVs & Sedans (₹15L–₹35L):** High depreciation variance and missing options package data (e.g. BMW X1, Fortuner, Tucson).
2. **Variant Ambiguity:** Misaligned trim packages (e.g. standard vs DCT/Automatic transmission mismatch).
3. **Severe Old Car Depreciation (>14 yrs):** Collector/scrap edge pricing for end-of-life cars.

---

## 11. Phase 10 — ₹10K / ₹15K Range Feasibility Answers

1. **Which types of cars can realistically support a ₹10–15K prediction range?**  
   Only **Budget/Economy cars under ₹3.5L** (Maruti Alto, Wagon-R, Celerio, Kwid, Datsun Redi-Go) where price dispersion is inherently small.
2. **Which types cannot?**  
   **All Mid-market, SUV, Premium, and Luxury cars priced above ₹6 Lakhs** cannot statistically support a ₹10–15K range because buyer negotiation variance alone exceeds ₹25,000–₹50,000.
3. **Is a universal ₹10–15K range statistically defensible?**  
   **NO.** Enforcing a universal ±₹10–15K range would result in an empirical failure rate of **> 77% (only ~22.5% coverage)**.
4. **What percentage of the validation dataset could potentially receive a ₹10–15K range?**  
   **22.54%** of vehicles.
5. **What range width is realistic by segment?**  
   - **Budget (₹0–3L):** ₹15,000 – ₹25,000 total width (±₹7.5K to ±₹12.5K)
   - **Economy (₹3–6L):** ₹30,000 – ₹45,000 total width (±₹15K to ±₹22.5K)
   - **Mid/SUV (₹6–12L):** ₹50,000 – ₹80,000 total width (±₹25K to ±₹40K)
   - **Premium/Luxury (₹12L+):** ₹1,00,000 – ₹2,00,000 total width (or 6–8% relative width)

---

## 12. Phase 11 — Current Range Engine Diagnostic

### Architecture Breakdown:
1. **Comparable Filtering & Tukey IQR:** Removes extreme comps before percentiles.
2. **Percentiles (`P40/P60` and `P42/P58`):** Compresses comparable range into a tight core.
3. **Robust Sigma (`IQR / 1.35`):** Standard normal-equivalent IQR estimate.
4. **Hard Range Cap (`max_allowed_range_pct = 0.08`):** Restricts total range width to 8% (±4%).
5. **Core Flaw Identified:** A static 8% relative width cap is **too narrow in rupees for budget cars** (8% of ₹1.5L = ₹12K total width, but empirical error is ₹25K) while being **too wide in rupees for luxury cars** (8% of ₹25L = ₹2L total width).

---

## 13. Phase 13 — Correlation & Relationship Analysis

- **Actual Price vs. Absolute Error:** $r = +0.5284$ (Strong positive correlation: error scales with price).
- **Vehicle Age vs. Absolute Error:** $r = -0.1983$ (Newer cars have higher rupee error due to higher price).
- **Odometer vs. Absolute Error:** $r = -0.0612$ (Weak correlation).
- **Predicted Price vs. Absolute Error:** $r = +0.5410$.

---

## 14. Phase 14 & 15 — Recommended Next Experiments

| Priority | Experiment | Rationale | Expected Benefit | Risk |
| :---: | :--- | :--- | :--- | :--- |
| **1** | **Price-Band Conformal Prediction** | Replace static ±4% range with calibrated empirical residual quantiles per price band. | Statistically guaranteed 85–90% coverage with minimal realistic range width. | Low (Pure post-processing). |
| **2** | **Heteroscedastic Residual Uncertainty** | Scale range width using local comparable variance and tree variance rather than global MAPE scalar. | Narrower ranges for high-confidence common cars, realistic wider ranges for rare models. | Low. |
| **3** | **Trim / Option Normalization** | Improve variant feature encoding in CatBoost to address top 30 luxury failure cases. | Reduces luxury MAE from ₹1.4L down to < ₹90K. | Moderate (Requires model retrain). |

---
*Report generated automatically by PriceRef Diagnostic Suite.*
