# PriceRef - Complete Vehicle Price & Brand Analysis Report

**Generated:** 2026-08-24 21:08:16  
**Dataset:** data/data.csv | **Total Listings:** 25,158

---

## 1. Overall Dataset Summary

| Metric | Value |
| :--- | :--- |
| Total Listings | **25,158** |
| Mean Price | **Rs.6.15L** |
| Median Price | **Rs.5.09L** |
| Std Deviation | **Rs.4.17L** |
| Min Price | **Rs.50.0K** |
| Max Price | **Rs.75.99L** |
| P10 | **Rs.2.38L** |
| P25 | **Rs.3.40L** |
| P75 | **Rs.7.70L** |
| P90 | **Rs.11.30L** |
| P95 | **Rs.13.89L** |

### Dataset Splits

| Split | Count | Share |
| :--- | :---: | :---: |
| Train | 17,632 | 70.1% |
| Valid | 3,778 | 15.0% |
| Test | 3,748 | 14.9% |

---

## 2. Price-Band Breakdown

### 2a. Volume & Pricing Stats

| Band | Count | Share | Mean | Median | Std Dev | Min | Max |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0–1L** | 76 | 0.3% | Rs.82.5K | Rs.83.0K | Rs.12.3K | Rs.50.0K | Rs.99.0K |
| **1–2L** | 1,210 | 4.8% | Rs.1.63L | Rs.1.68L | Rs.26.8K | Rs.1.00L | Rs.2.00L |
| **2–3L** | 3,296 | 13.1% | Rs.2.51L | Rs.2.50L | Rs.29.2K | Rs.2.00L | Rs.3.00L |
| **3–4L** | 4,075 | 16.2% | Rs.3.49L | Rs.3.50L | Rs.28.5K | Rs.3.00L | Rs.4.00L |
| **4–5L** | 3,558 | 14.1% | Rs.4.47L | Rs.4.47L | Rs.29.6K | Rs.4.00L | Rs.5.00L |
| **5–6L** | 3,095 | 12.3% | Rs.5.46L | Rs.5.45L | Rs.29.0K | Rs.5.00L | Rs.6.00L |
| **6–8L** | 3,991 | 15.9% | Rs.6.91L | Rs.6.89L | Rs.57.3K | Rs.6.00L | Rs.8.00L |
| **8–10L** | 2,334 | 9.3% | Rs.8.88L | Rs.8.83L | Rs.57.7K | Rs.8.00L | Rs.10.00L |
| **10–12L** | 1,379 | 5.5% | Rs.10.91L | Rs.10.90L | Rs.55.5K | Rs.10.00L | Rs.11.99L |
| **12–15L** | 1,254 | 5.0% | Rs.13.32L | Rs.13.20L | Rs.86.6K | Rs.12.00L | Rs.14.99L |
| **15–20L** | 689 | 2.7% | Rs.16.88L | Rs.16.58L | Rs.1.38L | Rs.15.00L | Rs.20.00L |
| **20–30L** | 140 | 0.6% | Rs.23.09L | Rs.22.21L | Rs.2.59L | Rs.20.00L | Rs.30.00L |
| **30L+** | 61 | 0.2% | Rs.41.05L | Rs.37.13L | Rs.12.60L | Rs.30.23L | Rs.75.99L |

### 2b. Percentile Distribution per Band

| Band | P10 | P25 | P50 (Median) | P75 | P90 | P95 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **0–1L** | Rs.65.0K | Rs.76.5K | Rs.83.0K | Rs.92.0K | Rs.96.5K | Rs.98.7K |
| **1–2L** | Rs.1.24L | Rs.1.45L | Rs.1.68L | Rs.1.86L | Rs.1.94L | Rs.1.97L |
| **2–3L** | Rs.2.10L | Rs.2.25L | Rs.2.50L | Rs.2.77L | Rs.2.90L | Rs.2.95L |
| **3–4L** | Rs.3.10L | Rs.3.25L | Rs.3.50L | Rs.3.73L | Rs.3.90L | Rs.3.94L |
| **4–5L** | Rs.4.07L | Rs.4.21L | Rs.4.47L | Rs.4.73L | Rs.4.90L | Rs.4.95L |
| **5–6L** | Rs.5.07L | Rs.5.21L | Rs.5.45L | Rs.5.70L | Rs.5.87L | Rs.5.92L |
| **6–8L** | Rs.6.14L | Rs.6.40L | Rs.6.89L | Rs.7.40L | Rs.7.72L | Rs.7.84L |
| **8–10L** | Rs.8.12L | Rs.8.39L | Rs.8.83L | Rs.9.35L | Rs.9.73L | Rs.9.86L |
| **10–12L** | Rs.10.16L | Rs.10.48L | Rs.10.90L | Rs.11.36L | Rs.11.70L | Rs.11.85L |
| **12–15L** | Rs.12.22L | Rs.12.55L | Rs.13.20L | Rs.14.00L | Rs.14.60L | Rs.14.80L |
| **15–20L** | Rs.15.29L | Rs.15.68L | Rs.16.58L | Rs.17.90L | Rs.19.05L | Rs.19.43L |
| **20–30L** | Rs.20.47L | Rs.20.92L | Rs.22.21L | Rs.24.40L | Rs.27.50L | Rs.28.51L |
| **30L+** | Rs.30.50L | Rs.31.90L | Rs.37.13L | Rs.43.50L | Rs.60.90L | Rs.74.07L |

### 2c. Fuel & Transmission Mix per Band

| Band | Count | Fuel Mix | Transmission Mix | Top Brands |
| :--- | :---: | :--- | :--- | :--- |
| **0–1L** | 76 | petrol 84%, diesel 16% | manual 100% | Tata(38), Maruti Suzuki(24), Chevrolet(5) |
| **1–2L** | 1,210 | petrol 82%, diesel 18%, cng 0%, electric 0% | manual 91%, automatic 9% | Maruti Suzuki(300), Tata(248), Hyundai(241) |
| **2–3L** | 3,296 | petrol 88%, diesel 11%, cng 0%, electric 0% | manual 83%, automatic 17% | Maruti Suzuki(966), Hyundai(765), Renault(545) |
| **3–4L** | 4,075 | petrol 85%, diesel 14%, cng 1%, electric 0% | manual 69%, automatic 31% | Maruti Suzuki(1207), Hyundai(1060), Renault(402) |
| **4–5L** | 3,558 | petrol 83%, diesel 16%, cng 1% | manual 71%, automatic 29% | Maruti Suzuki(991), Hyundai(861), Honda(347) |
| **5–6L** | 3,095 | petrol 83%, diesel 15%, cng 2% | manual 65%, automatic 35% | Maruti Suzuki(1025), Hyundai(576), Renault(377) |
| **6–8L** | 3,991 | petrol 76%, diesel 23%, cng 1%, electric 0% | manual 60%, automatic 40% | Maruti Suzuki(944), Hyundai(865), Tata(643) |
| **8–10L** | 2,334 | petrol 69%, diesel 29%, cng 1%, electric 1% | manual 54%, automatic 46% | Hyundai(570), Tata(374), Maruti Suzuki(325) |
| **10–12L** | 1,379 | petrol 71%, diesel 28%, cng 1%, electric 0%, hybrid 0% | manual 52%, automatic 48% | Kia(357), Hyundai(204), Tata(139) |
| **12–15L** | 1,254 | petrol 67%, diesel 28%, hybrid 4%, cng 0%, electric 0% | automatic 65%, manual 35% | Kia(301), Hyundai(194), Mahindra(138) |
| **15–20L** | 689 | petrol 49%, diesel 47%, hybrid 3%, electric 0% | automatic 77%, manual 23% | Hyundai(170), Mahindra(169), Kia(99) |
| **20–30L** | 140 | diesel 60%, petrol 31%, hybrid 9% | automatic 89%, manual 11% | Mahindra(47), Toyota(24), Bmw(21) |
| **30L+** | 61 | petrol 51%, diesel 44%, hybrid 3%, electric 2% | automatic 87%, manual 13% | Bmw(24), Audi(11), Mercedes-Benz(9) |

### 2d. Ownership & Certification per Band

| Band | Count | Single-Owner % | Certified % |
| :--- | :---: | :---: | :---: |
| **0–1L** | 76 | 64.5% | 56.6% |
| **1–2L** | 1,210 | 57.5% | 60.2% |
| **2–3L** | 3,296 | 60.6% | 58.4% |
| **3–4L** | 4,075 | 63.0% | 68.3% |
| **4–5L** | 3,558 | 71.2% | 71.6% |
| **5–6L** | 3,095 | 78.2% | 74.1% |
| **6–8L** | 3,991 | 82.3% | 78.5% |
| **8–10L** | 2,334 | 86.2% | 79.9% |
| **10–12L** | 1,379 | 88.2% | 78.5% |
| **12–15L** | 1,254 | 89.0% | 81.4% |
| **15–20L** | 689 | 95.2% | 85.2% |
| **20–30L** | 140 | 93.6% | 67.1% |
| **30L+** | 61 | 96.7% | 21.3% |

---

## 3. Brand-Wise Analysis

### 3a. All Brands - Volume & Pricing (sorted by listing count)

| Rank | Brand | Count | Share | Mean | Median | Min | Max | Top Model |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| 1 | **Maruti Suzuki** | 6,034 | 24.0% | Rs.4.90L | Rs.4.53L | Rs.68.0K | Rs.22.01L | Baleno |
| 2 | **Hyundai** | 5,518 | 21.9% | Rs.5.80L | Rs.4.75L | Rs.77.8K | Rs.31.00L | I20 |
| 3 | **Tata** | 2,553 | 10.1% | Rs.6.44L | Rs.6.18L | Rs.54.0K | Rs.20.06L | Nexon |
| 4 | **Renault** | 2,029 | 8.1% | Rs.4.30L | Rs.4.06L | Rs.50.0K | Rs.10.99L | Kwid |
| 5 | **Honda** | 1,627 | 6.5% | Rs.5.00L | Rs.4.50L | Rs.1.00L | Rs.16.10L | City |
| 6 | **Mahindra** | 1,327 | 5.3% | Rs.9.61L | Rs.8.51L | Rs.1.00L | Rs.24.00L | Xuv500 |
| 7 | **Ford** | 1,281 | 5.1% | Rs.4.20L | Rs.4.11L | Rs.90.0K | Rs.16.80L | Ecosport |
| 8 | **Kia** | 1,203 | 4.8% | Rs.11.10L | Rs.10.93L | Rs.5.73L | Rs.22.18L | Seltos |
| 9 | **Volkswagen** | 970 | 3.9% | Rs.6.28L | Rs.4.84L | Rs.1.30L | Rs.21.42L | Polo |
| 10 | **Skoda** | 645 | 2.6% | Rs.8.01L | Rs.7.71L | Rs.1.20L | Rs.18.70L | Rapid |
| 11 | **Nissan** | 436 | 1.7% | Rs.4.77L | Rs.4.04L | Rs.1.50L | Rs.10.01L | Magnite |
| 12 | **Toyota** | 413 | 1.6% | Rs.9.03L | Rs.7.14L | Rs.1.00L | Rs.32.50L | Glanza |
| 13 | **Mg** | 322 | 1.3% | Rs.11.85L | Rs.11.50L | Rs.7.34L | Rs.19.10L | Hector |
| 14 | **Datsun** | 190 | 0.8% | Rs.2.28L | Rs.2.21L | Rs.1.30L | Rs.4.19L | Redi-Go |
| 15 | **Jeep** | 130 | 0.5% | Rs.11.95L | Rs.11.31L | Rs.6.40L | Rs.25.50L | Compass |
| 16 | **Chevrolet** | 129 | 0.5% | Rs.2.06L | Rs.1.94L | Rs.58.0K | Rs.8.00L | Beat |
| 17 | **Bmw** | 87 | 0.3% | Rs.22.28L | Rs.20.00L | Rs.2.98L | Rs.75.99L | X1 |
| 18 | **Audi** | 83 | 0.3% | Rs.15.17L | Rs.12.50L | Rs.4.50L | Rs.39.10L | A4 |
| 19 | **Fiat** | 70 | 0.3% | Rs.2.29L | Rs.2.21L | Rs.95.0K | Rs.4.37L | Grand Punto |
| 20 | **Mercedes-Benz** | 37 | 0.1% | Rs.19.94L | Rs.9.26L | Rs.3.62L | Rs.64.00L | C Class |
| 21 | **Volvo** | 21 | 0.1% | Rs.23.04L | Rs.26.63L | Rs.8.25L | Rs.42.90L | Xc60 |
| 22 | **Citroen** | 18 | 0.1% | Rs.8.76L | Rs.9.07L | Rs.4.50L | Rs.12.50L | C3 Aircross |
| 23 | **Land Rover** | 12 | 0.0% | Rs.18.36L | Rs.20.20L | Rs.7.50L | Rs.27.50L | Discovery Sport |
| 24 | **Jaguar** | 7 | 0.0% | Rs.13.71L | Rs.13.40L | Rs.8.60L | Rs.22.00L | Xf |
| 25 | **Mini** | 6 | 0.0% | Rs.15.30L | Rs.11.94L | Rs.8.58L | Rs.25.50L | Countryman |
| 26 | **Mitsubishi** | 6 | 0.0% | Rs.10.13L | Rs.9.93L | Rs.9.00L | Rs.11.70L | Pajero Sport |

### 3b. Brand Segment Distribution (Count of listings per segment)

| Brand | Budget (0-3L) | Economy (3-6L) | Mid (6-12L) | Premium (12-20L) | Luxury (20L+) | Total |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Maruti Suzuki** | 1290 | 3223 | 1406 | 109 | 6 | 6034 |
| **Hyundai** | 1010 | 2497 | 1639 | 364 | 8 | 5518 |
| **Tata** | 410 | 803 | 1156 | 183 | 1 | 2553 |
| **Renault** | 579 | 1103 | 347 | 0 | 0 | 2029 |
| **Honda** | 225 | 1017 | 353 | 32 | 0 | 1627 |
| **Mahindra** | 68 | 232 | 673 | 307 | 47 | 1327 |
| **Ford** | 382 | 666 | 231 | 2 | 0 | 1281 |
| **Kia** | 0 | 2 | 798 | 400 | 3 | 1203 |
| **Volkswagen** | 79 | 575 | 199 | 113 | 4 | 970 |
| **Skoda** | 75 | 188 | 236 | 146 | 0 | 645 |
| **Nissan** | 74 | 237 | 125 | 0 | 0 | 436 |
| **Toyota** | 35 | 121 | 160 | 65 | 32 | 413 |
| **Mg** | 0 | 0 | 184 | 138 | 0 | 322 |
| **Datsun** | 177 | 13 | 0 | 0 | 0 | 190 |
| **Jeep** | 0 | 0 | 82 | 46 | 2 | 130 |
| **Chevrolet** | 115 | 13 | 1 | 0 | 0 | 129 |
| **Bmw** | 2 | 13 | 24 | 3 | 45 | 87 |
| **Audi** | 0 | 8 | 33 | 23 | 19 | 83 |
| **Fiat** | 60 | 10 | 0 | 0 | 0 | 70 |
| **Mercedes-Benz** | 0 | 2 | 20 | 3 | 12 | 37 |
| **Volvo** | 0 | 0 | 10 | 0 | 11 | 21 |
| **Citroen** | 0 | 4 | 13 | 1 | 0 | 18 |
| **Land Rover** | 0 | 0 | 4 | 2 | 6 | 12 |
| **Jaguar** | 0 | 0 | 1 | 5 | 1 | 7 |
| **Mini** | 0 | 0 | 3 | 1 | 2 | 6 |
| **Mitsubishi** | 0 | 0 | 6 | 0 | 0 | 6 |

### 3c. Brand Fuel & Transmission Mix

| Brand | Count | Fuel Mix | Transmission Mix |
| :--- | :---: | :--- | :--- |
| **Maruti Suzuki** | 6,034 | petrol 88%, diesel 9%, cng 2%, hybrid 1%, electric 0% | manual 66%, automatic 34% |
| **Hyundai** | 5,518 | petrol 87%, diesel 13%, cng 1%, electric 0% | manual 68%, automatic 32% |
| **Tata** | 2,553 | petrol 75%, diesel 22%, cng 2%, electric 1% | manual 61%, automatic 39% |
| **Renault** | 2,029 | petrol 83%, diesel 16%, cng 0% | manual 59%, automatic 41% |
| **Honda** | 1,627 | petrol 90%, diesel 10%, cng 0% | manual 64%, automatic 36% |
| **Mahindra** | 1,327 | diesel 60%, petrol 39%, electric 1% | manual 63%, automatic 37% |
| **Ford** | 1,281 | petrol 65%, diesel 34%, cng 0% | manual 86%, automatic 14% |
| **Kia** | 1,203 | petrol 63%, diesel 37% | manual 59%, automatic 41% |
| **Volkswagen** | 970 | petrol 79%, diesel 21% | manual 68%, automatic 32% |
| **Skoda** | 645 | petrol 72%, diesel 28%, cng 0% | automatic 52%, manual 48% |
| **Nissan** | 436 | petrol 85%, diesel 14%, cng 0% | manual 53%, automatic 47% |
| **Toyota** | 413 | petrol 70%, diesel 23%, hybrid 7% | manual 62%, automatic 38% |
| **Mg** | 322 | petrol 73%, diesel 26%, hybrid 1%, electric 1% | automatic 54%, manual 46% |
| **Datsun** | 190 | petrol 100% | manual 82%, automatic 18% |
| **Jeep** | 130 | diesel 65%, petrol 35% | manual 66%, automatic 34% |
| **Chevrolet** | 129 | diesel 50%, petrol 50% | manual 94%, automatic 6% |
| **Bmw** | 87 | diesel 71%, petrol 29% | automatic 100% |
| **Audi** | 83 | diesel 76%, petrol 24% | automatic 100% |
| **Fiat** | 70 | diesel 56%, petrol 44% | manual 100% |
| **Mercedes-Benz** | 37 | diesel 81%, petrol 19% | automatic 100% |
| **Volvo** | 21 | diesel 71%, petrol 29% | automatic 100% |
| **Citroen** | 18 | petrol 100% | manual 94%, automatic 6% |
| **Land Rover** | 12 | diesel 100% | automatic 100% |
| **Jaguar** | 7 | diesel 57%, petrol 43% | automatic 100% |
| **Mini** | 6 | diesel 100% | automatic 100% |
| **Mitsubishi** | 6 | diesel 100% | automatic 100% |

---

## 4. Top-15 Brands x Price Band Heat-Map (Listing Counts)

| Brand | 0–1L | 1–2L | 2–3L | 3–4L | 4–5L | 5–6L | 6–8L | 8–10L | 10–12L | 12–15L | 15–20L | 20–30L | 30L+ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Maruti Suzuki** | 24 | 300 | 966 | 1207 | 991 | 1025 | 944 | 325 | 137 | 82 | 27 | 6 | 0 |
| **Hyundai** | 4 | 241 | 765 | 1060 | 861 | 576 | 865 | 570 | 204 | 194 | 170 | 7 | 1 |
| **Tata** | 38 | 248 | 124 | 246 | 269 | 288 | 643 | 374 | 139 | 114 | 69 | 1 | 0 |
| **Renault** | 1 | 33 | 545 | 402 | 324 | 377 | 317 | 29 | 1 | 0 | 0 | 0 | 0 |
| **Honda** | 0 | 43 | 182 | 402 | 347 | 268 | 229 | 77 | 47 | 24 | 8 | 0 | 0 |
| **Mahindra** | 0 | 17 | 51 | 81 | 68 | 83 | 273 | 266 | 134 | 138 | 169 | 47 | 0 |
| **Ford** | 3 | 124 | 255 | 217 | 281 | 168 | 210 | 20 | 1 | 0 | 2 | 0 | 0 |
| **Kia** | 0 | 0 | 0 | 0 | 0 | 2 | 133 | 308 | 357 | 301 | 99 | 3 | 0 |
| **Volkswagen** | 0 | 21 | 58 | 182 | 267 | 126 | 94 | 34 | 71 | 92 | 21 | 4 | 0 |
| **Skoda** | 0 | 25 | 50 | 64 | 47 | 77 | 67 | 76 | 93 | 134 | 12 | 0 | 0 |
| **Nissan** | 0 | 10 | 64 | 141 | 44 | 52 | 88 | 36 | 1 | 0 | 0 | 0 | 0 |
| **Toyota** | 0 | 5 | 30 | 41 | 36 | 44 | 82 | 54 | 24 | 39 | 26 | 24 | 8 |
| **Mg** | 0 | 0 | 0 | 0 | 0 | 0 | 11 | 67 | 106 | 99 | 39 | 0 | 0 |
| **Datsun** | 0 | 54 | 123 | 11 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| **Jeep** | 0 | 0 | 0 | 0 | 0 | 0 | 19 | 26 | 37 | 21 | 25 | 2 | 0 |

---

## 5. Key Findings & Insights

1. **Most popular price band:** **3–4L** with 4,075 listings (16.2% of all cars).
2. **Most affordable band (by mean):** **0–1L** — mean price Rs.82.5K.
3. **Highest avg price band:** **30L+** — mean Rs.41.05L.
4. **Largest brand by volume:** **Maruti Suzuki** (6,034 cars, 24.0% share).
5. **Highest median price brand (>=5 cars):** **Volvo** — median Rs.26.63L.
6. **Lowest median price brand (>=5 cars):** **Chevrolet** — median Rs.1.94L.

### Fuel Type Distribution (All Listings)

| Fuel Type | Count | Share |
| :--- | :---: | :---: |
| Petrol | 19,813 | 78.8% |
| Diesel | 4,995 | 19.9% |
| Cng | 211 | 0.8% |
| Hybrid | 91 | 0.4% |
| Electric | 48 | 0.2% |

### Transmission Distribution (All Listings)

| Transmission | Count | Share |
| :--- | :---: | :---: |
| Manual | 16,250 | 64.6% |
| Automatic | 8,908 | 35.4% |

### Owner Count Distribution

| Owner Count | Listings | Share |
| :---: | :---: | :---: |
| 1 | 18,737 | 74.5% |
| 2 | 4,953 | 19.7% |
| 3 | 1,262 | 5.0% |
| 4 | 161 | 0.6% |
| 5 | 38 | 0.2% |
| 6 | 7 | 0.0% |

### Vehicle Age Distribution

| Age Group | Listings | Share | Mean Price | Median Price |
| :--- | :---: | :---: | :---: | :---: |
| 0-3 yrs | 3,239 | 12.9% | Rs.10.86L | Rs.9.80L |
| 4-5 yrs | 4,490 | 17.8% | Rs.8.59L | Rs.7.70L |
| 6-8 yrs | 5,692 | 22.6% | Rs.6.33L | Rs.5.71L |
| 9-10 yrs | 4,606 | 18.3% | Rs.4.63L | Rs.4.40L |
| 11-15 yrs | 6,680 | 26.6% | Rs.3.35L | Rs.3.10L |
| 16+ yrs | 415 | 1.6% | Rs.1.98L | Rs.1.59L |

---

## 6. Appendix - Top 20 Models by Listing Volume

| Rank | Brand | Model | Count | Mean | Median | Min | Max |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| 1 | Hyundai | I20 | 1,022 | Rs.5.16L | Rs.5.05L | Rs.1.10L | Rs.11.50L |
| 2 | Hyundai | Grand I10 | 1,016 | Rs.3.85L | Rs.3.77L | Rs.2.30L | Rs.6.00L |
| 3 | Renault | Kwid | 970 | Rs.3.18L | Rs.2.98L | Rs.1.78L | Rs.6.00L |
| 4 | Hyundai | Creta | 818 | Rs.10.24L | Rs.8.97L | Rs.4.50L | Rs.19.40L |
| 5 | Maruti Suzuki | Baleno | 797 | Rs.5.68L | Rs.5.46L | Rs.1.00L | Rs.10.58L |
| 6 | Tata | Nexon | 765 | Rs.8.09L | Rs.7.81L | Rs.4.55L | Rs.14.81L |
| 7 | Maruti Suzuki | Swift | 728 | Rs.5.02L | Rs.5.21L | Rs.80.0K | Rs.9.02L |
| 8 | Honda | City | 650 | Rs.5.76L | Rs.5.21L | Rs.1.00L | Rs.15.25L |
| 9 | Kia | Seltos | 625 | Rs.11.77L | Rs.11.59L | Rs.6.67L | Rs.19.20L |
| 10 | Ford | Ecosport | 595 | Rs.5.46L | Rs.5.33L | Rs.2.81L | Rs.10.31L |
| 11 | Maruti Suzuki | Celerio | 592 | Rs.3.66L | Rs.3.50L | Rs.1.50L | Rs.8.12L |
| 12 | Maruti Suzuki | Alto K10 | 576 | Rs.2.69L | Rs.2.61L | Rs.80.0K | Rs.5.60L |
| 13 | Tata | Tiago | 505 | Rs.4.31L | Rs.4.18L | Rs.2.30L | Rs.8.98L |
| 14 | Maruti Suzuki | Dzire | 484 | Rs.4.69L | Rs.4.80L | Rs.1.10L | Rs.10.29L |
| 15 | Hyundai | Verna | 459 | Rs.6.87L | Rs.5.33L | Rs.1.00L | Rs.19.43L |
| 16 | Volkswagen | Polo | 447 | Rs.4.50L | Rs.4.38L | Rs.1.60L | Rs.9.10L |
| 17 | Maruti Suzuki | Wagon R | 442 | Rs.3.15L | Rs.3.02L | Rs.1.20L | Rs.7.50L |
| 18 | Hyundai | I10 | 418 | Rs.2.45L | Rs.2.50L | Rs.83.0K | Rs.3.90L |
| 19 | Renault | Duster | 405 | Rs.5.09L | Rs.5.00L | Rs.1.56L | Rs.10.99L |
| 20 | Maruti Suzuki | Ciaz | 396 | Rs.5.89L | Rs.5.67L | Rs.3.16L | Rs.9.93L |

---

*Report generated automatically by PriceRef Analysis Suite.*