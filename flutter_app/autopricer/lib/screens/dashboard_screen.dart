import 'package:flutter/material.dart';
import 'package:fl_chart/fl_chart.dart';
import '../app_theme.dart';
import '../widgets/common.dart';

class DashboardScreen extends StatelessWidget {
  const DashboardScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: CustomScrollView(
        physics: const BouncingScrollPhysics(),
        slivers: [
          _buildAppBar(),
          SliverToBoxAdapter(
            child: Padding(
              padding: const EdgeInsets.fromLTRB(16, 8, 16, 40),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  _buildKpiRow(),
                  const SizedBox(height: 20),
                  _buildSectionLabel('FUEL TYPE ANALYSIS'),
                  const SizedBox(height: 12),
                  _buildFuelChart(),
                  const SizedBox(height: 20),
                  _buildSectionLabel('PRICE DEPRECIATION'),
                  const SizedBox(height: 12),
                  _buildDepreciationChart(),
                  const SizedBox(height: 20),
                  _buildSectionLabel('BRAND MARKET SHARE'),
                  const SizedBox(height: 12),
                  _buildBrandChart(),
                  const SizedBox(height: 20),
                  _buildSectionLabel('TRANSMISSION SPLIT'),
                  const SizedBox(height: 12),
                  _buildTransmissionChart(),
                  const SizedBox(height: 20),
                  _buildSectionLabel('PRICE DISTRIBUTION'),
                  const SizedBox(height: 12),
                  _buildPriceRangeChart(),
                  const SizedBox(height: 12),
                  _buildDataSourceBadge(),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }

  // ─── App bar ──────────────────────────────────────────────────────────────
  Widget _buildAppBar() {
    return SliverAppBar(
      floating: true,
      snap: true,
      backgroundColor: AppColors.bg,
      surfaceTintColor: Colors.transparent,
      titleSpacing: 16,
      title: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: const [
          Text('Market Intelligence', style: TextStyle(fontSize: 17, fontWeight: FontWeight.w800, color: AppColors.text)),
          Text('Indian used car market · Live dataset', style: TextStyle(fontSize: 11, color: AppColors.muted, fontWeight: FontWeight.w400)),
        ],
      ),
      bottom: PreferredSize(
        preferredSize: const Size.fromHeight(0.5),
        child: Container(height: 0.5, color: AppColors.border),
      ),
    );
  }

  // ─── Section label ────────────────────────────────────────────────────────
  Widget _buildSectionLabel(String label) {
    return Row(
      children: [
        Container(width: 3, height: 12, decoration: BoxDecoration(gradient: AppGradients.blueGlow, borderRadius: AppRadius.pill)),
        const SizedBox(width: 8),
        Text(label, style: const TextStyle(fontSize: 10, fontWeight: FontWeight.w700, color: AppColors.muted, letterSpacing: 1.2)),
      ],
    );
  }

  // ─── KPI Row ─────────────────────────────────────────────────────────────
  Widget _buildKpiRow() {
    return Row(
      children: const [
        KpiCard(value: '₹5.2L',   label: 'Median Price',  color: AppColors.blue),
        SizedBox(width: 8),
        KpiCard(value: '7.4 yr',  label: 'Avg Age',        color: AppColors.teal),
        SizedBox(width: 8),
        KpiCard(value: '42K km',  label: 'Avg Mileage',    color: AppColors.amber),
      ],
    );
  }

  // ─── Chart wrapper ────────────────────────────────────────────────────────
  Widget _chartCard({required String title, required String subtitle, required double height, required Widget child}) {
    return Container(
      padding: const EdgeInsets.all(18),
      decoration: BoxDecoration(
        color: AppColors.surface,
        borderRadius: AppRadius.lg,
        border: Border.all(color: AppColors.border, width: 0.5),
        boxShadow: const [BoxShadow(color: Color(0x0A4F8EF7), blurRadius: 16, offset: Offset(0, 4))],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(title, style: AppTextStyles.cardTitle),
          const SizedBox(height: 2),
          Text(subtitle, style: AppTextStyles.cardSubtitle),
          const SizedBox(height: 18),
          SizedBox(height: height, child: child),
        ],
      ),
    );
  }

  // ─── Fuel bar chart ───────────────────────────────────────────────────────
  Widget _buildFuelChart() {
    final data = [
      _FuelEntry('Petrol',   6.2,  AppColors.blue),
      _FuelEntry('Diesel',   8.1,  AppColors.teal),
      _FuelEntry('CNG',      4.1,  AppColors.amber),
      _FuelEntry('Electric', 12.5, AppColors.pink),
    ];
    return _chartCard(
      title: 'Avg Price by Fuel Type',
      subtitle: 'Lakhs (₹) · CarDekho dataset',
      height: 200,
      child: BarChart(
        BarChartData(
          backgroundColor: Colors.transparent,
          borderData: FlBorderData(show: false),
          gridData: FlGridData(
            show: true, drawVerticalLine: false, horizontalInterval: 3,
            getDrawingHorizontalLine: (_) => const FlLine(color: AppColors.border, strokeWidth: 0.5),
          ),
          titlesData: FlTitlesData(
            leftTitles:   const AxisTitles(sideTitles: SideTitles(showTitles: false)),
            rightTitles:  const AxisTitles(sideTitles: SideTitles(showTitles: false)),
            topTitles:    const AxisTitles(sideTitles: SideTitles(showTitles: false)),
            bottomTitles: AxisTitles(
              sideTitles: SideTitles(
                showTitles: true,
                getTitlesWidget: (v, _) {
                  final i = v.toInt();
                  if (i < 0 || i >= data.length) return const SizedBox.shrink();
                  return Padding(
                    padding: const EdgeInsets.only(top: 6),
                    child: Text(data[i].name, style: const TextStyle(fontSize: 10, color: AppColors.muted)),
                  );
                },
              ),
            ),
          ),
          barGroups: data.asMap().entries.map((e) => BarChartGroupData(
            x: e.key,
            barRods: [
              BarChartRodData(
                toY: e.value.value,
                width: 32,
                gradient: LinearGradient(
                  colors: [e.value.color, e.value.color.withOpacity(0.5)],
                  begin: Alignment.topCenter, end: Alignment.bottomCenter,
                ),
                borderRadius: const BorderRadius.vertical(top: Radius.circular(6)),
                backDrawRodData: BackgroundBarChartRodData(
                  show: true, toY: 14, color: AppColors.surface2,
                ),
              ),
            ],
          )).toList(),
          barTouchData: BarTouchData(
            touchTooltipData: BarTouchTooltipData(
              tooltipBgColor: AppColors.surface3,
              tooltipRoundedRadius: 8,
              getTooltipItem: (group, _, rod, __) {
                final entry = data[group.x];
                return BarTooltipItem(
                  '${entry.name}\n₹${rod.toY.toStringAsFixed(1)}L',
                  TextStyle(fontSize: 11, fontWeight: FontWeight.w700, color: entry.color),
                );
              },
            ),
          ),
        ),
      ),
    );
  }

  // ─── Depreciation line chart ──────────────────────────────────────────────
  Widget _buildDepreciationChart() {
    final spots = [
      const FlSpot(0,  10.0), const FlSpot(1, 8.5),
      const FlSpot(2,  7.2),  const FlSpot(3, 6.3),
      const FlSpot(4,  5.6),  const FlSpot(5, 5.0),
      const FlSpot(6,  4.5),  const FlSpot(8, 3.8),
      const FlSpot(10, 3.2),  const FlSpot(12, 2.8),
    ];
    return _chartCard(
      title: 'Price Depreciation Curve',
      subtitle: 'Median selling price (₹L) vs vehicle age (years)',
      height: 210,
      child: LineChart(
        LineChartData(
          backgroundColor: Colors.transparent,
          borderData: FlBorderData(show: false),
          gridData: FlGridData(
            show: true, drawVerticalLine: false, horizontalInterval: 2,
            getDrawingHorizontalLine: (_) => const FlLine(color: AppColors.border, strokeWidth: 0.5),
          ),
          titlesData: FlTitlesData(
            leftTitles: const AxisTitles(sideTitles: SideTitles(showTitles: false)),
            topTitles:  const AxisTitles(sideTitles: SideTitles(showTitles: false)),
            rightTitles: AxisTitles(
              sideTitles: SideTitles(
                showTitles: true, reservedSize: 36,
                getTitlesWidget: (v, _) => Text('₹${v.toInt()}L', style: const TextStyle(fontSize: 9, color: AppColors.muted)),
              ),
            ),
            bottomTitles: AxisTitles(
              sideTitles: SideTitles(
                showTitles: true, reservedSize: 22,
                getTitlesWidget: (v, _) {
                  final l = v.toInt();
                  if (l % 2 != 0) return const SizedBox.shrink();
                  return Text('${l}y', style: const TextStyle(fontSize: 9, color: AppColors.muted));
                },
              ),
            ),
          ),
          lineBarsData: [
            LineChartBarData(
              spots: spots,
              isCurved: true, curveSmoothness: 0.4,
              gradient: const LinearGradient(colors: [AppColors.teal, AppColors.blue]),
              barWidth: 2.5,
              dotData: FlDotData(
                show: true,
                getDotPainter: (s, _, __, ___) => FlDotCirclePainter(
                  radius: 3.5, color: AppColors.teal,
                  strokeWidth: 1.5, strokeColor: AppColors.bg,
                ),
              ),
              belowBarData: BarAreaData(
                show: true,
                gradient: LinearGradient(
                  colors: [AppColors.teal.withOpacity(0.15), Colors.transparent],
                  begin: Alignment.topCenter, end: Alignment.bottomCenter,
                ),
              ),
            ),
          ],
          lineTouchData: LineTouchData(
            touchTooltipData: LineTouchTooltipData(
              tooltipBgColor: AppColors.surface3,
              tooltipRoundedRadius: 8,
              getTooltipItems: (spots) => spots.map((s) => LineTooltipItem(
                'Age ${s.x.toInt()}yr\n₹${s.y.toStringAsFixed(1)}L',
                const TextStyle(fontSize: 11, fontWeight: FontWeight.w700, color: AppColors.teal),
              )).toList(),
            ),
          ),
        ),
      ),
    );
  }

  // ─── Brand volume chart ───────────────────────────────────────────────────
  Widget _buildBrandChart() {
    final brands = [
      _BrandEntry('Maruti',   32, AppColors.blue),
      _BrandEntry('Hyundai',  21, AppColors.teal),
      _BrandEntry('Honda',    11, AppColors.amber),
      _BrandEntry('Tata',      9, AppColors.pink),
      _BrandEntry('Mahindra',  7, AppColors.purple),
      _BrandEntry('Toyota',    6, AppColors.green),
    ];
    final maxPct = brands.map((b) => b.pct).reduce((a, b) => a > b ? a : b);

    return _chartCard(
      title: 'Top Brands by Volume',
      subtitle: '% share of used car listings in dataset',
      height: brands.length * 42.0,
      child: Column(
        children: brands.map((b) {
          return Padding(
            padding: const EdgeInsets.only(bottom: 10),
            child: Row(
              children: [
                SizedBox(
                  width: 56,
                  child: Text(b.name, style: const TextStyle(fontSize: 11, color: AppColors.muted)),
                ),
                const SizedBox(width: 10),
                Expanded(
                  child: ClipRRect(
                    borderRadius: AppRadius.pill,
                    child: Stack(
                      children: [
                        Container(height: 8, color: AppColors.surface2),
                        FractionallySizedBox(
                          widthFactor: b.pct / maxPct,
                          child: Container(
                            height: 8,
                            decoration: BoxDecoration(
                              gradient: LinearGradient(
                                colors: [b.color, b.color.withOpacity(0.5)],
                              ),
                              borderRadius: AppRadius.pill,
                            ),
                          ),
                        ),
                      ],
                    ),
                  ),
                ),
                const SizedBox(width: 10),
                SizedBox(
                  width: 32,
                  child: Text(
                    '${b.pct}%',
                    textAlign: TextAlign.right,
                    style: TextStyle(fontSize: 11, fontWeight: FontWeight.w700, color: b.color),
                  ),
                ),
              ],
            ),
          );
        }).toList(),
      ),
    );
  }

  // ─── Transmission pie ─────────────────────────────────────────────────────
  Widget _buildTransmissionChart() {
    return _chartCard(
      title: 'Manual vs Automatic',
      subtitle: 'Transmission split across all listings',
      height: 190,
      child: Row(
        children: [
          Expanded(
            flex: 3,
            child: PieChart(
              PieChartData(
                sections: [
                  PieChartSectionData(
                    value: 65, title: '65%', color: AppColors.blue, radius: 75,
                    titleStyle: const TextStyle(fontSize: 14, fontWeight: FontWeight.w800, color: Colors.white),
                  ),
                  PieChartSectionData(
                    value: 35, title: '35%', color: AppColors.teal, radius: 75,
                    titleStyle: const TextStyle(fontSize: 14, fontWeight: FontWeight.w800, color: Colors.black),
                  ),
                ],
                sectionsSpace: 3,
                centerSpaceRadius: 30,
              ),
            ),
          ),
          const SizedBox(width: 20),
          Expanded(
            flex: 2,
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                _legendItem(AppColors.blue, 'Manual',    '65%', 'Dominant segment'),
                const SizedBox(height: 16),
                _legendItem(AppColors.teal, 'Automatic', '35%', 'Growing demand'),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _legendItem(Color color, String label, String value, String sub) {
    return Row(
      children: [
        Container(
          width: 10, height: 10,
          decoration: BoxDecoration(color: color, borderRadius: AppRadius.sm),
        ),
        const SizedBox(width: 8),
        Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(label, style: const TextStyle(fontSize: 12, fontWeight: FontWeight.w700, color: AppColors.text)),
            Text(value, style: TextStyle(fontSize: 18, fontWeight: FontWeight.w900, color: color, height: 1.1)),
            Text(sub, style: const TextStyle(fontSize: 9, color: AppColors.muted)),
          ],
        ),
      ],
    );
  }

  // ─── Price range distribution ─────────────────────────────────────────────
  Widget _buildPriceRangeChart() {
    final buckets = [
      _PriceBucket('< 2L',    18, AppColors.green),
      _PriceBucket('2–5L',    35, AppColors.blue),
      _PriceBucket('5–10L',   27, AppColors.teal),
      _PriceBucket('10–20L',  14, AppColors.amber),
      _PriceBucket('> 20L',    6, AppColors.pink),
    ];
    final maxPct = buckets.map((b) => b.pct).reduce((a, b) => a > b ? a : b);

    return _chartCard(
      title: 'Listing Price Buckets',
      subtitle: '% of listings by price bracket',
      height: buckets.length * 42.0,
      child: Column(
        children: buckets.map((b) {
          return Padding(
            padding: const EdgeInsets.only(bottom: 10),
            child: Row(
              children: [
                SizedBox(
                  width: 42,
                  child: Text(b.label, style: const TextStyle(fontSize: 10, color: AppColors.muted)),
                ),
                const SizedBox(width: 10),
                Expanded(
                  child: ClipRRect(
                    borderRadius: AppRadius.pill,
                    child: Stack(
                      children: [
                        Container(height: 8, color: AppColors.surface2),
                        FractionallySizedBox(
                          widthFactor: b.pct / maxPct,
                          child: Container(
                            height: 8,
                            decoration: BoxDecoration(
                              gradient: LinearGradient(colors: [b.color, b.color.withOpacity(0.5)]),
                              borderRadius: AppRadius.pill,
                            ),
                          ),
                        ),
                      ],
                    ),
                  ),
                ),
                const SizedBox(width: 10),
                SizedBox(
                  width: 32,
                  child: Text('${b.pct}%',
                      textAlign: TextAlign.right,
                      style: TextStyle(fontSize: 11, fontWeight: FontWeight.w700, color: b.color)),
                ),
              ],
            ),
          );
        }).toList(),
      ),
    );
  }

  Widget _buildDataSourceBadge() {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
      decoration: BoxDecoration(
        color: AppColors.surface,
        borderRadius: AppRadius.md,
        border: Border.all(color: AppColors.border, width: 0.5),
      ),
      child: Row(
        children: const [
          Icon(Icons.dataset_rounded, size: 14, color: AppColors.muted),
          SizedBox(width: 8),
          Expanded(
            child: Text(
              'Data sourced from CarDekho India dataset · 400,000+ records · Updated 2026',
              style: TextStyle(fontSize: 10, color: AppColors.muted),
            ),
          ),
        ],
      ),
    );
  }
}

// ─── Data models ──────────────────────────────────────────────────────────────
class _FuelEntry {
  final String name; final double value; final Color color;
  const _FuelEntry(this.name, this.value, this.color);
}

class _BrandEntry {
  final String name; final int pct; final Color color;
  const _BrandEntry(this.name, this.pct, this.color);
}

class _PriceBucket {
  final String label; final int pct; final Color color;
  const _PriceBucket(this.label, this.pct, this.color);
}