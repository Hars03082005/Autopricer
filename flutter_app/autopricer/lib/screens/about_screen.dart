import 'package:flutter/material.dart';
import 'package:animate_do/animate_do.dart';
import '../app_theme.dart';
import '../widgets/common.dart';

class AboutScreen extends StatelessWidget {
  const AboutScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: CustomScrollView(
        physics: const BouncingScrollPhysics(),
        slivers: [
          _buildAppBar(),
          SliverToBoxAdapter(
            child: Padding(
              padding: const EdgeInsets.fromLTRB(20, 0, 20, 40),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  _buildHero(),
                  const SizedBox(height: 28),
                  _buildSectionLabel('TECHNOLOGY'),
                  const SizedBox(height: 12),
                  _buildTechCards(),
                  const SizedBox(height: 28),
                  _buildSectionLabel('PERFORMANCE'),
                  const SizedBox(height: 12),
                  _buildMetricsGrid(),
                  const SizedBox(height: 28),
                  _buildSectionLabel('STACK'),
                  const SizedBox(height: 12),
                  _buildStackBadges(),
                  const SizedBox(height: 32),
                  _buildTeamCard(),
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
      floating: true, snap: true,
      backgroundColor: AppColors.bg,
      surfaceTintColor: Colors.transparent,
      titleSpacing: 20,
      title: const Text('About AutoPricerAI',
          style: TextStyle(fontSize: 17, fontWeight: FontWeight.w800, color: AppColors.text)),
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

  // ─── Hero ─────────────────────────────────────────────────────────────────
  Widget _buildHero() {
    return FadeInDown(
      duration: const Duration(milliseconds: 500),
      child: Container(
        width: double.infinity,
        padding: const EdgeInsets.all(28),
        decoration: BoxDecoration(
          gradient: const LinearGradient(
            colors: [Color(0xFF0B1535), Color(0xFF091825)],
            begin: Alignment.topLeft, end: Alignment.bottomRight,
          ),
          borderRadius: AppRadius.xl,
          border: Border.all(color: AppColors.blue.withOpacity(0.25), width: 1),
        ),
        child: Column(
          children: [
            // Logo orb
            Container(
              width: 80, height: 80,
              decoration: BoxDecoration(
                gradient: AppGradients.blueGlow,
                borderRadius: AppRadius.xl,
                boxShadow: [BoxShadow(color: AppColors.blue.withOpacity(0.4), blurRadius: 24, spreadRadius: 2)],
              ),
              child: const Center(
                child: Text('A', style: TextStyle(fontSize: 38, fontWeight: FontWeight.w900, color: Colors.white)),
              ),
            ),
            const SizedBox(height: 16),
            const Text(
              'AutoPricerAI',
              style: TextStyle(fontSize: 26, fontWeight: FontWeight.w900, color: AppColors.text, letterSpacing: -0.8),
            ),
            const SizedBox(height: 4),
            const Text(
              'Pre-owned Vehicle Intelligence',
              style: TextStyle(fontSize: 14, color: AppColors.textSub),
            ),
            const SizedBox(height: 10),
            PillChip(label: 'ForgePoint AI · 2026', color: AppColors.blue),
          ],
        ),
      ),
    );
  }

  // ─── Tech cards ───────────────────────────────────────────────────────────
  Widget _buildTechCards() {
    final items = [
      _TechItem(Icons.psychology_rounded, 'Ensemble Model', 'A meta-learner stacks XGBoost, LightGBM, and CatBoost predictions for superior accuracy.', AppColors.blue, AppGradients.blueGlow),
      _TechItem(Icons.dataset_rounded, 'Training Data', 'Trained on 400,000+ vehicle records from CarDekho India — covering 18+ brands and all fuel types.', AppColors.teal, AppGradients.greenGlow),
      _TechItem(Icons.api_rounded, 'FastAPI Backend', 'High-performance async REST API built with Python FastAPI, deployed with sub-100ms response times.', AppColors.amber, AppGradients.amberGlow),
    ];

    return Column(
      children: items.asMap().entries.map((e) {
        return FadeInLeft(
          delay: Duration(milliseconds: 100 * e.key),
          duration: const Duration(milliseconds: 400),
          child: Container(
            margin: const EdgeInsets.only(bottom: 10),
            padding: const EdgeInsets.all(16),
            decoration: BoxDecoration(
              color: AppColors.surface,
              borderRadius: AppRadius.lg,
              border: Border.all(color: AppColors.border, width: 0.5),
            ),
            child: Row(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Container(
                  width: 44, height: 44,
                  decoration: BoxDecoration(gradient: e.value.gradient, borderRadius: AppRadius.md,
                      boxShadow: [BoxShadow(color: e.value.color.withOpacity(0.3), blurRadius: 10)]),
                  child: Icon(e.value.icon, color: Colors.white, size: 20),
                ),
                const SizedBox(width: 14),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(e.value.title, style: AppTextStyles.cardTitle),
                      const SizedBox(height: 5),
                      Text(e.value.body, style: const TextStyle(fontSize: 12, color: AppColors.textSub, height: 1.5)),
                    ],
                  ),
                ),
              ],
            ),
          ),
        );
      }).toList(),
    );
  }

  // ─── Metrics grid ─────────────────────────────────────────────────────────
  Widget _buildMetricsGrid() {
    final metrics = [
      _Metric('R² Score', '0.96', 'Coefficient of determination', AppColors.teal),
      _Metric('MAE', '~₹70K', 'Mean Absolute Error', AppColors.blue),
      _Metric('RMSE', '~₹1.1L', 'Root Mean Sq. Error', AppColors.amber),
      _Metric('Latency', '<2s', 'End-to-end predict time', AppColors.pink),
    ];

    return GridView.count(
      crossAxisCount: 2,
      crossAxisSpacing: 10,
      mainAxisSpacing: 10,
      shrinkWrap: true,
      childAspectRatio: 1.8,
      physics: const NeverScrollableScrollPhysics(),
      children: metrics.map((m) {
        return Container(
          padding: const EdgeInsets.all(14),
          decoration: BoxDecoration(
            color: AppColors.surface,
            borderRadius: AppRadius.lg,
            border: Border.all(color: m.color.withOpacity(0.3), width: 0.5),
          ),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Text(m.label, style: const TextStyle(fontSize: 10, color: AppColors.muted, fontWeight: FontWeight.w500)),
              Text(m.value, style: TextStyle(fontSize: 22, fontWeight: FontWeight.w900, color: m.color, height: 1.0)),
              Text(m.sub, style: const TextStyle(fontSize: 9, color: AppColors.muted)),
            ],
          ),
        );
      }).toList(),
    );
  }

  // ─── Stack badges ─────────────────────────────────────────────────────────
  Widget _buildStackBadges() {
    final stacks = [
      ('Flutter', AppColors.blue),
      ('FastAPI', AppColors.teal),
      ('Python 3.11', AppColors.amber),
      ('XGBoost', AppColors.purple),
      ('LightGBM', AppColors.green),
      ('CatBoost', AppColors.pink),
      ('Pandas', AppColors.blue),
      ('Scikit-Learn', AppColors.amber),
    ];

    return Wrap(
      spacing: 8,
      runSpacing: 8,
      children: stacks.map((s) => PillChip(label: s.$1, color: s.$2)).toList(),
    );
  }

  // ─── Team card ────────────────────────────────────────────────────────────
  Widget _buildTeamCard() {
    return FadeIn(
      delay: const Duration(milliseconds: 400),
      child: Container(
        padding: const EdgeInsets.all(20),
        decoration: BoxDecoration(
          gradient: const LinearGradient(
            colors: [Color(0xFF0B1535), Color(0xFF091825)],
            begin: Alignment.topLeft, end: Alignment.bottomRight,
          ),
          borderRadius: AppRadius.xl,
          border: Border.all(color: AppColors.blue.withOpacity(0.2), width: 1),
        ),
        child: Column(
          children: [
            Row(
              children: [
                Container(
                  width: 44, height: 44,
                  decoration: BoxDecoration(
                    gradient: AppGradients.purpleGlow,
                    borderRadius: AppRadius.md,
                    boxShadow: [BoxShadow(color: AppColors.purple.withOpacity(0.3), blurRadius: 10)],
                  ),
                  child: const Icon(Icons.rocket_launch_rounded, color: Colors.white, size: 20),
                ),
                const SizedBox(width: 12),
                const Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text('ForgePoint AI', style: TextStyle(fontSize: 15, fontWeight: FontWeight.w800, color: AppColors.text)),
                    Text('Research & Products', style: TextStyle(fontSize: 11, color: AppColors.muted)),
                  ],
                ),
              ],
            ),
            const SizedBox(height: 14),
            const Text(
              'AutoPricerAI is built to demonstrate how machine learning can democratise vehicle valuation for buyers, dealers, and fleet operators across India.',
              style: TextStyle(fontSize: 13, color: AppColors.textSub, height: 1.65),
            ),
            const SizedBox(height: 14),
            Row(
              children: [
                const Icon(Icons.favorite_rounded, size: 13, color: AppColors.pink),
                const SizedBox(width: 5),
                const Text('Built with passion for the Indian automotive market',
                    style: TextStyle(fontSize: 11, color: AppColors.muted)),
              ],
            ),
          ],
        ),
      ),
    );
  }
}

// ─── Data models ──────────────────────────────────────────────────────────────
class _TechItem {
  final IconData icon; final String title; final String body;
  final Color color; final Gradient gradient;
  const _TechItem(this.icon, this.title, this.body, this.color, this.gradient);
}

class _Metric {
  final String label, value, sub; final Color color;
  const _Metric(this.label, this.value, this.sub, this.color);
}