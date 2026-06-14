import 'package:flutter/material.dart';
import 'package:animate_do/animate_do.dart';
import '../app_theme.dart';
import '../widgets/common.dart';
import '../services/api_service.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  String _apiStatus = 'Checking';
  Color _statusColor = AppColors.amber;
  bool _loading = true;

  @override
  void initState() {
    super.initState();
    _checkApi();
  }

  Future<void> _checkApi() async {
    try {
      await ApiService.getHealth();
      if (mounted) {
        setState(() {
          _apiStatus = 'Online';
          _statusColor = AppColors.teal;
          _loading = false;
        });
      }
    } catch (_) {
      if (mounted) {
        setState(() {
          _apiStatus = 'Offline';
          _statusColor = AppColors.red;
          _loading = false;
        });
      }
    }
  }

  // ─── Build ────────────────────────────────────────────────────────────────

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: CustomScrollView(
        physics: const BouncingScrollPhysics(),
        slivers: [
          _buildAppBar(),
          SliverToBoxAdapter(
            child: Padding(
              padding: const EdgeInsets.fromLTRB(20, 0, 20, 32),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  _buildHeroCard(),
                  const SizedBox(height: 28),
                  _buildStatsRow(),
                  const SizedBox(height: 32),
                  _buildSectionLabel('CAPABILITIES'),
                  const SizedBox(height: 14),
                  _buildFeatures(),
                  const SizedBox(height: 32),
                  _buildSectionLabel('HOW IT WORKS'),
                  const SizedBox(height: 14),
                  _buildHowItWorks(),
                  const SizedBox(height: 32),
                  _buildFooterBadge(),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }

  // ─── App Bar ──────────────────────────────────────────────────────────────

  Widget _buildAppBar() {
    return SliverAppBar(
      floating: true,
      snap: true,
      expandedHeight: 0,
      backgroundColor: AppColors.bg,
      surfaceTintColor: Colors.transparent,
      titleSpacing: 20,
      title: Row(
        children: [
          // Logo
          Container(
            width: 36,
            height: 36,
            decoration: BoxDecoration(
              gradient: AppGradients.blueGlow,
              borderRadius: AppRadius.md,
              boxShadow: [BoxShadow(color: AppColors.blue.withOpacity(0.35), blurRadius: 10)],
            ),
            child: const Center(
              child: Text('A', style: TextStyle(fontSize: 18, fontWeight: FontWeight.w900, color: Colors.white)),
            ),
          ),
          const SizedBox(width: 10),
          Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              const Text(
                'AutoPricerAI',
                style: TextStyle(fontSize: 15, fontWeight: FontWeight.w800, color: AppColors.text, letterSpacing: -0.3),
              ),
              Text(
                'ForgePoint AI',
                style: TextStyle(fontSize: 10, fontWeight: FontWeight.w500, color: AppColors.muted),
              ),
            ],
          ),
          const Spacer(),
          // Status chip
          _buildStatusChip(),
        ],
      ),
      bottom: PreferredSize(
        preferredSize: const Size.fromHeight(0.5),
        child: Container(height: 0.5, color: AppColors.border),
      ),
    );
  }

  Widget _buildStatusChip() {
    return AnimatedContainer(
      duration: const Duration(milliseconds: 400),
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 5),
      decoration: BoxDecoration(
        color: _statusColor.withOpacity(0.1),
        borderRadius: AppRadius.pill,
        border: Border.all(color: _statusColor.withOpacity(0.4), width: 1),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          if (_loading)
            SizedBox(
              width: 8,
              height: 8,
              child: CircularProgressIndicator(
                strokeWidth: 1.5,
                valueColor: AlwaysStoppedAnimation(_statusColor),
              ),
            )
          else
            StatusDot(color: _statusColor),
          const SizedBox(width: 5),
          Text(
            'API $_apiStatus',
            style: TextStyle(fontSize: 11, fontWeight: FontWeight.w600, color: _statusColor),
          ),
        ],
      ),
    );
  }

  // ─── Hero ─────────────────────────────────────────────────────────────────

  Widget _buildHeroCard() {
    return FadeInDown(
      duration: const Duration(milliseconds: 500),
      child: Container(
        width: double.infinity,
        padding: const EdgeInsets.all(24),
        decoration: BoxDecoration(
          gradient: const LinearGradient(
            colors: [Color(0xFF0B1535), Color(0xFF0D1F40), Color(0xFF0A1628)],
            begin: Alignment.topLeft,
            end: Alignment.bottomRight,
          ),
          borderRadius: AppRadius.xl,
          border: Border.all(color: AppColors.blue.withOpacity(0.3), width: 1),
          boxShadow: [
            BoxShadow(color: AppColors.blue.withOpacity(0.1), blurRadius: 30, spreadRadius: 0),
          ],
        ),
        child: Stack(
          children: [
            // Background orbs
            Positioned(
              right: -20,
              top: -20,
              child: Container(
                width: 120,
                height: 120,
                decoration: BoxDecoration(
                  shape: BoxShape.circle,
                  gradient: RadialGradient(
                    colors: [AppColors.blue.withOpacity(0.15), Colors.transparent],
                  ),
                ),
              ),
            ),
            Positioned(
              right: 40,
              bottom: -30,
              child: Container(
                width: 80,
                height: 80,
                decoration: BoxDecoration(
                  shape: BoxShape.circle,
                  gradient: RadialGradient(
                    colors: [AppColors.teal.withOpacity(0.1), Colors.transparent],
                  ),
                ),
              ),
            ),
            // Content
            Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                PillChip(label: 'AI-POWERED VALUATION', color: AppColors.blue),
                const SizedBox(height: 16),
                const Text(
                  'Instant. Accurate.\nUnbeatable.',
                  style: TextStyle(
                    fontSize: 28,
                    fontWeight: FontWeight.w900,
                    color: AppColors.text,
                    height: 1.15,
                    letterSpacing: -0.8,
                  ),
                ),
                const SizedBox(height: 12),
                const Text(
                  'Get AI-powered price predictions for any pre-owned vehicle in the Indian market — in under 2 seconds.',
                  style: TextStyle(
                    fontSize: 14,
                    color: AppColors.textSub,
                    height: 1.6,
                  ),
                ),
                const SizedBox(height: 22),
                Row(
                  children: [
                    GradientButton(
                      label: 'Get Price Quote',
                      icon: const Icon(Icons.arrow_forward_rounded, color: Colors.white, size: 16),
                      height: 44,
                      onPressed: () {},
                    ),
                    const SizedBox(width: 12),
                    Container(
                      height: 44,
                      padding: const EdgeInsets.symmetric(horizontal: 16),
                      decoration: BoxDecoration(
                        border: Border.all(color: AppColors.border),
                        borderRadius: AppRadius.md,
                      ),
                      child: const Center(
                        child: Text(
                          'View Market',
                          style: TextStyle(
                            fontSize: 13,
                            fontWeight: FontWeight.w600,
                            color: AppColors.textSub,
                          ),
                        ),
                      ),
                    ),
                  ],
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }

  // ─── Stats ────────────────────────────────────────────────────────────────

  Widget _buildStatsRow() {
    return FadeInUp(
      delay: const Duration(milliseconds: 150),
      duration: const Duration(milliseconds: 400),
      child: Row(
        children: [
          _statTile('93%', 'Accuracy', AppColors.blue, Icons.verified_rounded),
          const SizedBox(width: 10),
          _statTile('<2s', 'Speed', AppColors.teal, Icons.bolt_rounded),
          const SizedBox(width: 10),
          _statTile('400K+', 'Cars', AppColors.amber, Icons.directions_car_rounded),
          const SizedBox(width: 10),
          _statTile('18+', 'Brands', AppColors.purple, Icons.star_rounded),
        ],
      ),
    );
  }

  Widget _statTile(String value, String label, Color color, IconData icon) {
    return Expanded(
      child: Container(
        padding: const EdgeInsets.symmetric(vertical: 14, horizontal: 8),
        decoration: BoxDecoration(
          color: AppColors.surface,
          borderRadius: AppRadius.md,
          border: Border.all(color: AppColors.border, width: 0.5),
        ),
        child: Column(
          children: [
            Icon(icon, color: color, size: 20),
            const SizedBox(height: 6),
            Text(value, style: TextStyle(fontSize: 15, fontWeight: FontWeight.w800, color: color)),
            Text(label, style: const TextStyle(fontSize: 10, color: AppColors.muted)),
          ],
        ),
      ),
    );
  }

  // ─── Features ─────────────────────────────────────────────────────────────

  Widget _buildSectionLabel(String label) {
    return Row(
      children: [
        Container(width: 3, height: 14, decoration: BoxDecoration(gradient: AppGradients.blueGlow, borderRadius: AppRadius.pill)),
        const SizedBox(width: 8),
        Text(label, style: const TextStyle(fontSize: 11, fontWeight: FontWeight.w700, color: AppColors.muted, letterSpacing: 1.2)),
      ],
    );
  }

  Widget _buildFeatures() {
    final features = [
      _Feature(
        icon: Icons.psychology_rounded,
        title: 'Ensemble AI Model',
        subtitle: 'XGBoost + LightGBM + CatBoost stacking for unmatched accuracy',
        color: AppColors.blue,
        gradient: AppGradients.blueGlow,
      ),
      _Feature(
        icon: Icons.trending_up_rounded,
        title: 'Market Analytics',
        subtitle: 'Real-time depreciation curves & demand heatmaps',
        color: AppColors.teal,
        gradient: AppGradients.greenGlow,
      ),
      _Feature(
        icon: Icons.flash_on_rounded,
        title: 'Instant Quotes',
        subtitle: 'Sub-2-second predictions, faster than any manual valuation',
        color: AppColors.amber,
        gradient: AppGradients.amberGlow,
      ),
      _Feature(
        icon: Icons.location_on_rounded,
        title: 'India-First Dataset',
        subtitle: '400K+ listings from CarDekho, calibrated for Indian roads',
        color: AppColors.pink,
        gradient: AppGradients.pinkGlow,
      ),
    ];

    return FadeInUp(
      delay: const Duration(milliseconds: 250),
      duration: const Duration(milliseconds: 400),
      child: Column(
        children: features
            .map((f) => _featureTile(f))
            .toList(),
      ),
    );
  }

  Widget _featureTile(_Feature f) {
    return Container(
      margin: const EdgeInsets.only(bottom: 10),
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: AppColors.surface,
        borderRadius: AppRadius.lg,
        border: Border.all(color: AppColors.border, width: 0.5),
      ),
      child: Row(
        children: [
          Container(
            width: 42,
            height: 42,
            decoration: BoxDecoration(
              gradient: f.gradient,
              borderRadius: AppRadius.md,
              boxShadow: [BoxShadow(color: f.color.withOpacity(0.3), blurRadius: 10)],
            ),
            child: Icon(f.icon, color: Colors.white, size: 20),
          ),
          const SizedBox(width: 14),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(f.title, style: AppTextStyles.cardTitle),
                const SizedBox(height: 3),
                Text(f.subtitle, style: AppTextStyles.caption),
              ],
            ),
          ),
          Icon(Icons.chevron_right_rounded, color: AppColors.muted, size: 18),
        ],
      ),
    );
  }

  // ─── How it works ─────────────────────────────────────────────────────────

  Widget _buildHowItWorks() {
    final steps = [
      ('01', 'Enter Details', 'Brand, year, mileage & specs', AppColors.blue),
      ('02', 'AI Analysis', 'Ensemble model evaluates 400K+ data points', AppColors.teal),
      ('03', 'Get Quote', 'Receive instant valuation with confidence range', AppColors.amber),
    ];

    return FadeInUp(
      delay: const Duration(milliseconds: 350),
      duration: const Duration(milliseconds: 400),
      child: Container(
        padding: const EdgeInsets.all(20),
        decoration: BoxDecoration(
          color: AppColors.surface,
          borderRadius: AppRadius.xl,
          border: Border.all(color: AppColors.border, width: 0.5),
        ),
        child: Column(
          children: steps.asMap().entries.map((e) {
            final step = e.value;
            final isLast = e.key == steps.length - 1;
            return Column(
              children: [
                Row(
                  children: [
                    Column(
                      children: [
                        Container(
                          width: 36,
                          height: 36,
                          decoration: BoxDecoration(
                            color: step.$4.withOpacity(0.12),
                            borderRadius: AppRadius.md,
                            border: Border.all(color: step.$4.withOpacity(0.35)),
                          ),
                          child: Center(
                            child: Text(
                              step.$1,
                              style: TextStyle(fontSize: 11, fontWeight: FontWeight.w800, color: step.$4),
                            ),
                          ),
                        ),
                        if (!isLast)
                          Container(
                            width: 1,
                            height: 28,
                            margin: const EdgeInsets.symmetric(vertical: 4),
                            decoration: BoxDecoration(
                              gradient: LinearGradient(
                                colors: [step.$4.withOpacity(0.4), Colors.transparent],
                                begin: Alignment.topCenter,
                                end: Alignment.bottomCenter,
                              ),
                            ),
                          ),
                      ],
                    ),
                    const SizedBox(width: 14),
                    Expanded(
                      child: Padding(
                        padding: EdgeInsets.only(bottom: isLast ? 0 : 28),
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Text(step.$2, style: AppTextStyles.cardTitle),
                            const SizedBox(height: 3),
                            Text(step.$3, style: AppTextStyles.caption),
                          ],
                        ),
                      ),
                    ),
                  ],
                ),
              ],
            );
          }).toList(),
        ),
      ),
    );
  }

  // ─── Footer ───────────────────────────────────────────────────────────────

  Widget _buildFooterBadge() {
    return FadeIn(
      delay: const Duration(milliseconds: 500),
      child: Center(
        child: Container(
          padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
          decoration: BoxDecoration(
            color: AppColors.surface2,
            borderRadius: AppRadius.pill,
            border: Border.all(color: AppColors.border, width: 0.5),
          ),
          child: const Row(
            mainAxisSize: MainAxisSize.min,
            children: [
              Icon(Icons.bolt_rounded, size: 14, color: AppColors.amber),
              SizedBox(width: 6),
              Text(
                'Built by ForgePoint AI · 2026',
                style: TextStyle(fontSize: 11, fontWeight: FontWeight.w500, color: AppColors.muted),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

// ─── Data class ───────────────────────────────────────────────────────────────

class _Feature {
  final IconData icon;
  final String title;
  final String subtitle;
  final Color color;
  final Gradient gradient;
  const _Feature({required this.icon, required this.title, required this.subtitle, required this.color, required this.gradient});
}