import 'package:flutter/material.dart';
import 'package:animate_do/animate_do.dart';
import 'package:intl/intl.dart';
import '../app_theme.dart';
import '../widgets/common.dart';
import '../services/api_service.dart';

class PredictScreen extends StatefulWidget {
  const PredictScreen({super.key});

  @override
  State<PredictScreen> createState() => _PredictScreenState();
}

class _PredictScreenState extends State<PredictScreen>
    with SingleTickerProviderStateMixin {

  final _formKey = GlobalKey<FormState>();

  // ── Form state ──────────────────────────────────────────────────────────────
  String _brand       = 'Maruti';
  int    _year        = 2019;
  double _kmDriven    = 45000;
  String _fuelType    = 'Petrol';
  String _transmission = 'Manual';
  String _owner       = 'First Owner';
  String _sellerType  = 'Individual';
  double _engineCc    = 1200;
  double _maxPower    = 85;
  int    _seats       = 5;

  bool   _isLoading   = false;
  Map<String, dynamic>? _result;

  late final AnimationController _pulseCtrl;
  late final Animation<double> _pulseAnim;

  // ── Reference data ──────────────────────────────────────────────────────────
  final List<String> _brands = [
    'Maruti','Hyundai','Honda','Tata','Kia',
    'Toyota','Mahindra','Volkswagen','Skoda',
    'BMW','Mercedes','Audi','Ford','Renault',
    'Nissan','MG','Jeep','Volvo',
  ];
  final List<String> _fuelTypes = ['Petrol','Diesel','CNG','Electric','Hybrid'];
  final List<String> _transmissions = ['Manual','Automatic'];
  final List<String> _owners = [
    'First Owner','Second Owner','Third Owner','Fourth & Above Owner',
  ];
  final List<String> _sellerTypes = [
    'Individual','Dealer','Trustmark Dealer',
  ];

  final _currency = NumberFormat.currency(
    locale: 'en_IN', symbol: '₹', decimalDigits: 0,
  );

  // ── Helpers ─────────────────────────────────────────────────────────────────
  String _getBrandTier(String b) {
    if (['Maruti','Datsun','Renault'].contains(b)) return 'Budget';
    if (['Hyundai','Honda','Tata','Kia','Nissan','Ford','Mahindra'].contains(b)) return 'Mid';
    if (['Volkswagen','Skoda','Toyota','MG','Jeep'].contains(b)) return 'Premium';
    if (['BMW','Mercedes','Audi','Volvo','Jaguar'].contains(b)) return 'Luxury';
    return 'Mid';
  }

  int _getOwnerNum(String o) {
    switch (o) {
      case 'First Owner':  return 1;
      case 'Second Owner': return 2;
      case 'Third Owner':  return 3;
      default:             return 4;
    }
  }

  // ── Lifecycle ────────────────────────────────────────────────────────────────
  @override
  void initState() {
    super.initState();
    _pulseCtrl = AnimationController(vsync: this, duration: const Duration(seconds: 2))
      ..repeat(reverse: true);
    _pulseAnim = Tween<double>(begin: 0.7, end: 1.0).animate(
      CurvedAnimation(parent: _pulseCtrl, curve: Curves.easeInOut),
    );
  }

  @override
  void dispose() {
    _pulseCtrl.dispose();
    super.dispose();
  }

  // ── Prediction ──────────────────────────────────────────────────────────────
  Future<void> _predict() async {
    setState(() { _isLoading = true; _result = null; });
    try {
      final age = 2026 - _year;
      final kmPerYear = _kmDriven / (age == 0 ? 0.5 : age.toDouble());
      final result = await ApiService.predictPrice({
        'brand': _brand, 'year': _year,
        'km_driven': _kmDriven, 'fuel_type': _fuelType,
        'transmission': _transmission, 'owner': _owner,
        'seller_type': _sellerType, 'engine_cc': _engineCc,
        'max_power_bhp': _maxPower, 'seats': _seats,
        'brand_tier': _getBrandTier(_brand),
        'owner_num': _getOwnerNum(_owner),
        'age': age, 'km_per_year': kmPerYear,
        'age_km': age * _kmDriven,
        'is_low_mileage': kmPerYear < 5000 ? 1 : 0,
      });
      setState(() { _result = result; _isLoading = false; });
    } catch (e) {
      setState(() => _isLoading = false);
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('Prediction failed: ${e.toString()}'),
            backgroundColor: AppColors.surface3,
          ),
        );
      }
    }
  }

  // ── UI ───────────────────────────────────────────────────────────────────────
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
              child: Form(
                key: _formKey,
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    if (_result != null) ...[
                      FadeInDown(child: _buildResultCard()),
                      const SizedBox(height: 20),
                    ],
                    _buildSectionChip('VEHICLE IDENTITY'),
                    const SizedBox(height: 10),
                    _buildDropdown('Brand', _brand, _brands, Icons.branding_watermark_rounded,
                        (v) => setState(() => _brand = v!)),
                    _buildYearAndKmRow(),
                    const SizedBox(height: 20),
                    _buildSectionChip('POWERTRAIN'),
                    const SizedBox(height: 10),
                    _buildFuelTransmissionRow(),
                    _buildSlider('Engine Displacement', _engineCc, 600, 5000,
                        Icons.settings_rounded, (v) => setState(() => _engineCc = v),
                        suffix: ' cc', divisions: 88),
                    _buildSlider('Max Power', _maxPower, 40, 500,
                        Icons.flash_on_rounded, (v) => setState(() => _maxPower = v),
                        suffix: ' BHP', divisions: 92),
                    const SizedBox(height: 20),
                    _buildSectionChip('OWNERSHIP & SELLER'),
                    const SizedBox(height: 10),
                    _buildDropdown('Ownership History', _owner, _owners, Icons.person_rounded,
                        (v) => setState(() => _owner = v!)),
                    _buildDropdown('Seller Type', _sellerType, _sellerTypes, Icons.storefront_rounded,
                        (v) => setState(() => _sellerType = v!)),
                    _buildDropdown('Seats', _seats.toString(), ['2','4','5','6','7','8'],
                        Icons.airline_seat_recline_normal_rounded,
                        (v) => setState(() => _seats = int.parse(v!))),
                    const SizedBox(height: 24),
                    _buildPredictButton(),
                    const SizedBox(height: 20),
                    _buildDisclaimerChip(),
                  ],
                ),
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
      titleSpacing: 20,
      title: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: const [
          Text('Price Predictor', style: TextStyle(fontSize: 17, fontWeight: FontWeight.w800, color: AppColors.text)),
          Text('Configure your vehicle below', style: TextStyle(fontSize: 11, color: AppColors.muted, fontWeight: FontWeight.w400)),
        ],
      ),
      bottom: PreferredSize(
        preferredSize: const Size.fromHeight(0.5),
        child: Container(height: 0.5, color: AppColors.border),
      ),
    );
  }

  // ─── Section chip ────────────────────────────────────────────────────────
  Widget _buildSectionChip(String label) {
    return Row(
      children: [
        Container(width: 3, height: 12, decoration: BoxDecoration(gradient: AppGradients.blueGlow, borderRadius: AppRadius.pill)),
        const SizedBox(width: 8),
        Text(label, style: const TextStyle(fontSize: 10, fontWeight: FontWeight.w700, color: AppColors.muted, letterSpacing: 1.2)),
      ],
    );
  }

  // ─── Two-column helpers ───────────────────────────────────────────────────
  Widget _buildYearAndKmRow() {
    return Column(
      children: [
        _buildSlider('Year of Manufacture', _year.toDouble(), 2000, 2025,
            Icons.calendar_today_rounded, (v) => setState(() => _year = v.toInt()),
            isInt: true),
        _buildSlider('Kilometres Driven', _kmDriven, 500, 400000,
            Icons.speed_rounded, (v) => setState(() => _kmDriven = v),
            suffix: ' km', divisions: 800),
      ],
    );
  }

  Widget _buildFuelTransmissionRow() {
    return Row(
      children: [
        Expanded(child: _buildDropdown('Fuel', _fuelType, _fuelTypes, Icons.local_gas_station_rounded,
            (v) => setState(() => _fuelType = v!), compact: true)),
        const SizedBox(width: 10),
        Expanded(child: _buildDropdown('Gearbox', _transmission, _transmissions, Icons.settings_rounded,
            (v) => setState(() => _transmission = v!), compact: true)),
      ],
    );
  }

  // ─── Result card ─────────────────────────────────────────────────────────
  Widget _buildResultCard() {
    final predicted = (_result!['predicted_price'] ?? _result!['predicted'] ?? 0) as num;
    final low       = (_result!['price_range_low'] ?? _result!['low'] ?? 0)       as num;
    final high      = (_result!['price_range_high'] ?? _result!['high'] ?? 0)     as num;
    final range     = high - low;
    final progress  = range > 0 ? ((predicted - low) / range).clamp(0.0, 1.0) : 0.92;

    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(24),
      decoration: BoxDecoration(
        gradient: const LinearGradient(
          colors: [Color(0xFF0A1F2E), Color(0xFF0C2335), Color(0xFF071A20)],
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
        ),
        borderRadius: AppRadius.xl,
        border: Border.all(color: AppColors.teal.withOpacity(0.4), width: 1),
        boxShadow: [
          BoxShadow(color: AppColors.teal.withOpacity(0.12), blurRadius: 30, spreadRadius: 0),
        ],
      ),
      child: Column(
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              PillChip(label: 'PREDICTION RESULT', color: AppColors.teal),
              PillChip(label: '92% CONFIDENCE', color: AppColors.blue),
            ],
          ),
          const SizedBox(height: 20),
          AnimatedBuilder(
            animation: _pulseAnim,
            builder: (_, __) => Opacity(
              opacity: _pulseAnim.value,
              child: Text(
                _currency.format(predicted),
                style: const TextStyle(
                  fontSize: 42,
                  fontWeight: FontWeight.w900,
                  color: AppColors.teal,
                  letterSpacing: -1.5,
                ),
              ),
            ),
          ),
          const SizedBox(height: 6),
          Text(
            'Fair Range: ${_currency.format(low)} — ${_currency.format(high)}',
            style: const TextStyle(fontSize: 13, color: AppColors.muted),
          ),
          const SizedBox(height: 20),
          // Progress bar
          Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  Text(_currency.format(low), style: const TextStyle(fontSize: 10, color: AppColors.muted)),
                  const Text('Predicted Value', style: TextStyle(fontSize: 10, color: AppColors.muted)),
                  Text(_currency.format(high), style: const TextStyle(fontSize: 10, color: AppColors.muted)),
                ],
              ),
              const SizedBox(height: 6),
              ClipRRect(
                borderRadius: AppRadius.pill,
                child: Stack(
                  children: [
                    Container(height: 6, color: AppColors.surface3),
                    FractionallySizedBox(
                      widthFactor: progress.toDouble(),
                      child: Container(
                        height: 6,
                        decoration: const BoxDecoration(
                          gradient: AppGradients.greenGlow,
                          borderRadius: AppRadius.pill,
                        ),
                      ),
                    ),
                  ],
                ),
              ),
            ],
          ),
          const SizedBox(height: 16),
          // Model breakdown
          Row(
            children: [
              _miniModelBadge('XGBoost', AppColors.blue),
              const SizedBox(width: 6),
              _miniModelBadge('LightGBM', AppColors.teal),
              const SizedBox(width: 6),
              _miniModelBadge('CatBoost', AppColors.amber),
            ],
          ),
        ],
      ),
    );
  }

  Widget _miniModelBadge(String name, Color color) {
    return Expanded(
      child: Container(
        padding: const EdgeInsets.symmetric(vertical: 7),
        decoration: BoxDecoration(
          color: color.withOpacity(0.1),
          borderRadius: AppRadius.sm,
          border: Border.all(color: color.withOpacity(0.3)),
        ),
        child: Center(
          child: Text(name, style: TextStyle(fontSize: 10, fontWeight: FontWeight.w700, color: color)),
        ),
      ),
    );
  }

  // ─── Dropdown ─────────────────────────────────────────────────────────────
  Widget _buildDropdown(
    String label,
    String value,
    List<String> items,
    IconData icon,
    ValueChanged<String?> onChanged, {
    bool compact = false,
  }) {
    return Container(
      margin: const EdgeInsets.only(bottom: 10),
      padding: EdgeInsets.symmetric(horizontal: 14, vertical: compact ? 0 : 2),
      decoration: BoxDecoration(
        color: AppColors.surface,
        borderRadius: AppRadius.md,
        border: Border.all(color: AppColors.border, width: 0.5),
      ),
      child: DropdownButtonFormField<String>(
        value: items.contains(value) ? value : items.first,
        dropdownColor: AppColors.surface2,
        isExpanded: true,
        icon: const Icon(Icons.keyboard_arrow_down_rounded, color: AppColors.muted, size: 18),
        decoration: InputDecoration(
          labelText: label,
          labelStyle: const TextStyle(color: AppColors.muted, fontSize: 12),
          prefixIcon: Padding(
            padding: const EdgeInsets.only(right: 4),
            child: Icon(icon, color: AppColors.blue, size: 18),
          ),
          border: InputBorder.none,
          contentPadding: EdgeInsets.symmetric(vertical: compact ? 10 : 14),
        ),
        style: const TextStyle(color: AppColors.text, fontSize: 14, fontWeight: FontWeight.w600),
        items: items.map((e) => DropdownMenuItem(value: e, child: Text(e))).toList(),
        onChanged: onChanged,
      ),
    );
  }

  // ─── Slider ───────────────────────────────────────────────────────────────
  Widget _buildSlider(
    String label,
    double value,
    double min,
    double max,
    IconData icon,
    ValueChanged<double> onChanged, {
    String suffix = '',
    int divisions = 100,
    bool isInt = false,
  }) {
    final display = isInt
        ? value.toInt().toString()
        : value >= 1000
            ? '${(value / 1000).toStringAsFixed(0)}K'
            : value.toStringAsFixed(0);

    return Container(
      margin: const EdgeInsets.only(bottom: 10),
      padding: const EdgeInsets.fromLTRB(16, 14, 16, 8),
      decoration: BoxDecoration(
        color: AppColors.surface,
        borderRadius: AppRadius.md,
        border: Border.all(color: AppColors.border, width: 0.5),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Icon(icon, color: AppColors.blue, size: 16),
              const SizedBox(width: 8),
              Text(label, style: const TextStyle(fontSize: 12, color: AppColors.muted, fontWeight: FontWeight.w500)),
              const Spacer(),
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
                decoration: BoxDecoration(
                  color: AppColors.surface2,
                  borderRadius: AppRadius.sm,
                  border: Border.all(color: AppColors.borderHi, width: 0.5),
                ),
                child: Text(
                  '$display$suffix',
                  style: const TextStyle(fontSize: 12, fontWeight: FontWeight.w700, color: AppColors.text),
                ),
              ),
            ],
          ),
          const SizedBox(height: 4),
          SliderTheme(
            data: SliderTheme.of(context).copyWith(
              trackHeight: 3,
              thumbShape: const RoundSliderThumbShape(enabledThumbRadius: 7),
              overlayShape: const RoundSliderOverlayShape(overlayRadius: 14),
            ),
            child: Slider(
              value: value.clamp(min, max),
              min: min, max: max,
              divisions: divisions,
              onChanged: onChanged,
            ),
          ),
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Text('${isInt ? min.toInt() : (min >= 1000 ? "${(min/1000).toStringAsFixed(0)}K" : min.toStringAsFixed(0))}$suffix',
                  style: const TextStyle(fontSize: 9, color: AppColors.muted)),
              Text('${isInt ? max.toInt() : (max >= 1000 ? "${(max/1000).toStringAsFixed(0)}K" : max.toStringAsFixed(0))}$suffix',
                  style: const TextStyle(fontSize: 9, color: AppColors.muted)),
            ],
          ),
        ],
      ),
    );
  }

  // ─── Predict button ───────────────────────────────────────────────────────
  Widget _buildPredictButton() {
    return SizedBox(
      width: double.infinity,
      child: GradientButton(
        label: 'Predict Market Value',
        icon: const Icon(Icons.auto_awesome_rounded, color: Colors.white, size: 18),
        isLoading: _isLoading,
        onPressed: _isLoading ? null : _predict,
        height: 56,
      ),
    );
  }

  Widget _buildDisclaimerChip() {
    return Center(
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: const [
          Icon(Icons.info_outline_rounded, size: 12, color: AppColors.muted),
          SizedBox(width: 5),
          Text(
            'Estimates are indicative. Actual market price may vary.',
            style: TextStyle(fontSize: 10, color: AppColors.muted),
          ),
        ],
      ),
    );
  }
}