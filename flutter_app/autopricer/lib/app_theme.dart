import 'package:flutter/material.dart';

// ─── Colour Palette ────────────────────────────────────────────────────────────
abstract class AppColors {
  // Backgrounds
  static const Color bg       = Color(0xFF080C14);
  static const Color surface  = Color(0xFF0F1624);
  static const Color surface2 = Color(0xFF162032);
  static const Color surface3 = Color(0xFF1C2A3F);

  // Accents
  static const Color blue     = Color(0xFF4F8EF7);
  static const Color blueDark = Color(0xFF2F5ECC);
  static const Color teal     = Color(0xFF00D9A3);
  static const Color amber    = Color(0xFFFFB84D);
  static const Color pink     = Color(0xFFFF6B9D);
  static const Color purple   = Color(0xFFA78BFA);
  static const Color green    = Color(0xFF34D399);
  static const Color red      = Color(0xFFFF5252);

  // Gradient stops
  static const Color gradStart = Color(0xFF1A2FFF);
  static const Color gradEnd   = Color(0xFF0099FF);

  // Typography
  static const Color text      = Color(0xFFF0F4FF);
  static const Color textSub   = Color(0xFFB0C0D8);
  static const Color muted     = Color(0xFF6B7A99);

  // Borders & dividers
  static const Color border    = Color(0xFF1F2E45);
  static const Color borderHi  = Color(0xFF2E4060);
}

// ─── Radius Tokens ────────────────────────────────────────────────────────────
abstract class AppRadius {
  static const BorderRadius sm   = BorderRadius.all(Radius.circular(8));
  static const BorderRadius md   = BorderRadius.all(Radius.circular(12));
  static const BorderRadius lg   = BorderRadius.all(Radius.circular(16));
  static const BorderRadius xl   = BorderRadius.all(Radius.circular(20));
  static const BorderRadius xxl  = BorderRadius.all(Radius.circular(24));
  static const BorderRadius pill = BorderRadius.all(Radius.circular(100));
}

// ─── Typography ───────────────────────────────────────────────────────────────
abstract class AppTextStyles {
  static const TextStyle h1 = TextStyle(
    fontSize: 28, fontWeight: FontWeight.w800,
    color: AppColors.text, letterSpacing: -0.5,
  );
  static const TextStyle h2 = TextStyle(
    fontSize: 22, fontWeight: FontWeight.w700,
    color: AppColors.text, letterSpacing: -0.3,
  );
  static const TextStyle h3 = TextStyle(
    fontSize: 17, fontWeight: FontWeight.w700,
    color: AppColors.text,
  );
  static const TextStyle cardTitle = TextStyle(
    fontSize: 15, fontWeight: FontWeight.w700,
    color: AppColors.text, letterSpacing: 0.1,
  );
  static const TextStyle cardSubtitle = TextStyle(
    fontSize: 11, fontWeight: FontWeight.w500,
    color: AppColors.muted,
  );
  static const TextStyle label = TextStyle(
    fontSize: 12, fontWeight: FontWeight.w600,
    color: AppColors.textSub, letterSpacing: 0.8,
  );
  static const TextStyle body = TextStyle(
    fontSize: 14, fontWeight: FontWeight.w400,
    color: AppColors.textSub, height: 1.6,
  );
  static const TextStyle caption = TextStyle(
    fontSize: 11, fontWeight: FontWeight.w500,
    color: AppColors.muted,
  );
  static const TextStyle kpi = TextStyle(
    fontSize: 22, fontWeight: FontWeight.w800,
    color: AppColors.text, letterSpacing: -0.5,
  );
}

// ─── Gradients ────────────────────────────────────────────────────────────────
abstract class AppGradients {
  static const LinearGradient hero = LinearGradient(
    colors: [Color(0xFF1A2FFF), Color(0xFF0099FF), Color(0xFF00D9A3)],
    begin: Alignment.topLeft,
    end: Alignment.bottomRight,
    stops: [0.0, 0.55, 1.0],
  );
  static const LinearGradient card = LinearGradient(
    colors: [Color(0xFF0F1624), Color(0xFF162032)],
    begin: Alignment.topLeft,
    end: Alignment.bottomRight,
  );
  static const LinearGradient blueGlow = LinearGradient(
    colors: [Color(0xFF4F8EF7), Color(0xFF0099FF)],
    begin: Alignment.topLeft,
    end: Alignment.bottomRight,
  );
  static const LinearGradient greenGlow = LinearGradient(
    colors: [Color(0xFF00D9A3), Color(0xFF00B884)],
    begin: Alignment.topLeft,
    end: Alignment.bottomRight,
  );
  static const LinearGradient amberGlow = LinearGradient(
    colors: [Color(0xFFFFB84D), Color(0xFFF59E0B)],
    begin: Alignment.topLeft,
    end: Alignment.bottomRight,
  );
  static const LinearGradient pinkGlow = LinearGradient(
    colors: [Color(0xFFFF6B9D), Color(0xFFEC4899)],
    begin: Alignment.topLeft,
    end: Alignment.bottomRight,
  );
  static const LinearGradient purpleGlow = LinearGradient(
    colors: [Color(0xFFA78BFA), Color(0xFF7C3AED)],
    begin: Alignment.topLeft,
    end: Alignment.bottomRight,
  );
}
