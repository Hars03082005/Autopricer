import 'dart:io';

/// Runtime configuration for the PriceRef mobile shell.
///
/// Override at build/run time with dart-define:
///   flutter run --dart-define=API_URL=http://192.168.1.10:8000
///   flutter run --dart-define=WEB_URL=http://192.168.1.10:5173
class AppConfig {
  static const String apiUrl = String.fromEnvironment('API_URL');
  static const String webUrl = String.fromEnvironment('WEB_URL');

  /// Live Render backend — used on physical devices and release builds.
  static const String _prodApiUrl =
      'https://price-prediction-backend.onrender.com';

  static String get apiBaseUrl {
    // --dart-define=API_URL wins (dev / custom env)
    if (apiUrl.isNotEmpty) return apiUrl;
    // Android emulator: 10.0.2.2 maps to the host machine's localhost
    if (Platform.isAndroid &&
        const bool.fromEnvironment('IS_EMULATOR', defaultValue: false)) {
      return 'http://10.0.2.2:8000';
    }
    // All other cases (physical device, iOS, release) → live Render backend
    return _prodApiUrl;
  }

  static bool get useBundledWeb => webUrl.isEmpty;

  static String get devWebUrl => webUrl;
}
