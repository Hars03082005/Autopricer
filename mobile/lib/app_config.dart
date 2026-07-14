import 'dart:io';

/// Runtime configuration for the PriceRef mobile shell.
///
/// Override at build/run time with dart-define:
///   flutter run --dart-define=API_URL=http://192.168.1.10:8000
///   flutter run --dart-define=WEB_URL=http://192.168.1.10:5173
class AppConfig {
  static const String apiUrl = String.fromEnvironment('API_URL');
  static const String webUrl = String.fromEnvironment('WEB_URL');

  static String get apiBaseUrl {
    if (apiUrl.isNotEmpty) return apiUrl;
    if (Platform.isAndroid) return 'http://10.0.2.2:8000';
    return 'http://localhost:8000';
  }

  static bool get useBundledWeb => webUrl.isEmpty;

  static String get devWebUrl => webUrl;
}
