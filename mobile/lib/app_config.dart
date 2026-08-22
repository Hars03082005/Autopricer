import 'dart:io';

/// Runtime configuration for the PriceRef mobile shell.
///
/// Override at build/run time with dart-define:
///   flutter run --dart-define=API_URL=https://priceref-frontend.azurecontainerapps.io
///   flutter run --dart-define=WEB_URL=https://priceref-frontend.azurecontainerapps.io
///
/// For local development:
///   flutter run --dart-define=API_URL=http://10.0.2.2:8000 (Android emulator)
///   flutter run --dart-define=API_URL=http://localhost:8000 (iOS simulator)
class AppConfig {
  static const String apiUrl = String.fromEnvironment('API_URL');
  static const String webUrl = String.fromEnvironment('WEB_URL');

  /// Production Azure Container Apps frontend ingress default.
  /// Overridden at build time via --dart-define=API_URL=...
  static const String _defaultAzureApiUrl =
      'https://priceref-frontend.azurecontainerapps.io';

  static String get apiBaseUrl {
    // 1. Explicit build/runtime define wins
    if (apiUrl.isNotEmpty) return apiUrl;

    // 2. Android emulator local mapping
    if (Platform.isAndroid &&
        const bool.fromEnvironment('IS_EMULATOR', defaultValue: false)) {
      return 'http://10.0.2.2:8000';
    }

    // 3. iOS simulator local mapping
    if (Platform.isIOS &&
        const bool.fromEnvironment('IS_EMULATOR', defaultValue: false)) {
      return 'http://localhost:8000';
    }

    // 4. Default production Azure backend ingress
    return _defaultAzureApiUrl;
  }

  static bool get useBundledWeb => webUrl.isEmpty;

  static String get devWebUrl => webUrl;
}
