import 'dart:async';
import 'dart:convert';
import 'dart:io';

import 'package:flutter/services.dart';

import 'app_config.dart';

/// Lightweight in-process HTTP server that serves the bundled React web app
/// from Flutter's rootBundle over http://localhost so the WebView can load
/// all assets (JS, CSS, images) without hitting Android's file:// cross-origin
/// security restrictions (net::ERR_ACCESS_DENIED).
///
/// The server also injects `window.PriceRef_API_URL` directly into the
/// index.html response so the React app has the correct backend URL
/// before any JavaScript module initialises — eliminating the race condition
/// where the Flutter onPageFinished injection arrives too late.
///
/// Usage:
///   final port = await AssetServer.start();                // idempotent
///   controller.loadRequest(Uri.parse('http://localhost:$port/'));
///   AssetServer.stop();                                    // call on dispose
class AssetServer {
  static HttpServer? _server;
  static int _port = 0;

  /// Starts the server on a random free port (OS picks port 0).
  /// Safe to call multiple times — returns the same port if already running.
  static Future<int> start() async {
    if (_server != null) return _port;

    _server = await HttpServer.bind(InternetAddress.loopbackIPv4, 0);
    _port = _server!.port;

    _server!.listen(_handleRequest, onError: (e) {
      // ignore socket errors — client disconnected
    });

    return _port;
  }

  static Future<void> _handleRequest(HttpRequest request) async {
    var path = request.uri.path;

    // Root → index.html
    if (path == '/') path = '/index.html';

    // Strip leading slash and map to asset path
    final assetPath = 'assets/web${path.replaceAll(RegExp(r'\.\.'), '')}';

    try {
      final data = await rootBundle.load(assetPath);
      final bytes = data.buffer.asUint8List();

      // Inject window.PriceRef_API_URL into index.html BEFORE any JS runs.
      // This eliminates the race condition where Flutter's onPageFinished
      // injection arrives after React's initial API calls.
      if (path == '/index.html') {
        final html = utf8.decode(bytes);
        final apiBase = AppConfig.apiBaseUrl.replaceAll("'", "\\'");
        final injected = html.replaceFirst(
          '<head>',
          "<head><script>window.PriceRef_API_URL='$apiBase';</script>",
        );
        final injectedBytes = utf8.encode(injected);
        request.response
          ..statusCode = HttpStatus.ok
          ..headers.set(HttpHeaders.contentTypeHeader, 'text/html; charset=utf-8')
          ..headers.set(HttpHeaders.cacheControlHeader, 'no-cache')
          ..add(injectedBytes);
      } else {
        request.response
          ..statusCode = HttpStatus.ok
          ..headers.set(HttpHeaders.contentTypeHeader, _mimeType(path))
          ..headers.set(HttpHeaders.cacheControlHeader, 'no-cache')
          ..add(bytes);
      }
    } catch (_) {
      request.response.statusCode = HttpStatus.notFound;
    }

    await request.response.close();
  }

  static String _mimeType(String path) {
    if (path.endsWith('.html')) return 'text/html; charset=utf-8';
    if (path.endsWith('.js'))   return 'application/javascript; charset=utf-8';
    if (path.endsWith('.css'))  return 'text/css; charset=utf-8';
    if (path.endsWith('.png'))  return 'image/png';
    if (path.endsWith('.jpg') || path.endsWith('.jpeg')) return 'image/jpeg';
    if (path.endsWith('.svg'))  return 'image/svg+xml';
    if (path.endsWith('.ico'))  return 'image/x-icon';
    if (path.endsWith('.json')) return 'application/json';
    if (path.endsWith('.woff2')) return 'font/woff2';
    if (path.endsWith('.woff'))  return 'font/woff';
    return 'application/octet-stream';
  }

  /// Stops the server and resets state. Call this from the widget's dispose().
  static void stop() {
    _server?.close(force: true);
    _server = null;
    _port = 0;
  }
}
