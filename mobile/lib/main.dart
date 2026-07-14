import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:webview_flutter/webview_flutter.dart';

import 'app_config.dart';
import 'asset_server.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  SystemChrome.setPreferredOrientations([
    DeviceOrientation.portraitUp,
    DeviceOrientation.portraitDown,
  ]);
  runApp(const PriceRefApp());
}

class PriceRefApp extends StatelessWidget {
  const PriceRefApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'PriceRef',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: const Color(0xFFF75D34)),
        useMaterial3: true,
      ),
      home: const PriceRefWebShell(),
    );
  }
}

class PriceRefWebShell extends StatefulWidget {
  const PriceRefWebShell({super.key});

  @override
  State<PriceRefWebShell> createState() => _PriceRefWebShellState();
}

class _PriceRefWebShellState extends State<PriceRefWebShell> {
  late final WebViewController _controller;
  var _loading = true;
  var _loadError = '';
  int _serverPort = 0;

  @override
  void initState() {
    super.initState();
    _controller = WebViewController()
      ..setJavaScriptMode(JavaScriptMode.unrestricted)
      ..setBackgroundColor(const Color(0xFFF5F5F5))
      ..setNavigationDelegate(
        NavigationDelegate(
          onPageStarted: (_) {
            if (mounted) {
              setState(() {
                _loading = true;
                _loadError = '';
              });
            }
          },
          onPageFinished: (_) => _injectRuntimeConfig(),
          onWebResourceError: (error) {
            // Only show the full-screen error overlay for the main page
            // load failure. API fetch timeouts (ERR_CONNECTION_TIMED_OUT)
            // targeting the FastAPI backend are sub-resource errors and
            // must NOT replace the entire loaded UI.
            final url = error.url ?? '';
            final isMainPage = url.isEmpty ||
                url == 'http://localhost:$_serverPort/' ||
                url.endsWith('/index.html') ||
                url == AppConfig.devWebUrl;
            if (isMainPage && mounted) {
              setState(() {
                _loadError = error.description;
                _loading = false;
              });
            }
          },
        ),
      );

    _initAndLoad();
  }

  /// Starts the local asset HTTP server then loads the web app.
  Future<void> _initAndLoad() async {
    if (AppConfig.useBundledWeb) {
      _serverPort = await AssetServer.start();
    }
    await _loadWebApp();
  }

  @override
  void dispose() {
    AssetServer.stop();
    super.dispose();
  }

  Future<void> _injectRuntimeConfig() async {
    final apiBase = AppConfig.apiBaseUrl.replaceAll("'", "\\'");
    await _controller.runJavaScript(
      "window.PriceRef_API_URL = '$apiBase';",
    );
    if (mounted) {
      setState(() => _loading = false);
    }
  }

  Future<void> _loadWebApp() async {
    if (AppConfig.useBundledWeb) {
      // Serve via local HTTP server to avoid Android file:// cross-origin
      // restrictions (net::ERR_ACCESS_DENIED with loadFlutterAsset).
      final url = 'http://localhost:$_serverPort/';
      await _controller.loadRequest(Uri.parse(url));
      return;
    }
    await _controller.loadRequest(Uri.parse(AppConfig.devWebUrl));
  }

  Future<void> _reload() async {
    setState(() {
      _loading = true;
      _loadError = '';
    });
    await _loadWebApp();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: SafeArea(
        child: Stack(
          children: [
            WebViewWidget(controller: _controller),
            if (_loading)
              const ColoredBox(
                color: Color(0xFFF5F5F5),
                child: Center(
                  child: Column(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      Icon(Icons.directions_car, size: 48, color: Color(0xFFF75D34)),
                      SizedBox(height: 16),
                      Text(
                        'PriceRef',
                        style: TextStyle(
                          fontSize: 22,
                          fontWeight: FontWeight.w700,
                          color: Color(0xFF24272C),
                        ),
                      ),
                      SizedBox(height: 12),
                      CircularProgressIndicator(color: Color(0xFFF75D34)),
                    ],
                  ),
                ),
              ),
            if (_loadError.isNotEmpty)
              ColoredBox(
                color: const Color(0xFFF5F5F5),
                child: Center(
                  child: Padding(
                    padding: const EdgeInsets.all(24),
                    child: Column(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        const Icon(Icons.wifi_off, size: 40, color: Color(0xFFE02020)),
                        const SizedBox(height: 12),
                        const Text(
                          'Could not load PriceRef',
                          style: TextStyle(fontSize: 18, fontWeight: FontWeight.w700),
                        ),
                        const SizedBox(height: 8),
                        Text(
                          _loadError,
                          textAlign: TextAlign.center,
                          style: const TextStyle(color: Color(0xFF666666)),
                        ),
                        const SizedBox(height: 16),
                        FilledButton(
                          onPressed: _reload,
                          style: FilledButton.styleFrom(
                            backgroundColor: const Color(0xFFF75D34),
                          ),
                          child: const Text('Retry'),
                        ),
                      ],
                    ),
                  ),
                ),
              ),
          ],
        ),
      ),
    );
  }
}
