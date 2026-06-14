import 'dart:convert';
import 'package:http/http.dart' as http;

class ApiService {
  // Change this to your Railway URL
  // when deployed
  static const String baseUrl =
    'https://brush-altitude-silk.ngrok-free.dev'; // Wi-Fi LAN IP — phone must be on same network
  // Use 10.0.2.2 for Android emulator
  // Use your IP for physical device
  // e.g. http://192.168.1.5:8000

  static Future<Map<String, dynamic>>
      predictPrice(
          Map<String, dynamic> data) async {
    try {
      final response = await http.post(
        Uri.parse('$baseUrl/predict'),
        headers: {
          'Content-Type':
              'application/json'
        },
        body: jsonEncode(data),
      );

      if (response.statusCode == 200) {
        return jsonDecode(response.body);
      } else {
        throw Exception(
            'Prediction failed: '
            '${response.statusCode}');
      }
    } catch (e) {
      throw Exception(
          'Connection error: $e');
    }
  }

  static Future<Map<String, dynamic>>
      getHealth() async {
    try {
      final response = await http.get(
        Uri.parse('$baseUrl/health'),
      );
      if (response.statusCode == 200) {
        return jsonDecode(response.body);
      }
      throw Exception('API not healthy');
    } catch (e) {
      throw Exception('API offline: $e');
    }
  }

  static Future<Map<String, dynamic>>
      getStats() async {
    try {
      final response = await http.get(
        Uri.parse('$baseUrl/stats'),
      );
      if (response.statusCode == 200) {
        return jsonDecode(response.body);
      }
      throw Exception('Stats failed');
    } catch (e) {
      throw Exception('Error: $e');
    }
  }
}