import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:autopricer/main.dart';

void main() {
  testWidgets('AutoPricerAI smoke test', (WidgetTester tester) async {
    await tester.pumpWidget(const AutoPricerApp());
    expect(find.byType(MaterialApp), findsOneWidget);
  });
}
