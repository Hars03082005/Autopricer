import 'package:flutter_test/flutter_test.dart';

import 'package:PriceRef_mobile/main.dart';

void main() {
  testWidgets('PriceRef app loads', (WidgetTester tester) async {
    await tester.pumpWidget(const PriceRefApp());
    expect(find.text('PriceRef'), findsOneWidget);
  });
}
