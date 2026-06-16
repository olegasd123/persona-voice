import 'dart:math';

import 'package:flutter_test/flutter_test.dart';
import 'package:personavoice_client/services/telephony.dart';

void main() {
  group('parseCallControlEvent', () {
    test('decodes endCall', () {
      expect(parseCallControlEvent({'event': 'endCall'}), const EndCallRequested());
    });

    test('decodes setMuted with the muted flag', () {
      expect(
        parseCallControlEvent({'event': 'setMuted', 'muted': true}),
        const MuteRequested(muted: true),
      );
      // Missing / non-true muted is treated as false.
      expect(
        parseCallControlEvent({'event': 'setMuted'}),
        const MuteRequested(muted: false),
      );
    });

    test('returns null for unmodeled / malformed payloads', () {
      expect(parseCallControlEvent({'event': 'somethingNew'}), isNull);
      expect(parseCallControlEvent('not a map'), isNull);
      expect(parseCallControlEvent(null), isNull);
    });
  });

  group('generateCallId', () {
    final uuidV4 = RegExp(
      r'^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$',
    );

    test('produces a valid UUID-v4 string', () {
      // Seeded Random keeps it deterministic; the format (version 4, RFC 4122 variant) must hold
      // regardless so iOS `UUID(uuidString:)` accepts it.
      for (var seed = 0; seed < 50; seed++) {
        expect(generateCallId(Random(seed)), matches(uuidV4));
      }
    });

    test('is effectively unique across calls', () {
      final ids = {for (var i = 0; i < 200; i++) generateCallId()};
      expect(ids.length, 200);
    });
  });
}
