import 'package:flutter_test/flutter_test.dart';
import 'package:personavoice_client/services/audio_session.dart';

void main() {
  group('parseAudioRoute', () {
    test('maps known route names', () {
      expect(parseAudioRoute('speaker'), AudioRoute.speaker);
      expect(parseAudioRoute('receiver'), AudioRoute.receiver);
      expect(parseAudioRoute('headphones'), AudioRoute.headphones);
      expect(parseAudioRoute('bluetooth'), AudioRoute.bluetooth);
      expect(parseAudioRoute('carAudio'), AudioRoute.carAudio);
    });

    test('falls back to unknown for unrecognized / null', () {
      expect(parseAudioRoute('hdmi'), AudioRoute.unknown);
      expect(parseAudioRoute(null), AudioRoute.unknown);
    });
  });

  group('parseAudioEvent', () {
    test('decodes interruptionBegan', () {
      expect(parseAudioEvent({'event': 'interruptionBegan'}), const InterruptionBegan());
    });

    test('decodes interruptionEnded with shouldResume', () {
      expect(
        parseAudioEvent({'event': 'interruptionEnded', 'shouldResume': true}),
        const InterruptionEnded(shouldResume: true),
      );
      // Missing / non-true shouldResume is treated as false.
      expect(
        parseAudioEvent({'event': 'interruptionEnded'}),
        const InterruptionEnded(shouldResume: false),
      );
    });

    test('decodes routeChanged', () {
      expect(
        parseAudioEvent({'event': 'routeChanged', 'route': 'bluetooth'}),
        const RouteChanged(AudioRoute.bluetooth),
      );
    });

    test('returns null for unmodeled / malformed payloads', () {
      expect(parseAudioEvent({'event': 'somethingNew'}), isNull);
      expect(parseAudioEvent('not a map'), isNull);
      expect(parseAudioEvent(null), isNull);
    });
  });
}
