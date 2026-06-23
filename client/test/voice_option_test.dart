import 'package:flutter_test/flutter_test.dart';
import 'package:personavoice_client/models/voice_option.dart';

void main() {
  test('VoiceOption parses fields and defaults', () {
    final v = VoiceOption.fromJson({
      'id': 'my_voice',
      'name': 'My Voice',
      'kind': 'clone',
      'emotion': 'warm',
      'available': false,
      'reason': 'needs a cloning backend',
    });
    expect(v.id, 'my_voice');
    expect(v.name, 'My Voice');
    expect(v.isClone, isTrue);
    expect(v.kindLabel, 'Clone');
    expect(v.available, isFalse);
    expect(v.reason, 'needs a cloning backend');
  });

  test('VoiceOption falls back name to id and nulls blank fields', () {
    final v = VoiceOption.fromJson({'id': 'af_heart', 'kind': 'preset', 'emotion': '  '});
    expect(v.name, 'af_heart');
    expect(v.emotion, isNull);
    expect(v.available, isTrue); // defaults true when absent
    expect(v.isClone, isFalse);
  });

  test('VoiceCatalog splits available vs unavailable and looks up by id', () {
    final c = VoiceCatalog.fromJson({
      'voices': [
        {'id': 'af_heart', 'name': 'af_heart', 'kind': 'preset', 'available': true},
        {'id': 'my_voice', 'name': 'my_voice', 'kind': 'clone', 'available': false,
          'reason': 'no cloning backend'},
      ],
      'backend': 'mac',
      'tts': 'kokoro',
      'supports_cloning': false,
    });
    expect(c.voices.length, 2);
    expect(c.supportsCloning, isFalse);
    expect(c.tts, 'kokoro');
    expect(c.available.map((v) => v.id), ['af_heart']);
    expect(c.unavailable.single.id, 'my_voice');
    expect(c.byId('my_voice')?.reason, 'no cloning backend');
    expect(c.byId('ghost'), isNull);
    expect(c.byId(null), isNull);
  });

  test('empty catalog is well-formed', () {
    expect(VoiceCatalog.empty.voices, isEmpty);
    expect(VoiceCatalog.empty.supportsCloning, isFalse);
    expect(VoiceCatalog.fromJson({}).voices, isEmpty);
  });
}
