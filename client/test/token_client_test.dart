import 'dart:convert';
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';
import 'package:personavoice_client/models/connection_settings.dart';
import 'package:personavoice_client/models/session_options.dart';
import 'package:personavoice_client/services/token_client.dart';

ConnectionSettings _settings({String token = ''}) => ConnectionSettings(
      tokenServerUrl: 'http://localhost:8080/',
      apiToken: token,
      identity: 'oleg',
    );

String? _contentType(http.BaseRequest req) {
  for (final e in req.headers.entries) {
    if (e.key.toLowerCase() == 'content-type') return e.value;
  }
  return null;
}

void main() {
  test('fetchPersonas parses the list and default', () async {
    final mock = MockClient((req) async {
      expect(req.url.path, '/personas');
      return http.Response(
        jsonEncode({
          'personas': [
            {
              'id': 'companion',
              'name': 'Companion',
              'description': 'A warm, attentive friend.',
              'voice': 'warm, soft, feminine',
            },
            {'id': 'hr_interviewer', 'name': 'HR Interviewer'},
          ],
          'default': 'companion',
        }),
        200,
      );
    });
    final (personas, defaultId) =
        await TokenClient(_settings(), httpClient: mock).fetchPersonas();
    expect(personas.map((p) => p.id), ['companion', 'hr_interviewer']);
    expect(defaultId, 'companion');
    // Rich fields parse when present…
    expect(personas.first.description, 'A warm, attentive friend.');
    expect(personas.first.voice, 'warm, soft, feminine');
    // …and default to empty when the server omits them (older server).
    expect(personas[1].description, '');
    expect(personas[1].voice, '');
  });

  test('requestToken posts persona + identity and parses the grant', () async {
    final mock = MockClient((req) async {
      expect(req.method, 'POST');
      final body = jsonDecode(req.body) as Map<String, dynamic>;
      expect(body['persona'], 'hr_interviewer');
      expect(body['identity'], 'oleg');
      return http.Response(
        jsonEncode({
          'url': 'wss://lk.local:7880',
          'token': 'jwt.abc.def',
          'room': 'pv-123',
          'identity': 'oleg',
          'persona': 'hr_interviewer',
        }),
        200,
      );
    });
    final grant = await TokenClient(_settings(), httpClient: mock)
        .requestToken(persona: 'hr_interviewer');
    expect(grant.url, 'wss://lk.local:7880');
    expect(grant.token, 'jwt.abc.def');
    expect(grant.room, 'pv-123');
    expect(grant.persona, 'hr_interviewer');
  });

  test('sends a bearer header when an API token is set', () async {
    var sawAuth = false;
    final mock = MockClient((req) async {
      sawAuth = req.headers['Authorization'] == 'Bearer sekret';
      return http.Response(jsonEncode({'personas': [], 'default': ''}), 200);
    });
    await TokenClient(_settings(token: 'sekret'), httpClient: mock).fetchPersonas();
    expect(sawAuth, isTrue);
  });

  test('omits the bearer header when no API token is set', () async {
    var hadAuth = true;
    final mock = MockClient((req) async {
      hadAuth = req.headers.containsKey('Authorization');
      return http.Response(jsonEncode({'personas': [], 'default': ''}), 200);
    });
    await TokenClient(_settings(), httpClient: mock).fetchPersonas();
    expect(hadAuth, isFalse);
  });

  test('surfaces the server error message on 4xx', () async {
    final mock = MockClient((req) async {
      return http.Response(jsonEncode({'error': 'unknown persona'}), 400);
    });
    expect(
      () => TokenClient(_settings(), httpClient: mock).requestToken(persona: 'ghost'),
      throwsA(isA<TokenClientException>()
          .having((e) => e.message, 'message', contains('unknown persona'))),
    );
  });

  test('normalizes a trailing slash in the base URL', () async {
    late Uri seen;
    final mock = MockClient((req) async {
      seen = req.url;
      return http.Response(jsonEncode({'personas': [], 'default': ''}), 200);
    });
    await TokenClient(_settings(), httpClient: mock).fetchPersonas();
    expect(seen.toString(), 'http://localhost:8080/personas');
  });

  test('requestToken includes set session options in the body', () async {
    final mock = MockClient((req) async {
      final body = jsonDecode(req.body) as Map<String, dynamic>;
      expect(body['persona'], 'language_teacher');
      expect(body['voice'], 'my_voice');
      expect(body['cefr'], 'B1');
      expect(body['demeanor'], 'kind');
      return http.Response(
        jsonEncode({
          'url': 'wss://lk',
          'token': 't',
          'room': 'r',
          'identity': 'oleg',
          'persona': 'language_teacher',
        }),
        200,
      );
    });
    await TokenClient(_settings(), httpClient: mock).requestToken(
      persona: 'language_teacher',
      options: const SessionOptions(
        voice: 'my_voice',
        cefr: CefrLevel.b1,
        demeanor: Demeanor.kind,
      ),
    );
  });

  test('requestToken omits option keys when none are set', () async {
    final mock = MockClient((req) async {
      final body = jsonDecode(req.body) as Map<String, dynamic>;
      expect(body.containsKey('voice'), isFalse);
      expect(body.containsKey('cefr'), isFalse);
      expect(body.containsKey('demeanor'), isFalse);
      return http.Response(
        jsonEncode(
            {'url': 'wss://lk', 'token': 't', 'room': 'r', 'identity': 'oleg', 'persona': 'x'}),
        200,
      );
    });
    await TokenClient(_settings(), httpClient: mock).requestToken(persona: 'x');
  });

  test('fetchVoices parses the catalog and capability flags', () async {
    final mock = MockClient((req) async {
      expect(req.url.path, '/voices');
      return http.Response(
        jsonEncode({
          'voices': [
            {'id': 'af_heart', 'name': 'af_heart', 'kind': 'preset', 'available': true},
            {
              'id': 'my_voice',
              'name': 'my_voice',
              'kind': 'clone',
              'available': false,
              'reason': 'needs a cloning backend',
            },
          ],
          'backend': 'mac',
          'tts': 'kokoro',
          'supports_cloning': false,
        }),
        200,
      );
    });
    final catalog = await TokenClient(_settings(), httpClient: mock).fetchVoices();
    expect(catalog.voices.length, 2);
    expect(catalog.supportsCloning, isFalse);
    expect(catalog.tts, 'kokoro');
    expect(catalog.available.map((v) => v.id), ['af_heart']);
    expect(catalog.unavailable.single.reason, 'needs a cloning backend');
  });

  test('cloneVoice posts the raw wav body with name/text/authorized query params', () async {
    final sample = Uint8List.fromList([82, 73, 70, 70, 1, 2, 3, 4]); // "RIFF"…
    late http.Request seen;
    final mock = MockClient((req) async {
      seen = req;
      return http.Response(
        jsonEncode({'id': 'my_voice', 'name': 'my_voice', 'kind': 'clone', 'available': true}),
        201,
      );
    });
    final option = await TokenClient(_settings(), httpClient: mock)
        .cloneVoice('my_voice', sample, refText: 'hello there', authorized: true);
    expect(seen.method, 'POST');
    expect(seen.url.path, '/voices/clone');
    expect(seen.url.queryParameters['name'], 'my_voice');
    expect(seen.url.queryParameters['text'], 'hello there');
    expect(seen.url.queryParameters['authorized'], 'true');
    expect(seen.bodyBytes, sample);
    expect(_contentType(seen), 'audio/wav');
    expect(option.id, 'my_voice');
    expect(option.isClone, isTrue);
  });

  test('deleteVoice issues a DELETE to the named clone', () async {
    late http.Request seen;
    final mock = MockClient((req) async {
      seen = req;
      return http.Response(jsonEncode({'deleted': 'my_voice'}), 200);
    });
    await TokenClient(_settings(), httpClient: mock).deleteVoice('my_voice');
    expect(seen.method, 'DELETE');
    expect(seen.url.path, '/voices/clone/my_voice');
  });

  test('cloneVoice surfaces a server rejection', () async {
    final mock = MockClient((req) async {
      return http.Response(jsonEncode({'error': 'clone quota reached'}), 400);
    });
    expect(
      () => TokenClient(_settings(), httpClient: mock)
          .cloneVoice('x', Uint8List(4), authorized: true),
      throwsA(isA<TokenClientException>()
          .having((e) => e.message, 'message', contains('clone quota reached'))),
    );
  });
}
