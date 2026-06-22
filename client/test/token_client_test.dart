import 'dart:convert';

import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';
import 'package:personavoice_client/models/connection_settings.dart';
import 'package:personavoice_client/services/token_client.dart';

ConnectionSettings _settings({String token = ''}) => ConnectionSettings(
      tokenServerUrl: 'http://localhost:8080/',
      apiToken: token,
      identity: 'oleg',
    );

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
}
