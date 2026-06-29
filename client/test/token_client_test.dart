import 'dart:convert';
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';
import 'package:personavoice_client/models/connection_settings.dart';
import 'package:personavoice_client/models/persona.dart';
import 'package:personavoice_client/models/session_options.dart';
import 'package:personavoice_client/services/token_client.dart';

ConnectionSettings _settings({String token = ''}) => ConnectionSettings(
      tokenServerUrl: 'http://localhost:8080/',
      apiToken: token,
      displayName: 'oleg',
      userId: 'oleg',
      participantId: 'pv-test',
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
      expect(body['identity'], 'pv-test'); // stable participant id, not the display name
      expect(body['name'], 'oleg'); // display name rides the separate `name` field
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
    expect(seen.toString(), 'http://localhost:8080/personas?user=oleg');
  });

  test('fetchPersonas scopes the list by the effective user', () async {
    late Uri seen;
    final mock = MockClient((req) async {
      seen = req.url;
      return http.Response(jsonEncode({'personas': [], 'default': ''}), 200);
    });
    // No identity set → falls back to the "default" bucket.
    await TokenClient(
      ConnectionSettings(tokenServerUrl: 'http://localhost:8080'),
      httpClient: mock,
    ).fetchPersonas();
    expect(seen.queryParameters['user'], 'default');
  });

  test('parses the custom flag on user-authored personas', () async {
    final mock = MockClient((req) async {
      return http.Response(
        jsonEncode({
          'personas': [
            {'id': 'companion', 'name': 'Companion', 'custom': false},
            {'id': 'french-tutor', 'name': 'French Tutor', 'custom': true},
          ],
          'default': 'companion',
        }),
        200,
      );
    });
    final (personas, _) = await TokenClient(_settings(), httpClient: mock).fetchPersonas();
    expect(personas[0].custom, isFalse);
    expect(personas[1].custom, isTrue);
  });

  test('requestToken sends the scoping user in the body', () async {
    final mock = MockClient((req) async {
      final body = jsonDecode(req.body) as Map<String, dynamic>;
      expect(body['user'], 'oleg');
      return http.Response(
        jsonEncode(
            {'url': 'wss://lk', 'token': 't', 'room': 'r', 'identity': 'oleg', 'persona': 'x'}),
        200,
      );
    });
    await TokenClient(_settings(), httpClient: mock).requestToken(persona: 'x');
  });

  test('requestToken keeps participant id, display name, and account id distinct', () async {
    late Map<String, dynamic> body;
    final mock = MockClient((req) async {
      body = jsonDecode(req.body) as Map<String, dynamic>;
      return http.Response(
        jsonEncode(
            {'url': 'wss://lk', 'token': 't', 'room': 'r', 'identity': 'x', 'persona': 'x'}),
        200,
      );
    });
    // A display name with a space (the old crash case) rides verbatim as `name`; the account
    // id is sanitized to the server's charset for `user`; the participant id is the stable
    // `identity`.
    final settings = ConnectionSettings(
      tokenServerUrl: 'http://localhost:8080',
      displayName: 'Oleg Smith',
      userId: 'Oleg Smith',
      participantId: 'pv-stable',
    );
    await TokenClient(settings, httpClient: mock).requestToken(persona: 'x');
    expect(body['identity'], 'pv-stable');
    expect(body['name'], 'Oleg Smith');
    expect(body['user'], 'Oleg-Smith');
  });

  test('omits identity/name when unset and falls back to the default bucket', () async {
    late Map<String, dynamic> body;
    final mock = MockClient((req) async {
      body = jsonDecode(req.body) as Map<String, dynamic>;
      return http.Response(
        jsonEncode(
            {'url': 'wss://lk', 'token': 't', 'room': 'r', 'identity': 'x', 'persona': 'x'}),
        200,
      );
    });
    // Nothing set → the server generates a participant id, and data pools in the shared bucket.
    final settings = ConnectionSettings(tokenServerUrl: 'http://localhost:8080');
    await TokenClient(settings, httpClient: mock).requestToken(persona: 'x');
    expect(body.containsKey('identity'), isFalse);
    expect(body.containsKey('name'), isFalse);
    expect(body['user'], 'default');
  });

  test('createPersona posts the draft and parses the stored body', () async {
    late http.Request seen;
    final mock = MockClient((req) async {
      seen = req;
      final body = jsonDecode(req.body) as Map<String, dynamic>;
      expect(body['name'], 'French Tutor');
      expect(body['system_prompt'], 'Teach French.');
      // Server owns the id — the draft must not send one on create.
      expect(body.containsKey('id'), isFalse);
      return http.Response(
        jsonEncode({
          'id': 'french-tutor',
          'name': 'French Tutor',
          'custom': true,
          'persona': {
            'id': 'french-tutor',
            'name': 'French Tutor',
            'system_prompt': 'Teach French.',
            'llm': {'base_model': 'qwen', 'lora': null},
            'voice': {'ref': 'voices/companion_soft', 'emotion': 'neutral'},
            'behavior': {'turn_style': 'balanced'},
            'memory': {'enabled': false},
            'session_defaults': {'cefr': 'A1'},
          },
        }),
        201,
      );
    });
    final draft = PersonaDraft(name: 'French Tutor', systemPrompt: 'Teach French.');
    final stored = await TokenClient(_settings(), httpClient: mock).createPersona(draft);
    expect(seen.method, 'POST');
    expect(seen.url.path, '/personas');
    expect(seen.url.queryParameters['user'], 'oleg');
    expect(stored.id, 'french-tutor');
    expect(stored.sessionDefaults.cefr, CefrLevel.a1);
  });

  test('updatePersona PUTs to the persona id', () async {
    late http.Request seen;
    final mock = MockClient((req) async {
      seen = req;
      return http.Response(
        jsonEncode({
          'id': 'french-tutor',
          'name': 'Spanish Tutor',
          'custom': true,
          'persona': {
            'id': 'french-tutor',
            'name': 'Spanish Tutor',
            'system_prompt': 'Teach Spanish.',
            'llm': {'base_model': 'qwen'},
            'voice': {'ref': 'voices/companion_soft'},
          },
        }),
        200,
      );
    });
    final draft = PersonaDraft(
      id: 'french-tutor',
      name: 'Spanish Tutor',
      systemPrompt: 'Teach Spanish.',
    );
    final updated = await TokenClient(_settings(), httpClient: mock)
        .updatePersona('french-tutor', draft);
    expect(seen.method, 'PUT');
    expect(seen.url.path, '/personas/french-tutor');
    expect(updated.name, 'Spanish Tutor');
  });

  test('deletePersona issues a DELETE to the persona id', () async {
    late http.Request seen;
    final mock = MockClient((req) async {
      seen = req;
      return http.Response(jsonEncode({'deleted': 'french-tutor'}), 200);
    });
    await TokenClient(_settings(), httpClient: mock).deletePersona('french-tutor');
    expect(seen.method, 'DELETE');
    expect(seen.url.path, '/personas/french-tutor');
    expect(seen.url.queryParameters['user'], 'oleg');
  });

  test('fetchPersona returns the full body for an edit form', () async {
    final mock = MockClient((req) async {
      expect(req.url.path, '/personas/french-tutor');
      return http.Response(
        jsonEncode({
          'id': 'french-tutor',
          'name': 'French Tutor',
          'custom': true,
          'persona': {
            'id': 'french-tutor',
            'name': 'French Tutor',
            'system_prompt': 'Teach French.',
            'llm': {'base_model': 'qwen', 'lora': 'adapters/fr'},
            'voice': {'ref': 'my_voice', 'emotion': 'neutral'},
            'behavior': {'turn_style': 'concise'},
            'memory': {'enabled': true},
            'session_defaults': {'demeanor': 'kind'},
          },
        }),
        200,
      );
    });
    final draft =
        await TokenClient(_settings(), httpClient: mock).fetchPersona('french-tutor');
    expect(draft.systemPrompt, 'Teach French.');
    expect(draft.voiceRef, 'my_voice');
    expect(draft.lora, 'adapters/fr');
    expect(draft.turnStyle, TurnStyle.concise);
    expect(draft.memoryEnabled, isTrue);
    expect(draft.sessionDefaults.demeanor, Demeanor.kind);
  });

  test('fetchLoras parses the served adapters and capability', () async {
    final mock = MockClient((req) async {
      expect(req.url.path, '/loras');
      return http.Response(
        jsonEncode({
          'loras': [
            {'id': 'hr', 'name': 'hr', 'available': true},
          ],
          'llm': 'vllm',
          'supports_lora': true,
          'reason': null,
        }),
        200,
      );
    });
    final cat = await TokenClient(_settings(), httpClient: mock).fetchLoras();
    expect(cat.supportsLora, isTrue);
    expect(cat.loras.single.id, 'hr');
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

  test('cloneVoice posts the raw wav body with name/authorized query params', () async {
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
        .cloneVoice('my_voice', sample, authorized: true);
    expect(seen.method, 'POST');
    expect(seen.url.path, '/voices/clone');
    expect(seen.url.queryParameters['name'], 'my_voice');
    expect(seen.url.queryParameters.containsKey('text'), isFalse);
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

  test('a 503 surfaces as busy with the retry_after hint (for the queue)', () async {
    final mock = MockClient((req) async {
      return http.Response(
        jsonEncode({'error': 'all sessions are busy', 'retry_after': 12}),
        503,
      );
    });
    try {
      await TokenClient(_settings(), httpClient: mock).requestToken(persona: 'x');
      fail('expected a TokenClientException');
    } on TokenClientException catch (e) {
      expect(e.isBusy, isTrue);
      expect(e.statusCode, 503);
      expect(e.retryAfter, 12);
    }
  });

  test('retry_after falls back to the Retry-After header', () async {
    final mock = MockClient((req) async {
      return http.Response(jsonEncode({'error': 'busy'}), 503,
          headers: {'retry-after': '15'});
    });
    try {
      await TokenClient(_settings(), httpClient: mock).requestToken(persona: 'x');
      fail('expected a TokenClientException');
    } on TokenClientException catch (e) {
      expect(e.retryAfter, 15);
    }
  });

  test('fetchMemory scopes by user and parses the profile + turns', () async {
    late Uri seen;
    final mock = MockClient((req) async {
      seen = req.url;
      return http.Response(
        jsonEncode({
          'user': 'oleg',
          'granted': true,
          'summary': 'Sam, a returning user.',
          'facts': [
            {'text': 'Prefers to be called Sam', 'ts': '2026-06-29T10:00:00+00:00'},
          ],
          'turns': [
            {'role': 'user', 'content': "I'm Sam.", 'persona_id': 'companion'},
            {'role': 'assistant', 'content': 'Hi Sam!', 'persona_id': 'companion'},
          ],
          'updated_at': '2026-06-29T10:00:00+00:00',
        }),
        200,
      );
    });
    final memory = await TokenClient(_settings(), httpClient: mock).fetchMemory();
    expect(seen.path, '/memory');
    expect(seen.queryParameters['user'], 'oleg');
    expect(memory.granted, isTrue);
    expect(memory.isEmpty, isFalse);
    expect(memory.summary, 'Sam, a returning user.');
    expect(memory.facts.single.text, 'Prefers to be called Sam');
    expect(memory.turns.map((t) => t.content), ["I'm Sam.", 'Hi Sam!']);
    expect(memory.turns.first.isUser, isTrue);
    expect(memory.turns.last.isUser, isFalse);
  });

  test('fetchMemory treats a blank server payload as empty', () async {
    final mock = MockClient((req) async {
      return http.Response(
        jsonEncode({
          'user': 'oleg',
          'granted': false,
          'summary': '',
          'facts': [],
          'turns': [],
          'updated_at': null,
        }),
        200,
      );
    });
    final memory = await TokenClient(_settings(), httpClient: mock).fetchMemory();
    expect(memory.isEmpty, isTrue);
    expect(memory.granted, isFalse);
    expect(memory.updatedAt, isNull);
  });

  test('fetchMemory passes the limit through as a query param', () async {
    late Uri seen;
    final mock = MockClient((req) async {
      seen = req.url;
      return http.Response(
        jsonEncode({'granted': true, 'summary': '', 'facts': [], 'turns': []}),
        200,
      );
    });
    await TokenClient(_settings(), httpClient: mock).fetchMemory(limit: 5);
    expect(seen.queryParameters['limit'], '5');
  });

  test('deleteMemory DELETEs and returns the deleted flag', () async {
    late http.Request seen;
    final mock = MockClient((req) async {
      seen = req;
      return http.Response(jsonEncode({'user': 'oleg', 'deleted': true}), 200);
    });
    final deleted = await TokenClient(_settings(), httpClient: mock).deleteMemory();
    expect(seen.method, 'DELETE');
    expect(seen.url.path, '/memory');
    expect(seen.url.queryParameters['user'], 'oleg');
    expect(deleted, isTrue);
  });

  test('friendlyTokenError maps statuses to user-facing copy', () {
    expect(friendlyTokenError(TokenClientException('x', statusCode: 503)), contains('busy'));
    expect(friendlyTokenError(TokenClientException('x', statusCode: 401)),
        contains('authorized'));
    // A transport/parse failure (no status) hides the raw message.
    expect(friendlyTokenError(TokenClientException('socket boom')),
        contains('Could not reach'));
    // A non-TokenClient error gets a generic line.
    expect(friendlyTokenError(StateError('boom')), isNotEmpty);
  });
}
