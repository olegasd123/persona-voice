import 'dart:convert';
import 'dart:typed_data';

import 'package:http/http.dart' as http;

import '../models/connection_settings.dart';
import '../models/lora_option.dart';
import '../models/persona.dart';
import '../models/session_options.dart';
import '../models/voice_option.dart';

/// What the token server hands back from `POST /token`: everything needed to join a room.
class JoinGrant {
  const JoinGrant({
    required this.url,
    required this.token,
    required this.room,
    required this.identity,
    required this.persona,
  });

  /// LiveKit server URL (e.g. `wss://...`) — the client connects here, not to the token server.
  final String url;
  final String token;
  final String room;
  final String identity;
  final String persona;

  factory JoinGrant.fromJson(Map<String, dynamic> json) => JoinGrant(
        url: json['url'] as String,
        token: json['token'] as String,
        room: json['room'] as String,
        identity: json['identity'] as String,
        persona: (json['persona'] as String?) ?? '',
      );
}

class TokenClientException implements Exception {
  TokenClientException(this.message);
  final String message;
  @override
  String toString() => message;
}

/// Thin HTTP client for the persona-voice token server (`server/token_server.py`).
class TokenClient {
  TokenClient(this.settings, {http.Client? httpClient})
      : _http = httpClient ?? http.Client();

  final ConnectionSettings settings;
  final http.Client _http;

  /// Bearer auth header (when a token is configured) — used by every request.
  Map<String, String> get _authHeaders => {
        if (settings.apiToken.trim().isNotEmpty)
          'Authorization': 'Bearer ${settings.apiToken.trim()}',
      };

  Map<String, String> get _jsonHeaders => {
        'Content-Type': 'application/json',
        ..._authHeaders,
      };

  Uri _uri(String path, [Map<String, String>? query]) {
    final base = settings.tokenServerUrl.trim().replaceAll(RegExp(r'/+$'), '');
    final uri = Uri.parse('$base$path');
    return (query == null || query.isEmpty) ? uri : uri.replace(queryParameters: query);
  }

  /// Fetch the selectable personas (curated + this user's custom ones) and the default id.
  /// Scoped by [ConnectionSettings.effectiveUser] so the caller's custom personas come back.
  Future<(List<Persona>, String)> fetchPersonas() async {
    final resp = await _http.get(
      _uri('/personas', {'user': settings.effectiveUser}),
      headers: _jsonHeaders,
    );
    final body = _decode(resp);
    final personas = (body['personas'] as List<dynamic>)
        .map((e) => Persona.fromJson(e as Map<String, dynamic>))
        .toList();
    return (personas, (body['default'] as String?) ?? '');
  }

  /// Fetch one persona's full body (`GET /personas/{id}`) to prefill the edit form — the list
  /// route returns only a picker summary.
  Future<PersonaDraft> fetchPersona(String id) async {
    final resp = await _http.get(
      _uri('/personas/${Uri.encodeComponent(id)}', {'user': settings.effectiveUser}),
      headers: _jsonHeaders,
    );
    final body = _decode(resp);
    return PersonaDraft.fromBody((body['persona'] as Map).cast<String, dynamic>());
  }

  /// Create a custom persona (`POST /personas`). Returns the stored persona's full body.
  Future<PersonaDraft> createPersona(PersonaDraft draft) async {
    final resp = await _http.post(
      _uri('/personas', {'user': settings.effectiveUser}),
      headers: _jsonHeaders,
      body: jsonEncode(draft.toJson()),
    );
    final body = _decode(resp);
    return PersonaDraft.fromBody((body['persona'] as Map).cast<String, dynamic>());
  }

  /// Replace one of the user's own personas (`PUT /personas/{id}`).
  Future<PersonaDraft> updatePersona(String id, PersonaDraft draft) async {
    final resp = await _http.put(
      _uri('/personas/${Uri.encodeComponent(id)}', {'user': settings.effectiveUser}),
      headers: _jsonHeaders,
      body: jsonEncode(draft.toJson()),
    );
    final body = _decode(resp);
    return PersonaDraft.fromBody((body['persona'] as Map).cast<String, dynamic>());
  }

  /// Delete one of the user's own personas (`DELETE /personas/{id}`).
  Future<void> deletePersona(String id) async {
    final resp = await _http.delete(
      _uri('/personas/${Uri.encodeComponent(id)}', {'user': settings.effectiveUser}),
      headers: _authHeaders,
    );
    _decode(resp); // throws on error; body is {"deleted": id}
  }

  /// Fetch the served LoRA adapters for the active backend (`GET /loras`) — empty on Mac.
  Future<LoraCatalog> fetchLoras() async {
    final resp = await _http.get(_uri('/loras'), headers: _jsonHeaders);
    return LoraCatalog.fromJson(_decode(resp));
  }

  /// Fetch the voice catalog for the active backend (`GET /voices`).
  Future<VoiceCatalog> fetchVoices() async {
    final resp = await _http.get(_uri('/voices'), headers: _jsonHeaders);
    return VoiceCatalog.fromJson(_decode(resp));
  }

  /// Enroll a clone from a wav [sample] under [name] (`POST /voices/clone`). The sample rides
  /// as the raw request body (the server is multipart-free); [authorized] affirms the caller
  /// may use the voice and [refText] is an optional transcript (auto-transcribed otherwise).
  /// Returns the new voice's catalog entry.
  Future<VoiceOption> cloneVoice(
    String name,
    Uint8List sample, {
    String? refText,
    required bool authorized,
  }) async {
    final uri = _uri('/voices/clone').replace(queryParameters: {
      'name': name.trim(),
      if (refText != null && refText.trim().isNotEmpty) 'text': refText.trim(),
      'authorized': authorized.toString(),
    });
    final resp = await _http.post(
      uri,
      headers: {'Content-Type': 'audio/wav', ..._authHeaders},
      body: sample,
    );
    return VoiceOption.fromJson(_decode(resp));
  }

  /// Remove a cloned voice (`DELETE /voices/clone/{name}`).
  Future<void> deleteVoice(String name) async {
    final resp = await _http.delete(
      _uri('/voices/clone/${Uri.encodeComponent(name.trim())}'),
      headers: _authHeaders,
    );
    _decode(resp); // throws on error; body is {"deleted": name}
  }

  /// Mint a join token for [persona], optionally with per-session [options]
  /// (voice / CEFR / demeanor) and a fixed [room].
  Future<JoinGrant> requestToken({
    required String persona,
    SessionOptions options = const SessionOptions(),
    String? room,
  }) async {
    final payload = <String, dynamic>{
      'persona': persona,
      if (settings.identity.trim().isNotEmpty) 'identity': settings.identity.trim(),
      // Scopes custom-persona resolution + memory in the agent (falls back to "default").
      'user': settings.effectiveUser,
      if (room != null && room.trim().isNotEmpty) 'room': room.trim(),
      ...options.toWireMap(),
    };
    final resp = await _http.post(
      _uri('/token'),
      headers: _jsonHeaders,
      body: jsonEncode(payload),
    );
    return JoinGrant.fromJson(_decode(resp));
  }

  Map<String, dynamic> _decode(http.Response resp) {
    Map<String, dynamic> body;
    try {
      body = jsonDecode(resp.body) as Map<String, dynamic>;
    } catch (_) {
      throw TokenClientException('HTTP ${resp.statusCode}: ${resp.body}');
    }
    if (resp.statusCode >= 400) {
      throw TokenClientException(
          '${body['error'] ?? 'request failed'} (HTTP ${resp.statusCode})');
    }
    return body;
  }

  void close() => _http.close();
}
