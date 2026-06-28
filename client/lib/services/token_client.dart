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

/// A user's server-side recording consent (`GET`/`POST /consent`). [granted] gates whether the
/// assistant stores anything between calls at all; [allowTraining] is the stricter, separate
/// opt-in for letting transcripts feed a training distill.
class ConsentState {
  const ConsentState({required this.granted, required this.allowTraining});

  final bool granted;
  final bool allowTraining;

  factory ConsentState.fromJson(Map<String, dynamic> json) => ConsentState(
        granted: (json['granted'] as bool?) ?? false,
        allowTraining: (json['allow_training'] as bool?) ?? false,
      );
}

class TokenClientException implements Exception {
  TokenClientException(this.message, {this.statusCode, this.retryAfter});

  final String message;

  /// The HTTP status, when the failure was an HTTP response (null for a transport/parse error).
  /// `503` is the admission-control "all sessions busy" signal the queue waits on.
  final int? statusCode;

  /// Seconds the server suggests waiting before retrying (`retry_after` body field / `Retry-After`
  /// header), present on a `503`. Null when the server didn't advise one.
  final int? retryAfter;

  /// Whether this is the "all sessions are busy" response — the cue to queue rather than fail.
  bool get isBusy => statusCode == 503;

  @override
  String toString() => message;
}

/// A short, user-facing line for a token request failure (the raw server text / exception type is
/// for logs, not a SnackBar). Keep it actionable and free of HTTP/jargon.
String friendlyTokenError(Object error) {
  if (error is TokenClientException) {
    switch (error.statusCode) {
      case 401:
      case 403:
        return 'Not authorized — check the API token in Settings.';
      case 503:
        return 'The assistant is busy right now. Please try again shortly.';
      case null:
        // Transport / parse failure (server down, bad URL): the raw message is usually noise.
        return 'Could not reach the server. Check your connection and the address in Settings.';
      default:
        return error.message;
    }
  }
  return 'Something went wrong. Please try again.';
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
  /// may use the voice. Returns the new voice's catalog entry.
  Future<VoiceOption> cloneVoice(
    String name,
    Uint8List sample, {
    required bool authorized,
  }) async {
    final uri = _uri('/voices/clone').replace(queryParameters: {
      'name': name.trim(),
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

  /// Fetch the current recording-consent state for this account (`GET /consent`). Scoped by
  /// [ConnectionSettings.effectiveUser] — consent is per account id.
  Future<ConsentState> fetchConsent() async {
    final resp = await _http.get(
      _uri('/consent', {'user': settings.effectiveUser}),
      headers: _jsonHeaders,
    );
    return ConsentState.fromJson(_decode(resp));
  }

  /// Grant or withdraw recording consent for this account (`POST /consent`). Withdrawing keeps
  /// any already-stored data on the server (erase it from the memory tools). [allowTraining] is
  /// the separate training opt-in; the server forces it off when [granted] is false.
  Future<ConsentState> setConsent({
    required bool granted,
    bool allowTraining = false,
  }) async {
    final resp = await _http.post(
      _uri('/consent', {'user': settings.effectiveUser}),
      headers: _jsonHeaders,
      body: jsonEncode({'granted': granted, 'allow_training': allowTraining}),
    );
    return ConsentState.fromJson(_decode(resp));
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
      // The stable, opaque LiveKit participant id (`sub`); omitted when unset so the server
      // generates a throwaway one.
      if (settings.participantId.trim().isNotEmpty)
        'identity': settings.participantId.trim(),
      // The cosmetic display name (LiveKit `name` claim); omitted when blank.
      if (settings.displayName.trim().isNotEmpty) 'name': settings.displayName.trim(),
      // The account selector — scopes custom-persona resolution + memory in the agent
      // (normalized to the server's charset; falls back to "default").
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
      throw TokenClientException('HTTP ${resp.statusCode}: ${resp.body}',
          statusCode: resp.statusCode);
    }
    if (resp.statusCode >= 400) {
      throw TokenClientException(
        '${body['error'] ?? 'request failed'} (HTTP ${resp.statusCode})',
        statusCode: resp.statusCode,
        retryAfter: _retryAfter(body, resp.headers),
      );
    }
    return body;
  }

  /// The retry hint from the `retry_after` body field, falling back to the `Retry-After` header.
  static int? _retryAfter(Map<String, dynamic> body, Map<String, String> headers) {
    final fromBody = body['retry_after'];
    if (fromBody is num) return fromBody.toInt();
    final fromHeader = headers['retry-after'];
    if (fromHeader != null) return int.tryParse(fromHeader.trim());
    return null;
  }

  void close() => _http.close();
}
