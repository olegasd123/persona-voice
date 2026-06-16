import 'dart:convert';

import 'package:http/http.dart' as http;

import '../models/connection_settings.dart';
import '../models/persona.dart';

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

  Map<String, String> get _headers => {
        'Content-Type': 'application/json',
        if (settings.apiToken.trim().isNotEmpty)
          'Authorization': 'Bearer ${settings.apiToken.trim()}',
      };

  Uri _uri(String path) {
    final base = settings.tokenServerUrl.trim().replaceAll(RegExp(r'/+$'), '');
    return Uri.parse('$base$path');
  }

  /// Fetch the selectable personas and the server's default id.
  Future<(List<Persona>, String)> fetchPersonas() async {
    final resp = await _http.get(_uri('/personas'), headers: _headers);
    final body = _decode(resp);
    final personas = (body['personas'] as List<dynamic>)
        .map((e) => Persona.fromJson(e as Map<String, dynamic>))
        .toList();
    return (personas, (body['default'] as String?) ?? '');
  }

  /// Mint a join token for [persona] (and optionally a fixed [room]).
  Future<JoinGrant> requestToken({required String persona, String? room}) async {
    final payload = <String, dynamic>{
      'persona': persona,
      if (settings.identity.trim().isNotEmpty) 'identity': settings.identity.trim(),
      if (room != null && room.trim().isNotEmpty) 'room': room.trim(),
    };
    final resp = await _http.post(
      _uri('/token'),
      headers: _headers,
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
