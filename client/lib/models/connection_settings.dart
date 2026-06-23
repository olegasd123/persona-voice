import 'package:shared_preferences/shared_preferences.dart';

class ConnectionSettings {
  ConnectionSettings({
    this.tokenServerUrl = 'http://localhost:8080',
    this.apiToken = '',
    this.identity = '',
  });

  static const _tokenServerUrlKey = 'connection.tokenServerUrl';
  static const _apiTokenKey = 'connection.apiToken';
  static const _identityKey = 'connection.identity';

  String tokenServerUrl;
  String apiToken;
  String identity;

  /// The id that scopes custom personas (and memory). Falls back to `"default"` when no
  /// identity is set, so a single-user device still has a stable, shareable persona bucket.
  /// The same value rides the `/token` body so a custom persona resolves at call time.
  static const defaultUser = 'default';
  String get effectiveUser =>
      identity.trim().isNotEmpty ? identity.trim() : defaultUser;

  static Future<ConnectionSettings> load() async {
    final prefs = await SharedPreferences.getInstance();
    return ConnectionSettings(
      tokenServerUrl:
          prefs.getString(_tokenServerUrlKey) ?? 'http://localhost:8080',
      apiToken: prefs.getString(_apiTokenKey) ?? '',
      identity: prefs.getString(_identityKey) ?? '',
    );
  }

  Future<void> save() async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(_tokenServerUrlKey, tokenServerUrl);
    await prefs.setString(_apiTokenKey, apiToken);
    await prefs.setString(_identityKey, identity);
  }
}
