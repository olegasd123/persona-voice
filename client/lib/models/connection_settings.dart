import 'dart:math';

import 'package:shared_preferences/shared_preferences.dart';

class ConnectionSettings {
  ConnectionSettings({
    this.tokenServerUrl = 'http://localhost:8080',
    this.apiToken = '',
    this.displayName = '',
    this.userId = '',
    this.participantId = '',
  });

  static const _tokenServerUrlKey = 'connection.tokenServerUrl';
  static const _apiTokenKey = 'connection.apiToken';
  static const _displayNameKey = 'connection.displayName';
  static const _userIdKey = 'connection.userId';
  static const _participantIdKey = 'connection.participantId';
  // Legacy single field that doubled as both display name and scope key (pre-split).
  static const _legacyIdentityKey = 'connection.identity';

  String tokenServerUrl;
  String apiToken;

  /// Cosmetic. Rides the `/token` body as `name` (the LiveKit display-name claim — the label
  /// the assistant addresses you by). Free text — editing it never changes which personas or
  /// memory you see, nor your participant identity.
  String displayName;

  /// The account selector. Scopes your custom personas and memory server-side: type the same
  /// id again — even on another device — to pick up where you left off. Switching is
  /// non-destructive (the old id's data stays put). Empty falls back to the shared
  /// [defaultUser] bucket. Normalized to the server's charset via [normalizeUserId] before
  /// it leaves the device.
  String userId;

  /// Stable, opaque LiveKit participant id — rides the `/token` body as `identity` (`sub`).
  /// Generated once per install and persisted, so it's unique and independent of both
  /// [displayName] (a mutable label that may collide) and [userId] (the data scope). Renaming
  /// yourself or switching accounts never changes it. Not user-facing.
  String participantId;

  /// The shared bucket for a user who hasn't claimed an account id.
  static const defaultUser = 'default';

  /// A fresh per-install participant id, e.g. `pv-3f9a…` (16 random bytes, hex).
  static String _generateParticipantId() {
    final rnd = Random.secure();
    final hex = List.generate(
      16,
      (_) => rnd.nextInt(256).toRadixString(16).padLeft(2, '0'),
    ).join();
    return 'pv-$hex';
  }

  /// The normalized scope key sent to the server (`user`), or [defaultUser] when unset.
  String get effectiveUser {
    final id = normalizeUserId(userId);
    return id.isNotEmpty ? id : defaultUser;
  }

  /// Coerce free text into a valid account id. The server accepts
  /// `[A-Za-z0-9][A-Za-z0-9._@-]{0,127}` (and uses it as a directory name for memory), so we
  /// map anything else to `-`, collapse repeats, drop leading non-alphanumerics, and cap the
  /// length. Case is preserved on purpose — lowercasing would silently merge a returning
  /// user's existing bucket into a different one. Returns `''` when nothing usable remains.
  static String normalizeUserId(String raw) {
    var s = raw.trim();
    if (s.isEmpty) return '';
    s = s.replaceAll(RegExp(r'[^A-Za-z0-9._@-]'), '-');
    s = s.replaceAll(RegExp(r'-{2,}'), '-');
    s = s.replaceFirst(RegExp(r'^[^A-Za-z0-9]+'), '');
    if (s.length > 128) s = s.substring(0, 128);
    return s;
  }

  static Future<ConnectionSettings> load() async {
    final prefs = await SharedPreferences.getInstance();
    // Migrate the legacy `connection.identity` field into the split fields. Preserve its exact
    // value as the userId so a returning user keeps the same personas/memory bucket; it was
    // already constrained to the valid charset (the old client couldn't have sent otherwise).
    final legacy = prefs.getString(_legacyIdentityKey);
    // Mint the stable participant id on first run and persist it so it never changes after.
    var participantId = prefs.getString(_participantIdKey) ?? '';
    if (participantId.isEmpty) {
      participantId = _generateParticipantId();
      await prefs.setString(_participantIdKey, participantId);
    }
    return ConnectionSettings(
      tokenServerUrl:
          prefs.getString(_tokenServerUrlKey) ?? 'http://localhost:8080',
      apiToken: prefs.getString(_apiTokenKey) ?? '',
      displayName: prefs.getString(_displayNameKey) ?? legacy ?? '',
      userId: prefs.getString(_userIdKey) ?? legacy ?? '',
      participantId: participantId,
    );
  }

  Future<void> save() async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(_tokenServerUrlKey, tokenServerUrl);
    await prefs.setString(_apiTokenKey, apiToken);
    await prefs.setString(_displayNameKey, displayName);
    await prefs.setString(_userIdKey, userId);
    if (participantId.isNotEmpty) {
      await prefs.setString(_participantIdKey, participantId);
    }
  }
}
