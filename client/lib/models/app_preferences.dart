import 'dart:convert';

import 'package:flutter/material.dart';
import 'package:shared_preferences/shared_preferences.dart';

import '../services/voice_session.dart' show MicMode;
import 'session_options.dart';

/// App-level preferences that aren't connection details: how the call opens (mic mode),
/// the theme, which persona was last called (so the home shelf can surface it), and the
/// per-persona session overrides (voice / CEFR / demeanor) the user has chosen. Kept
/// separate from [ConnectionSettings] because these are tastes, not server credentials.
class AppPreferences {
  AppPreferences({
    this.defaultMicMode = MicMode.openMic,
    this.themeMode = ThemeMode.system,
    this.lastPersonaId = '',
    Map<String, SessionOptions>? personaOptions,
  }) : personaOptions = personaOptions ?? {};

  static const _micModeKey = 'prefs.defaultMicMode';
  static const _themeModeKey = 'prefs.themeMode';
  static const _lastPersonaKey = 'prefs.lastPersonaId';
  static const _personaOptionsKey = 'prefs.personaOptions';

  MicMode defaultMicMode;
  ThemeMode themeMode;
  String lastPersonaId;

  /// Per-persona session overrides, keyed by persona id. A persona with no entry (or an empty
  /// [SessionOptions]) runs exactly as authored.
  Map<String, SessionOptions> personaOptions;

  /// The overrides chosen for [personaId] (empty options when none).
  SessionOptions optionsFor(String personaId) =>
      personaOptions[personaId] ?? const SessionOptions();

  /// Persist the overrides for [personaId]. Empty options drop the entry so a "reset to
  /// default" leaves nothing behind.
  Future<void> setOptionsFor(String personaId, SessionOptions options) async {
    if (options.isEmpty) {
      personaOptions.remove(personaId);
    } else {
      personaOptions[personaId] = options;
    }
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(_personaOptionsKey, _encodePersonaOptions(personaOptions));
  }

  static Future<AppPreferences> load() async {
    final prefs = await SharedPreferences.getInstance();
    return AppPreferences(
      defaultMicMode: _micModeFromName(prefs.getString(_micModeKey)),
      themeMode: _themeFromName(prefs.getString(_themeModeKey)),
      lastPersonaId: prefs.getString(_lastPersonaKey) ?? '',
      personaOptions: _decodePersonaOptions(prefs.getString(_personaOptionsKey)),
    );
  }

  Future<void> save() async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(_micModeKey, defaultMicMode.name);
    await prefs.setString(_themeModeKey, themeMode.name);
    await prefs.setString(_lastPersonaKey, lastPersonaId);
    await prefs.setString(_personaOptionsKey, _encodePersonaOptions(personaOptions));
  }

  /// Persist just the last-called persona (called when a call starts).
  static Future<void> rememberPersona(String id) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(_lastPersonaKey, id);
  }

  static String _encodePersonaOptions(Map<String, SessionOptions> options) =>
      jsonEncode(options.map((id, o) => MapEntry(id, o.toJson())));

  static Map<String, SessionOptions> _decodePersonaOptions(String? stored) {
    if (stored == null || stored.trim().isEmpty) return {};
    try {
      final raw = jsonDecode(stored);
      if (raw is! Map) return {};
      final out = <String, SessionOptions>{};
      raw.forEach((id, value) {
        if (id is String && value is Map<String, dynamic>) {
          final opts = SessionOptions.fromJson(value);
          if (opts.isNotEmpty) out[id] = opts;
        }
      });
      return out;
    } catch (_) {
      return {};
    }
  }

  static MicMode _micModeFromName(String? name) =>
      MicMode.values.firstWhere((m) => m.name == name, orElse: () => MicMode.openMic);

  static ThemeMode _themeFromName(String? name) =>
      ThemeMode.values.firstWhere((m) => m.name == name, orElse: () => ThemeMode.system);
}
