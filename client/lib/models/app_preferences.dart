import 'package:flutter/material.dart';
import 'package:shared_preferences/shared_preferences.dart';

import '../services/voice_session.dart' show MicMode;

/// App-level preferences that aren't connection details: how the call opens (mic mode),
/// the theme, and which persona was last called (so the home shelf can surface it). Kept
/// separate from [ConnectionSettings] because these are tastes, not server credentials.
class AppPreferences {
  AppPreferences({
    this.defaultMicMode = MicMode.openMic,
    this.themeMode = ThemeMode.system,
    this.lastPersonaId = '',
  });

  static const _micModeKey = 'prefs.defaultMicMode';
  static const _themeModeKey = 'prefs.themeMode';
  static const _lastPersonaKey = 'prefs.lastPersonaId';

  MicMode defaultMicMode;
  ThemeMode themeMode;
  String lastPersonaId;

  static Future<AppPreferences> load() async {
    final prefs = await SharedPreferences.getInstance();
    return AppPreferences(
      defaultMicMode: _micModeFromName(prefs.getString(_micModeKey)),
      themeMode: _themeFromName(prefs.getString(_themeModeKey)),
      lastPersonaId: prefs.getString(_lastPersonaKey) ?? '',
    );
  }

  Future<void> save() async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(_micModeKey, defaultMicMode.name);
    await prefs.setString(_themeModeKey, themeMode.name);
    await prefs.setString(_lastPersonaKey, lastPersonaId);
  }

  /// Persist just the last-called persona (called when a call starts).
  static Future<void> rememberPersona(String id) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(_lastPersonaKey, id);
  }

  static MicMode _micModeFromName(String? name) =>
      MicMode.values.firstWhere((m) => m.name == name, orElse: () => MicMode.openMic);

  static ThemeMode _themeFromName(String? name) =>
      ThemeMode.values.firstWhere((m) => m.name == name, orElse: () => ThemeMode.system);
}
