import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:personavoice_client/models/app_preferences.dart';
import 'package:personavoice_client/models/session_options.dart';
import 'package:personavoice_client/services/voice_session.dart' show MicMode;
import 'package:shared_preferences/shared_preferences.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  setUp(() => SharedPreferences.setMockInitialValues({}));

  test('defaults are sensible when nothing is stored', () async {
    final prefs = await AppPreferences.load();
    expect(prefs.defaultMicMode, MicMode.openMic);
    expect(prefs.themeMode, ThemeMode.system);
    expect(prefs.lastPersonaId, '');
  });

  test('round-trips through save/load', () async {
    final prefs = await AppPreferences.load()
      ..defaultMicMode = MicMode.pushToTalk
      ..themeMode = ThemeMode.dark
      ..lastPersonaId = 'companion';
    await prefs.save();

    final restored = await AppPreferences.load();
    expect(restored.defaultMicMode, MicMode.pushToTalk);
    expect(restored.themeMode, ThemeMode.dark);
    expect(restored.lastPersonaId, 'companion');
  });

  test('rememberPersona persists just the last persona', () async {
    await AppPreferences.rememberPersona('tutor');
    expect((await AppPreferences.load()).lastPersonaId, 'tutor');
  });

  test('tolerates unknown stored enum names', () async {
    SharedPreferences.setMockInitialValues({
      'prefs.defaultMicMode': 'bogus',
      'prefs.themeMode': 'nonsense',
    });
    final prefs = await AppPreferences.load();
    expect(prefs.defaultMicMode, MicMode.openMic);
    expect(prefs.themeMode, ThemeMode.system);
  });

  test('per-persona options default to empty and persist', () async {
    final prefs = await AppPreferences.load();
    expect(prefs.optionsFor('companion').isEmpty, isTrue);

    await prefs.setOptionsFor(
      'companion',
      const SessionOptions(voice: 'my_voice', cefr: CefrLevel.b1),
    );
    final restored = await AppPreferences.load();
    expect(restored.optionsFor('companion'),
        const SessionOptions(voice: 'my_voice', cefr: CefrLevel.b1));
    // Personas with no entry are still untouched.
    expect(restored.optionsFor('hr_interviewer').isEmpty, isTrue);
  });

  test('setting empty options clears a persona entry', () async {
    final prefs = await AppPreferences.load();
    await prefs.setOptionsFor('companion', const SessionOptions(demeanor: Demeanor.rude));
    expect((await AppPreferences.load()).optionsFor('companion').isNotEmpty, isTrue);

    await prefs.setOptionsFor('companion', const SessionOptions());
    expect((await AppPreferences.load()).optionsFor('companion').isEmpty, isTrue);
  });
}
