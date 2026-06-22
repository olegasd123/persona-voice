import 'package:flutter/material.dart';

import 'models/app_preferences.dart';
import 'screens/home_screen.dart';

void main() => runApp(const PersonaVoiceApp());

class PersonaVoiceApp extends StatefulWidget {
  const PersonaVoiceApp({super.key});

  @override
  State<PersonaVoiceApp> createState() => _PersonaVoiceAppState();
}

class _PersonaVoiceAppState extends State<PersonaVoiceApp> {
  // Owned at the root so a change in Settings (theme / default mic mode) rebuilds the whole
  // app and is persisted in one place.
  AppPreferences _prefs = AppPreferences();

  @override
  void initState() {
    super.initState();
    AppPreferences.load().then((p) {
      if (mounted) setState(() => _prefs = p);
    });
  }

  void _onPrefsChanged() {
    setState(() {});
    _prefs.save();
  }

  @override
  Widget build(BuildContext context) {
    const seed = Color(0xFF6750A4);
    return MaterialApp(
      title: 'Persona Voice',
      debugShowCheckedModeBanner: false,
      themeMode: _prefs.themeMode,
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: seed),
        useMaterial3: true,
      ),
      darkTheme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: seed, brightness: Brightness.dark),
        useMaterial3: true,
      ),
      home: HomeScreen(prefs: _prefs, onPrefsChanged: _onPrefsChanged),
    );
  }
}
