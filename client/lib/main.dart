import 'package:flutter/material.dart';

import 'screens/home_screen.dart';

void main() => runApp(const PersonaVoiceApp());

class PersonaVoiceApp extends StatelessWidget {
  const PersonaVoiceApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Persona Voice',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: const Color(0xFF6750A4)),
        useMaterial3: true,
      ),
      home: const HomeScreen(),
    );
  }
}
