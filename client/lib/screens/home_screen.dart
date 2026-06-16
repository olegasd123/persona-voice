import 'package:flutter/material.dart';
import 'package:permission_handler/permission_handler.dart';

import '../models/connection_settings.dart';
import '../models/persona.dart';
import '../services/token_client.dart';
import '../services/voice_session.dart';
import 'call_screen.dart';

/// Connection settings + persona picker + "Connect". The entry screen.
class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  final _settings = ConnectionSettings();
  final _urlCtrl = TextEditingController();
  final _tokenCtrl = TextEditingController();
  final _identityCtrl = TextEditingController();

  List<Persona> _personas = [];
  String? _selectedPersona;
  bool _loadingPersonas = false;
  bool _connecting = false;
  String? _status;

  @override
  void initState() {
    super.initState();
    _restore();
  }

  Future<void> _restore() async {
    final s = await ConnectionSettings.load();
    setState(() {
      _settings
        ..tokenServerUrl = s.tokenServerUrl
        ..apiToken = s.apiToken
        ..identity = s.identity;
      _urlCtrl.text = s.tokenServerUrl;
      _tokenCtrl.text = s.apiToken;
      _identityCtrl.text = s.identity;
    });
  }

  void _syncSettings() {
    _settings
      ..tokenServerUrl = _urlCtrl.text
      ..apiToken = _tokenCtrl.text
      ..identity = _identityCtrl.text;
  }

  Future<void> _loadPersonas() async {
    _syncSettings();
    await _settings.save();
    setState(() {
      _loadingPersonas = true;
      _status = null;
    });
    final client = TokenClient(_settings);
    try {
      final (personas, defaultId) = await client.fetchPersonas();
      setState(() {
        _personas = personas;
        _selectedPersona = personas.any((p) => p.id == defaultId)
            ? defaultId
            : (personas.isNotEmpty ? personas.first.id : null);
      });
    } catch (e) {
      setState(() => _status = 'Could not load personas: $e');
    } finally {
      client.close();
      if (mounted) setState(() => _loadingPersonas = false);
    }
  }

  Future<void> _connect() async {
    if (_selectedPersona == null) {
      setState(() => _status = 'Load and pick a persona first.');
      return;
    }
    _syncSettings();
    await _settings.save();

    final mic = await Permission.microphone.request();
    if (!mic.isGranted) {
      setState(() => _status = 'Microphone permission is required to talk.');
      return;
    }

    setState(() {
      _connecting = true;
      _status = null;
    });
    final client = TokenClient(_settings);
    try {
      final grant = await client.requestToken(persona: _selectedPersona!);
      if (!mounted) return;
      final session = VoiceSession();
      await Navigator.of(context).push(MaterialPageRoute(
        builder: (_) => CallScreen(
          session: session,
          grant: grant,
          personas: _personas,
        ),
      ));
    } catch (e) {
      setState(() => _status = 'Could not connect: $e');
    } finally {
      client.close();
      if (mounted) setState(() => _connecting = false);
    }
  }

  @override
  void dispose() {
    _urlCtrl.dispose();
    _tokenCtrl.dispose();
    _identityCtrl.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Persona Voice')),
      body: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          TextField(
            controller: _urlCtrl,
            decoration: const InputDecoration(
              labelText: 'Token server URL',
              hintText: 'http://localhost:8080',
              border: OutlineInputBorder(),
            ),
            keyboardType: TextInputType.url,
          ),
          const SizedBox(height: 12),
          TextField(
            controller: _tokenCtrl,
            decoration: const InputDecoration(
              labelText: 'API token (optional)',
              border: OutlineInputBorder(),
            ),
            obscureText: true,
          ),
          const SizedBox(height: 12),
          TextField(
            controller: _identityCtrl,
            decoration: const InputDecoration(
              labelText: 'Your name / identity (optional)',
              border: OutlineInputBorder(),
            ),
          ),
          const SizedBox(height: 16),
          FilledButton.tonal(
            onPressed: _loadingPersonas ? null : _loadPersonas,
            child: _loadingPersonas
                ? const SizedBox(
                    height: 18, width: 18, child: CircularProgressIndicator(strokeWidth: 2))
                : const Text('Load personas'),
          ),
          const SizedBox(height: 8),
          if (_personas.isNotEmpty) ...[
            const Text('Persona', style: TextStyle(fontWeight: FontWeight.bold)),
            RadioGroup<String>(
              groupValue: _selectedPersona,
              onChanged: (v) => setState(() => _selectedPersona = v),
              child: Column(
                children: _personas
                    .map((p) => RadioListTile<String>(
                          value: p.id,
                          title: Text(p.name),
                          subtitle: Text(p.id),
                          dense: true,
                        ))
                    .toList(),
              ),
            ),
          ],
          const SizedBox(height: 16),
          FilledButton(
            onPressed: _connecting || _personas.isEmpty ? null : _connect,
            child: _connecting
                ? const SizedBox(
                    height: 18, width: 18, child: CircularProgressIndicator(strokeWidth: 2))
                : const Text('Connect & talk'),
          ),
          if (_status != null) ...[
            const SizedBox(height: 16),
            Text(_status!, style: TextStyle(color: Theme.of(context).colorScheme.error)),
          ],
        ],
      ),
    );
  }
}
