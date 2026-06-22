import 'package:flutter/material.dart';

import '../models/app_preferences.dart';
import '../models/connection_settings.dart';
import '../services/token_client.dart';
import '../services/voice_session.dart' show MicMode;

/// Connection details (server URL, API token, your name) + app defaults (mic mode, theme).
/// This is where the config that used to clutter the home screen now lives — you only come
/// here on first run or when something needs changing.
class SettingsScreen extends StatefulWidget {
  const SettingsScreen({
    super.key,
    required this.prefs,
    required this.onPrefsChanged,
  });

  final AppPreferences prefs;
  final VoidCallback onPrefsChanged;

  @override
  State<SettingsScreen> createState() => _SettingsScreenState();
}

class _SettingsScreenState extends State<SettingsScreen> {
  final _settings = ConnectionSettings();
  final _urlCtrl = TextEditingController();
  final _tokenCtrl = TextEditingController();
  final _identityCtrl = TextEditingController();

  bool _testing = false;
  ({bool ok, String message})? _testResult;

  @override
  void initState() {
    super.initState();
    _restore();
  }

  Future<void> _restore() async {
    final s = await ConnectionSettings.load();
    if (!mounted) return;
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

  // Persist connection fields as they change; the home screen re-fetches on return so any
  // edit takes effect without an explicit save button.
  void _persistConnection() {
    _settings
      ..tokenServerUrl = _urlCtrl.text
      ..apiToken = _tokenCtrl.text
      ..identity = _identityCtrl.text;
    _settings.save();
    // A changed server invalidates the last test result.
    if (_testResult != null) setState(() => _testResult = null);
  }

  Future<void> _testConnection() async {
    _persistConnection();
    setState(() {
      _testing = true;
      _testResult = null;
    });
    final client = TokenClient(_settings);
    try {
      final (personas, _) = await client.fetchPersonas();
      if (!mounted) return;
      setState(() => _testResult = (
            ok: true,
            message: personas.isEmpty
                ? 'Reached the server — no personas configured yet.'
                : 'Connected — ${personas.length} persona${personas.length == 1 ? '' : 's'} available.',
          ));
    } catch (e) {
      if (mounted) setState(() => _testResult = (ok: false, message: '$e'));
    } finally {
      client.close();
      if (mounted) setState(() => _testing = false);
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
      appBar: AppBar(title: const Text('Settings')),
      body: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          _SectionLabel('Server'),
          TextField(
            controller: _urlCtrl,
            decoration: const InputDecoration(
              labelText: 'Token server URL',
              hintText: 'http://localhost:8080',
              border: OutlineInputBorder(),
              prefixIcon: Icon(Icons.dns_outlined),
            ),
            keyboardType: TextInputType.url,
            autocorrect: false,
            onChanged: (_) => _persistConnection(),
          ),
          const SizedBox(height: 12),
          TextField(
            controller: _tokenCtrl,
            decoration: const InputDecoration(
              labelText: 'API token (optional)',
              border: OutlineInputBorder(),
              prefixIcon: Icon(Icons.key_outlined),
            ),
            obscureText: true,
            onChanged: (_) => _persistConnection(),
          ),
          const SizedBox(height: 12),
          Row(
            children: [
              FilledButton.tonalIcon(
                onPressed: _testing ? null : _testConnection,
                icon: _testing
                    ? const SizedBox(
                        height: 16, width: 16, child: CircularProgressIndicator(strokeWidth: 2))
                    : const Icon(Icons.wifi_tethering),
                label: const Text('Test connection'),
              ),
            ],
          ),
          if (_testResult case final r?) ...[
            const SizedBox(height: 12),
            _TestResultBanner(ok: r.ok, message: r.message),
          ],
          const SizedBox(height: 24),
          _SectionLabel('You'),
          TextField(
            controller: _identityCtrl,
            decoration: const InputDecoration(
              labelText: 'Display name (optional)',
              hintText: 'How the assistant addresses you',
              border: OutlineInputBorder(),
              prefixIcon: Icon(Icons.person_outline),
            ),
            onChanged: (_) => _persistConnection(),
          ),
          const SizedBox(height: 24),
          _SectionLabel('Call defaults'),
          const SizedBox(height: 4),
          const Text('Mic mode', style: TextStyle(fontWeight: FontWeight.w500)),
          const SizedBox(height: 8),
          SegmentedButton<MicMode>(
            segments: const [
              ButtonSegment(
                value: MicMode.openMic,
                label: Text('Open mic'),
                icon: Icon(Icons.hearing),
              ),
              ButtonSegment(
                value: MicMode.pushToTalk,
                label: Text('Push to talk'),
                icon: Icon(Icons.touch_app),
              ),
            ],
            selected: {widget.prefs.defaultMicMode},
            onSelectionChanged: (s) {
              setState(() => widget.prefs.defaultMicMode = s.first);
              widget.onPrefsChanged();
            },
          ),
          const SizedBox(height: 20),
          const Text('Appearance', style: TextStyle(fontWeight: FontWeight.w500)),
          const SizedBox(height: 8),
          SegmentedButton<ThemeMode>(
            segments: const [
              ButtonSegment(value: ThemeMode.system, label: Text('System')),
              ButtonSegment(value: ThemeMode.light, label: Text('Light')),
              ButtonSegment(value: ThemeMode.dark, label: Text('Dark')),
            ],
            selected: {widget.prefs.themeMode},
            onSelectionChanged: (s) {
              setState(() => widget.prefs.themeMode = s.first);
              widget.onPrefsChanged();
            },
          ),
        ],
      ),
    );
  }
}

class _SectionLabel extends StatelessWidget {
  const _SectionLabel(this.text);
  final String text;

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 12),
      child: Text(
        text.toUpperCase(),
        style: Theme.of(context).textTheme.labelMedium?.copyWith(
              color: Theme.of(context).colorScheme.primary,
              letterSpacing: 0.8,
            ),
      ),
    );
  }
}

class _TestResultBanner extends StatelessWidget {
  const _TestResultBanner({required this.ok, required this.message});
  final bool ok;
  final String message;

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    final bg = ok ? scheme.secondaryContainer : scheme.errorContainer;
    final fg = ok ? scheme.onSecondaryContainer : scheme.onErrorContainer;
    return Container(
      padding: const EdgeInsets.symmetric(vertical: 10, horizontal: 12),
      decoration: BoxDecoration(color: bg, borderRadius: BorderRadius.circular(8)),
      child: Row(
        children: [
          Icon(ok ? Icons.check_circle_outline : Icons.error_outline, size: 18, color: fg),
          const SizedBox(width: 8),
          Expanded(child: Text(message, style: TextStyle(color: fg))),
        ],
      ),
    );
  }
}
