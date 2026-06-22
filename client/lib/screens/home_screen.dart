import 'dart:async';

import 'package:flutter/material.dart';
import 'package:permission_handler/permission_handler.dart';

import '../models/app_preferences.dart';
import '../models/connection_settings.dart';
import '../models/persona.dart';
import '../services/token_client.dart';
import '../services/voice_session.dart';
import 'call_screen.dart';
import 'settings_screen.dart';

/// The launch screen: a shelf of personas to call. Connection settings live in
/// [SettingsScreen] now — here you just pick who to talk to and tap to call.
enum _LoadState { loading, ok, error, unconfigured }

class HomeScreen extends StatefulWidget {
  const HomeScreen({
    super.key,
    required this.prefs,
    required this.onPrefsChanged,
  });

  final AppPreferences prefs;
  final VoidCallback onPrefsChanged;

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  final _settings = ConnectionSettings();

  List<Persona> _personas = [];
  String? _defaultId;
  _LoadState _state = _LoadState.loading;
  String? _error;
  String? _connectingId; // persona currently being dialled

  @override
  void initState() {
    super.initState();
    _bootstrap();
  }

  Future<void> _bootstrap() async {
    final s = await ConnectionSettings.load();
    _settings
      ..tokenServerUrl = s.tokenServerUrl
      ..apiToken = s.apiToken
      ..identity = s.identity;
    await _loadPersonas();
  }

  Future<void> _loadPersonas() async {
    if (mounted) setState(() => _state = _LoadState.loading);
    final client = TokenClient(_settings);
    try {
      final (personas, defaultId) = await client.fetchPersonas();
      if (!mounted) return;
      setState(() {
        _personas = personas;
        _defaultId = defaultId;
        _state = personas.isEmpty ? _LoadState.unconfigured : _LoadState.ok;
        _error = null;
      });
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _state = _LoadState.error;
        _error = '$e';
      });
    } finally {
      client.close();
    }
  }

  Future<void> _openSettings() async {
    await Navigator.of(context).push(MaterialPageRoute(
      builder: (_) => SettingsScreen(
        prefs: widget.prefs,
        onPrefsChanged: widget.onPrefsChanged,
      ),
    ));
    // Settings may have changed the server / token — refresh against the new config.
    await _bootstrap();
  }

  Future<void> _call(Persona persona) async {
    final mic = await Permission.microphone.request();
    if (!mic.isGranted) {
      _snack('Microphone permission is required to talk.');
      return;
    }
    widget.prefs.lastPersonaId = persona.id;
    unawaited(AppPreferences.rememberPersona(persona.id));

    setState(() => _connectingId = persona.id);
    final client = TokenClient(_settings);
    try {
      final grant = await client.requestToken(persona: persona.id);
      if (!mounted) return;
      final session = VoiceSession();
      await Navigator.of(context).push(MaterialPageRoute(
        builder: (_) => CallScreen(
          session: session,
          grant: grant,
          personas: _personas,
          initialMicMode: widget.prefs.defaultMicMode,
        ),
      ));
    } catch (e) {
      if (mounted) _snack('Could not connect: $e');
    } finally {
      client.close();
      if (mounted) setState(() => _connectingId = null);
    }
  }

  void _snack(String message) {
    ScaffoldMessenger.of(context)
      ..clearSnackBars()
      ..showSnackBar(SnackBar(content: Text(message)));
  }

  // Last-called persona floats to the top so the common case is one tap away; the rest keep
  // the server's order (a plain sort here wouldn't — Dart's List.sort isn't stable).
  List<Persona> get _ordered {
    final idx = _personas.indexWhere((p) => p.id == widget.prefs.lastPersonaId);
    if (idx <= 0) return _personas; // not found, or already first
    return [
      _personas[idx],
      ..._personas.sublist(0, idx),
      ..._personas.sublist(idx + 1),
    ];
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Personas'),
        actions: [
          IconButton(
            icon: const Icon(Icons.refresh),
            tooltip: 'Reload personas',
            onPressed: _state == _LoadState.loading ? null : _loadPersonas,
          ),
          IconButton(
            icon: const Icon(Icons.settings_outlined),
            tooltip: 'Settings',
            onPressed: _openSettings,
          ),
        ],
      ),
      body: Column(
        children: [
          _ConnectionChip(
            state: _state,
            host: _hostLabel(_settings.tokenServerUrl),
          ),
          Expanded(
            child: RefreshIndicator(
              onRefresh: _loadPersonas,
              child: _body(),
            ),
          ),
        ],
      ),
    );
  }

  Widget _body() {
    switch (_state) {
      case _LoadState.loading when _personas.isEmpty:
        return const _CenteredScroll(child: CircularProgressIndicator());
      case _LoadState.error:
        return _CenteredScroll(
          child: _EmptyState(
            icon: Icons.cloud_off_outlined,
            title: "Can't reach the server",
            message: _error ?? 'Check the token server URL in Settings.',
            primaryLabel: 'Open settings',
            onPrimary: _openSettings,
            secondaryLabel: 'Retry',
            onSecondary: _loadPersonas,
          ),
        );
      case _LoadState.unconfigured:
        return _CenteredScroll(
          child: _EmptyState(
            icon: Icons.theater_comedy_outlined,
            title: 'No personas yet',
            message: 'This server has no personas configured.',
            primaryLabel: 'Open settings',
            onPrimary: _openSettings,
            secondaryLabel: 'Retry',
            onSecondary: _loadPersonas,
          ),
        );
      default:
        final personas = _ordered;
        final lastId = widget.prefs.lastPersonaId;
        return ListView.builder(
          padding: const EdgeInsets.fromLTRB(12, 12, 12, 24),
          itemCount: personas.length,
          itemBuilder: (_, i) {
            final p = personas[i];
            return _PersonaCard(
              persona: p,
              isDefault: p.id == _defaultId,
              isLastUsed: p.id == lastId,
              connecting: p.id == _connectingId,
              disabled: _connectingId != null && p.id != _connectingId,
              onTap: () => _call(p),
            );
          },
        );
    }
  }

  static String _hostLabel(String url) {
    final uri = Uri.tryParse(url.trim());
    if (uri == null || uri.host.isEmpty) return url.trim();
    return uri.hasPort ? '${uri.host}:${uri.port}' : uri.host;
  }
}

class _ConnectionChip extends StatelessWidget {
  const _ConnectionChip({required this.state, required this.host});
  final _LoadState state;
  final String host;

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    final (IconData icon, String text, Color bg, Color fg) = switch (state) {
      _LoadState.ok || _LoadState.unconfigured => (
          Icons.check_circle,
          'Connected · $host',
          scheme.secondaryContainer,
          scheme.onSecondaryContainer,
        ),
      _LoadState.error => (
          Icons.error_outline,
          "Can't reach $host",
          scheme.errorContainer,
          scheme.onErrorContainer,
        ),
      _LoadState.loading => (
          Icons.sync,
          'Connecting to $host…',
          scheme.surfaceContainerHighest,
          scheme.onSurfaceVariant,
        ),
    };
    return Container(
      width: double.infinity,
      color: bg,
      padding: const EdgeInsets.symmetric(vertical: 8, horizontal: 16),
      child: Row(
        children: [
          Icon(icon, size: 16, color: fg),
          const SizedBox(width: 8),
          Expanded(
            child: Text(text,
                style: TextStyle(color: fg, fontSize: 13), overflow: TextOverflow.ellipsis),
          ),
        ],
      ),
    );
  }
}

/// A tappable persona row: avatar, name, description, voice — and a call affordance that
/// becomes a spinner while dialling.
class _PersonaCard extends StatelessWidget {
  const _PersonaCard({
    required this.persona,
    required this.isDefault,
    required this.isLastUsed,
    required this.connecting,
    required this.disabled,
    required this.onTap,
  });

  final Persona persona;
  final bool isDefault;
  final bool isLastUsed;
  final bool connecting;
  final bool disabled;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    final subtitle = persona.description.isNotEmpty ? persona.description : persona.id;
    return Opacity(
      opacity: disabled ? 0.5 : 1,
      child: Card(
        margin: const EdgeInsets.symmetric(vertical: 6),
        clipBehavior: Clip.antiAlias,
        child: InkWell(
          onTap: disabled || connecting ? null : onTap,
          child: Padding(
            padding: const EdgeInsets.all(14),
            child: Row(
              children: [
                CircleAvatar(
                  radius: 24,
                  backgroundColor: scheme.primaryContainer,
                  child: Text(
                    persona.initial,
                    style: TextStyle(
                      color: scheme.onPrimaryContainer,
                      fontWeight: FontWeight.w600,
                      fontSize: 18,
                    ),
                  ),
                ),
                const SizedBox(width: 14),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Row(
                        children: [
                          Flexible(
                            child: Text(
                              persona.name,
                              style: const TextStyle(
                                  fontSize: 16, fontWeight: FontWeight.w600),
                              overflow: TextOverflow.ellipsis,
                            ),
                          ),
                          if (isLastUsed) _MiniTag('Last used', scheme.tertiaryContainer,
                              scheme.onTertiaryContainer)
                          else if (isDefault) _MiniTag('Default',
                              scheme.secondaryContainer, scheme.onSecondaryContainer),
                        ],
                      ),
                      const SizedBox(height: 2),
                      Text(
                        subtitle,
                        style: TextStyle(fontSize: 13, color: scheme.onSurfaceVariant),
                        maxLines: 2,
                        overflow: TextOverflow.ellipsis,
                      ),
                      if (persona.voice.isNotEmpty) ...[
                        const SizedBox(height: 4),
                        Row(
                          children: [
                            Icon(Icons.graphic_eq, size: 13, color: scheme.outline),
                            const SizedBox(width: 4),
                            Flexible(
                              child: Text(
                                persona.voice,
                                style: TextStyle(fontSize: 12, color: scheme.outline),
                                overflow: TextOverflow.ellipsis,
                              ),
                            ),
                          ],
                        ),
                      ],
                    ],
                  ),
                ),
                const SizedBox(width: 8),
                connecting
                    ? const SizedBox(
                        width: 22, height: 22, child: CircularProgressIndicator(strokeWidth: 2))
                    : Icon(Icons.call, color: scheme.primary),
              ],
            ),
          ),
        ),
      ),
    );
  }
}

class _MiniTag extends StatelessWidget {
  const _MiniTag(this.label, this.bg, this.fg);
  final String label;
  final Color bg;
  final Color fg;

  @override
  Widget build(BuildContext context) {
    return Container(
      margin: const EdgeInsets.only(left: 6),
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 2),
      decoration: BoxDecoration(color: bg, borderRadius: BorderRadius.circular(999)),
      child: Text(label, style: TextStyle(fontSize: 10, color: fg, fontWeight: FontWeight.w500)),
    );
  }
}

class _EmptyState extends StatelessWidget {
  const _EmptyState({
    required this.icon,
    required this.title,
    required this.message,
    required this.primaryLabel,
    required this.onPrimary,
    required this.secondaryLabel,
    required this.onSecondary,
  });

  final IconData icon;
  final String title;
  final String message;
  final String primaryLabel;
  final VoidCallback onPrimary;
  final String secondaryLabel;
  final VoidCallback onSecondary;

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    return Padding(
      padding: const EdgeInsets.all(32),
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Icon(icon, size: 56, color: scheme.outline),
          const SizedBox(height: 16),
          Text(title, style: Theme.of(context).textTheme.titleMedium),
          const SizedBox(height: 8),
          Text(message,
              textAlign: TextAlign.center,
              style: TextStyle(color: scheme.onSurfaceVariant)),
          const SizedBox(height: 20),
          FilledButton(onPressed: onPrimary, child: Text(primaryLabel)),
          const SizedBox(height: 8),
          TextButton(onPressed: onSecondary, child: Text(secondaryLabel)),
        ],
      ),
    );
  }
}

/// Wraps a centered child in an always-scrollable view so [RefreshIndicator] (pull-to-refresh)
/// still works on the empty / loading states.
class _CenteredScroll extends StatelessWidget {
  const _CenteredScroll({required this.child});
  final Widget child;

  @override
  Widget build(BuildContext context) {
    return LayoutBuilder(
      builder: (_, constraints) => SingleChildScrollView(
        physics: const AlwaysScrollableScrollPhysics(),
        child: ConstrainedBox(
          constraints: BoxConstraints(minHeight: constraints.maxHeight),
          child: Center(child: child),
        ),
      ),
    );
  }
}
