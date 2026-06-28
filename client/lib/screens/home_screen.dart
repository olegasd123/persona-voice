import 'dart:async';

import 'package:flutter/material.dart';
import 'package:permission_handler/permission_handler.dart';

import '../models/app_preferences.dart';
import '../models/connection_settings.dart';
import '../models/persona.dart';
import '../models/session_options.dart';
import '../models/voice_option.dart';
import '../services/token_client.dart';
import '../services/voice_session.dart';
import 'call_screen.dart';
import 'persona_form_screen.dart';
import 'persona_options_sheet.dart';
import 'queue_screen.dart';
import 'settings_screen.dart';
import 'voice_library_screen.dart';

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
  VoiceCatalog _catalog = VoiceCatalog.empty;
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
      ..displayName = s.displayName
      ..userId = s.userId
      ..participantId = s.participantId;
    await _loadPersonas();
  }

  Future<void> _loadPersonas() async {
    if (mounted) setState(() => _state = _LoadState.loading);
    final client = TokenClient(_settings);
    try {
      final (personas, defaultId) = await client.fetchPersonas();
      // The voice catalog is best-effort — an older server or unconfigured registry shouldn't
      // block the persona shelf; the customize sheet just falls back to CEFR/demeanor only.
      var catalog = VoiceCatalog.empty;
      try {
        catalog = await client.fetchVoices();
      } catch (_) {}
      if (!mounted) return;
      setState(() {
        _personas = personas;
        _defaultId = defaultId;
        _catalog = catalog;
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

  Future<void> _openVoiceLibrary() async {
    await Navigator.of(context).push(MaterialPageRoute(
      builder: (_) => VoiceLibraryScreen(settings: _settings),
    ));
    // Adding/removing clones changes the catalog the customize sheet offers.
    if (mounted) await _loadPersonas();
  }

  Future<void> _customize(Persona persona) async {
    final result = await showPersonaOptionsSheet(
      context,
      persona: persona,
      current: widget.prefs.optionsFor(persona.id),
      catalog: _catalog,
      onManageVoices: _openVoiceLibrary,
    );
    if (result == null) return; // dismissed without applying
    await widget.prefs.setOptionsFor(persona.id, result);
    if (mounted) setState(() {});
  }

  /// Open the New / Edit persona form. [personaId] null = create; set = edit a custom persona.
  Future<void> _openPersonaForm({String? personaId}) async {
    final saved = await Navigator.of(context).push<bool>(MaterialPageRoute(
      builder: (_) => PersonaFormScreen(
        settings: _settings,
        catalog: _catalog,
        personaId: personaId,
      ),
    ));
    if (saved == true && mounted) await _loadPersonas();
  }

  Future<void> _deletePersona(Persona persona) async {
    final confirmed = await showDialog<bool>(
      context: context,
      builder: (ctx) => AlertDialog(
        title: Text('Delete ${persona.name}?'),
        content: const Text('This removes your custom persona. It can\'t be undone.'),
        actions: [
          TextButton(onPressed: () => Navigator.of(ctx).pop(false), child: const Text('Cancel')),
          FilledButton.tonal(
            onPressed: () => Navigator.of(ctx).pop(true),
            child: const Text('Delete'),
          ),
        ],
      ),
    );
    if (confirmed != true) return;
    final client = TokenClient(_settings);
    try {
      await client.deletePersona(persona.id);
      await widget.prefs.setOptionsFor(persona.id, const SessionOptions()); // drop stale options
      if (mounted) await _loadPersonas();
    } catch (e) {
      if (mounted) _snack(friendlyTokenError(e), error: true);
    } finally {
      client.close();
    }
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
    final options = widget.prefs.optionsFor(persona.id);
    try {
      final grant = await client.requestToken(persona: persona.id, options: options);
      if (!mounted) return;
      _enterCall(grant, options);
    } on TokenClientException catch (e) {
      if (!mounted) return;
      // Every session is busy: queue (wait for a free slot) instead of failing with an error.
      if (e.isBusy) {
        _enterQueue(persona, options, e.retryAfter);
      } else {
        _snack(friendlyTokenError(e), error: true);
      }
    } catch (e) {
      if (mounted) _snack(friendlyTokenError(e), error: true);
    } finally {
      client.close();
      if (mounted) setState(() => _connectingId = null);
    }
  }

  /// Drop into the live call with a freshly minted token.
  void _enterCall(JoinGrant grant, SessionOptions options) {
    Navigator.of(context).push(MaterialPageRoute(
      builder: (_) => CallScreen(
        session: VoiceSession(),
        grant: grant,
        personas: _personas,
        options: options,
        initialMicMode: widget.prefs.defaultMicMode,
      ),
    ));
  }

  /// All sessions busy: open the queue page, which waits for a slot and then enters the call.
  void _enterQueue(Persona persona, SessionOptions options, int? retryAfter) {
    Navigator.of(context).push(MaterialPageRoute(
      builder: (_) => QueueScreen(
        settings: _settings,
        persona: persona,
        personas: _personas,
        options: options,
        initialMicMode: widget.prefs.defaultMicMode,
        initialRetryAfter: retryAfter,
      ),
    ));
  }

  void _snack(String message, {bool error = false}) {
    final scheme = Theme.of(context).colorScheme;
    ScaffoldMessenger.of(context)
      ..clearSnackBars()
      ..showSnackBar(SnackBar(
        content: Text(
          message,
          style: error ? TextStyle(color: scheme.onErrorContainer) : null,
        ),
        backgroundColor: error ? scheme.errorContainer : null,
      ));
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
            icon: const Icon(Icons.record_voice_over_outlined),
            tooltip: 'Voice library',
            onPressed: _openVoiceLibrary,
          ),
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
      // Authoring needs a reachable server (the form fetches voices / LoRAs and posts the draft).
      floatingActionButton: (_state == _LoadState.ok || _state == _LoadState.unconfigured)
          ? FloatingActionButton.extended(
              onPressed: _connectingId == null ? () => _openPersonaForm() : null,
              icon: const Icon(Icons.add),
              label: const Text('New persona'),
            )
          : null,
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
              options: widget.prefs.optionsFor(p.id),
              catalog: _catalog,
              connecting: p.id == _connectingId,
              disabled: _connectingId != null && p.id != _connectingId,
              onTap: () => _call(p),
              onCustomize: () => _customize(p),
              onEdit: p.custom ? () => _openPersonaForm(personaId: p.id) : null,
              onDelete: p.custom ? () => _deletePersona(p) : null,
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

/// A tappable persona row: avatar, name, description, and the voice / CEFR / demeanor it will
/// speak with (reflecting any per-call override) — plus a call affordance that becomes a
/// spinner while dialling.
class _PersonaCard extends StatelessWidget {
  const _PersonaCard({
    required this.persona,
    required this.isDefault,
    required this.isLastUsed,
    required this.options,
    required this.catalog,
    required this.connecting,
    required this.disabled,
    required this.onTap,
    required this.onCustomize,
    this.onEdit,
    this.onDelete,
  });

  final Persona persona;
  final bool isDefault;
  final bool isLastUsed;

  /// The user's saved per-call overrides for this persona (voice / CEFR / demeanor); empty
  /// when untouched. Drives the "Tuned" tag and the *effective* attributes shown on the card.
  final SessionOptions options;

  /// The active backend's voice catalog — used to resolve an overridden voice id to its
  /// descriptor (and to tell presets, which have one, from clones/fine-tunes, which don't).
  final VoiceCatalog catalog;

  final bool connecting;
  final bool disabled;
  final VoidCallback onTap;
  final VoidCallback onCustomize;

  /// Set only for user-authored (custom) personas — curated personas are read-only.
  final VoidCallback? onEdit;
  final VoidCallback? onDelete;

  bool get hasOptions => options.isNotEmpty;

  /// The voice blurb to show. When a voice override is set, that wins: a preset resolves to its
  /// descriptor ("warm, soft, feminine"); a clone/fine-tune (no descriptor) or an unknown/stale
  /// id hides the line. With no override, fall back to the persona's authored voice blurb (which
  /// is itself empty when the authored voice is a clone). Null = render no voice line.
  String? get _voiceDescriptor {
    final overrideId = options.voice;
    if (overrideId != null) {
      final option = catalog.byId(overrideId);
      return (option != null && option.isPreset) ? option.name : null;
    }
    return persona.voice.isNotEmpty ? persona.voice : null;
  }

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    final isDark = scheme.brightness == Brightness.dark;
    final subtitle = persona.description.isNotEmpty ? persona.description : persona.id;
    // Effective attributes = the per-call override, falling back to the persona's own default.
    final voiceDescriptor = _voiceDescriptor;
    final cefr = options.cefr ?? persona.cefr;
    final demeanor = options.demeanor ?? persona.demeanor;
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
                          if (isLastUsed)
                            _MiniTag(
                              'Last used',
                              isDark ? Colors.green.shade800 : Colors.green.shade100,
                              isDark ? Colors.green.shade100 : Colors.green.shade800,
                            )
                          else if (isDefault) _MiniTag('Default',
                              scheme.secondaryContainer, scheme.onSecondaryContainer),
                          if (persona.custom)
                            _MiniTag('Custom', scheme.tertiaryContainer,
                                scheme.onTertiaryContainer),
                          if (hasOptions)
                            _MiniTag('Tuned', scheme.primaryContainer,
                                scheme.onPrimaryContainer),
                        ],
                      ),
                      const SizedBox(height: 2),
                      Text(
                        subtitle,
                        style: TextStyle(fontSize: 13, color: scheme.onSurfaceVariant),
                        maxLines: 2,
                        overflow: TextOverflow.ellipsis,
                      ),
                      if (voiceDescriptor != null) ...[
                        const SizedBox(height: 4),
                        Row(
                          children: [
                            Icon(Icons.graphic_eq, size: 13, color: scheme.outline),
                            const SizedBox(width: 4),
                            Flexible(
                              child: Text(
                                voiceDescriptor,
                                style: TextStyle(fontSize: 12, color: scheme.outline),
                                overflow: TextOverflow.ellipsis,
                              ),
                            ),
                          ],
                        ),
                      ],
                      if (cefr != null || demeanor != null) ...[
                        const SizedBox(height: 4),
                        Row(
                          children: [
                            if (cefr != null) ...[
                              Icon(Icons.school_outlined, size: 13, color: scheme.outline),
                              const SizedBox(width: 4),
                              Text(cefr.label,
                                  style: TextStyle(fontSize: 12, color: scheme.outline)),
                            ],
                            if (cefr != null && demeanor != null) const SizedBox(width: 12),
                            if (demeanor != null) ...[
                              Icon(Icons.mood_outlined, size: 13, color: scheme.outline),
                              const SizedBox(width: 4),
                              Text(demeanor.label,
                                  style: TextStyle(fontSize: 12, color: scheme.outline)),
                            ],
                          ],
                        ),
                      ],
                    ],
                  ),
                ),
                const SizedBox(width: 4),
                if (onEdit != null || onDelete != null)
                  PopupMenuButton<String>(
                    tooltip: 'Edit or delete',
                    enabled: !disabled && !connecting,
                    onSelected: (v) {
                      if (v == 'edit') onEdit?.call();
                      if (v == 'delete') onDelete?.call();
                    },
                    itemBuilder: (_) => [
                      if (onEdit != null)
                        const PopupMenuItem(value: 'edit', child: Text('Edit')),
                      if (onDelete != null)
                        const PopupMenuItem(value: 'delete', child: Text('Delete')),
                    ],
                  ),
                IconButton(
                  icon: Icon(
                    hasOptions ? Icons.tune : Icons.tune_outlined,
                    color: hasOptions ? scheme.primary : scheme.outline,
                  ),
                  tooltip: 'Customize',
                  onPressed: disabled || connecting ? null : onCustomize,
                ),
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
