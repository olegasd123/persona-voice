import 'package:flutter/material.dart';

import '../models/persona.dart';
import '../models/session_options.dart';
import '../models/voice_option.dart';

/// Bottom sheet to pick per-persona session overrides (voice / CEFR / demeanor) before a call.
/// Returns the chosen [SessionOptions] on "Apply", or null if dismissed. An empty result means
/// "run the persona as authored".
Future<SessionOptions?> showPersonaOptionsSheet(
  BuildContext context, {
  required Persona persona,
  required SessionOptions current,
  required VoiceCatalog catalog,
  VoidCallback? onManageVoices,
}) {
  return showModalBottomSheet<SessionOptions>(
    context: context,
    isScrollControlled: true,
    showDragHandle: true,
    builder: (_) => _PersonaOptionsSheet(
      persona: persona,
      current: current,
      catalog: catalog,
      onManageVoices: onManageVoices,
    ),
  );
}

class _PersonaOptionsSheet extends StatefulWidget {
  const _PersonaOptionsSheet({
    required this.persona,
    required this.current,
    required this.catalog,
    this.onManageVoices,
  });

  final Persona persona;
  final SessionOptions current;
  final VoiceCatalog catalog;
  final VoidCallback? onManageVoices;

  @override
  State<_PersonaOptionsSheet> createState() => _PersonaOptionsSheetState();
}

class _PersonaOptionsSheetState extends State<_PersonaOptionsSheet> {
  late SessionOptions _opts = widget.current;

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    final unavailableVoices = widget.catalog.unavailable.length;
    return Padding(
      padding: EdgeInsets.fromLTRB(
        20,
        4,
        20,
        20 + MediaQuery.of(context).viewInsets.bottom,
      ),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Expanded(
                child: Text(
                  'Customize ${widget.persona.name}',
                  style: Theme.of(context).textTheme.titleLarge,
                ),
              ),
              if (_opts.isNotEmpty)
                TextButton(
                  onPressed: () => setState(() => _opts = const SessionOptions()),
                  child: const Text('Reset'),
                ),
            ],
          ),
          const SizedBox(height: 2),
          Text(
            'Applied just for your next call — the persona is unchanged otherwise.',
            style: TextStyle(fontSize: 13, color: scheme.onSurfaceVariant),
          ),
          const SizedBox(height: 20),
          _Label('Voice'),
          _voiceField(scheme),
          if (unavailableVoices > 0) ...[
            const SizedBox(height: 6),
            Text(
              '$unavailableVoices voice${unavailableVoices == 1 ? '' : 's'} need a cloning '
              'backend and are hidden.',
              style: TextStyle(fontSize: 12, color: scheme.outline),
            ),
          ],
          const SizedBox(height: 20),
          _Label('Proficiency (CEFR)'),
          _cefrField(),
          const SizedBox(height: 20),
          _Label('Demeanor'),
          _demeanorField(),
          const SizedBox(height: 24),
          Row(
            children: [
              if (widget.onManageVoices != null)
                TextButton.icon(
                  onPressed: () {
                    Navigator.of(context).pop();
                    widget.onManageVoices!();
                  },
                  icon: const Icon(Icons.record_voice_over_outlined, size: 18),
                  label: const Text('Voice library'),
                ),
              const Spacer(),
              FilledButton(
                onPressed: () => Navigator.of(context).pop(_opts),
                child: const Text('Apply'),
              ),
            ],
          ),
        ],
      ),
    );
  }

  Widget _voiceField(ColorScheme scheme) {
    final available = widget.catalog.available;
    // Build the selectable values: "persona default" (null) + every available voice, plus the
    // currently-chosen voice even if it's missing/unavailable now (so we never silently drop it).
    final values = <String?>[null, ...available.map((v) => v.id)];
    final chosen = _opts.voice;
    if (chosen != null && !values.contains(chosen)) values.add(chosen);

    if (available.isEmpty && chosen == null) {
      return Text(
        widget.catalog.voices.isEmpty
            ? 'No voice catalog from the server — using the persona voice.'
            : 'No selectable voices on this backend — using the persona voice.',
        style: TextStyle(fontSize: 13, color: scheme.outline),
      );
    }

    return _decoratedDropdown<String?>(
      icon: Icons.graphic_eq,
      value: chosen,
      items: [
        for (final v in values)
          DropdownMenuItem<String?>(
            value: v,
            child: Text(_voiceLabel(v), overflow: TextOverflow.ellipsis),
          ),
      ],
      onChanged: (v) => setState(() => _opts = _opts.copyWith(voice: v, clearVoice: v == null)),
    );
  }

  String _voiceLabel(String? id) {
    if (id == null) return 'Persona default';
    final option = widget.catalog.byId(id);
    if (option == null) return '$id (unavailable)';
    final kind = option.kind == 'preset' ? '' : ' · ${option.kindLabel}';
    return '${option.name}$kind';
  }

  Widget _cefrField() {
    return _decoratedDropdown<CefrLevel?>(
      icon: Icons.school_outlined,
      value: _opts.cefr,
      items: [
        const DropdownMenuItem<CefrLevel?>(value: null, child: Text('Persona default')),
        for (final level in CefrLevel.values)
          DropdownMenuItem<CefrLevel?>(value: level, child: Text(_cefrLabel(level))),
      ],
      onChanged: (v) => setState(() => _opts = _opts.copyWith(cefr: v, clearCefr: v == null)),
    );
  }

  static String _cefrLabel(CefrLevel level) => switch (level) {
        CefrLevel.a1 => 'A1 · Beginner',
        CefrLevel.a2 => 'A2 · Elementary',
        CefrLevel.b1 => 'B1 · Intermediate',
        CefrLevel.b2 => 'B2 · Upper-intermediate',
        CefrLevel.c1 => 'C1 · Advanced',
        CefrLevel.c2 => 'C2 · Near-native',
      };

  Widget _demeanorField() {
    return _decoratedDropdown<Demeanor?>(
      icon: Icons.mood_outlined,
      value: _opts.demeanor,
      items: [
        const DropdownMenuItem<Demeanor?>(value: null, child: Text('Persona default')),
        for (final d in Demeanor.values)
          DropdownMenuItem<Demeanor?>(value: d, child: Text(_demeanorLabel(d))),
      ],
      onChanged: (v) =>
          setState(() => _opts = _opts.copyWith(demeanor: v, clearDemeanor: v == null)),
    );
  }

  /// A bordered, *controlled* dropdown (value tracks state, so "Reset" updates it live —
  /// unlike `DropdownButtonFormField`, whose internal state ignores later value changes).
  Widget _decoratedDropdown<T>({
    required IconData icon,
    required T value,
    required List<DropdownMenuItem<T>> items,
    required ValueChanged<T?> onChanged,
  }) {
    return InputDecorator(
      decoration: InputDecoration(
        border: const OutlineInputBorder(),
        prefixIcon: Icon(icon),
        contentPadding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
      ),
      child: DropdownButtonHideUnderline(
        child: DropdownButton<T>(
          value: value,
          isExpanded: true,
          isDense: true,
          items: items,
          onChanged: onChanged,
        ),
      ),
    );
  }

  static String _demeanorLabel(Demeanor d) => switch (d) {
        Demeanor.kind => 'Kind · warm and patient',
        Demeanor.natural => 'Natural · as authored',
        Demeanor.rude => 'Blunt · terse and impatient',
      };
}

class _Label extends StatelessWidget {
  const _Label(this.text);
  final String text;

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 8),
      child: Text(text, style: const TextStyle(fontWeight: FontWeight.w600)),
    );
  }
}
