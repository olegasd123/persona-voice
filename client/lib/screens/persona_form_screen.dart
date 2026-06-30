import 'package:flutter/material.dart';

import '../models/connection_settings.dart';
import '../models/lora_option.dart';
import '../models/persona.dart';
import '../models/session_options.dart';
import '../models/voice_option.dart';
import '../services/token_client.dart';

/// Create or edit a custom persona. Returns `true` on a successful save (the caller reloads),
/// or null if dismissed. [personaId] null = a new persona; set = edit one of the user's own.
class PersonaFormScreen extends StatefulWidget {
  const PersonaFormScreen({
    super.key,
    required this.settings,
    required this.catalog,
    this.personaId,
  });

  final ConnectionSettings settings;
  final VoiceCatalog catalog;
  final String? personaId;

  bool get isEditing => personaId != null;

  @override
  State<PersonaFormScreen> createState() => _PersonaFormScreenState();
}

enum _Load { loading, ready, error }

/// How a *new* persona is being authored: by hand, or generated from a one-line description that
/// then prefills the manual form. Irrelevant when editing (an existing persona is always manual).
enum _Mode { manual, fromDraft }

class _PersonaFormScreenState extends State<PersonaFormScreen> {
  /// A few starter descriptions to make the draft mode discoverable (tap to fill the field).
  static const _examples = [
    'a patient French tutor who only speaks in B1',
    'a blunt product manager running a mock interview',
    'a calm bedtime storyteller for young kids',
  ];

  final _formKey = GlobalKey<FormState>();
  final _nameCtrl = TextEditingController();
  final _descCtrl = TextEditingController();
  final _promptCtrl = TextEditingController();
  final _draftDescCtrl = TextEditingController();

  PersonaDraft _draft = PersonaDraft();
  LoraCatalog _loras = LoraCatalog.empty;
  _Load _state = _Load.loading;
  String? _error;
  bool _saving = false;

  _Mode _mode = _Mode.manual;
  bool _drafting = false;
  String? _draftError;

  @override
  void initState() {
    super.initState();
    _bootstrap();
  }

  @override
  void dispose() {
    _nameCtrl.dispose();
    _descCtrl.dispose();
    _promptCtrl.dispose();
    _draftDescCtrl.dispose();
    super.dispose();
  }

  /// Draft a persona from [_draftDescCtrl] on the server, then prefill the manual form so the
  /// user reviews and saves. The draft is never persisted here — the existing Save (`createPersona`)
  /// does the write, exactly as for a hand-filled persona.
  Future<void> _generate() async {
    final description = _draftDescCtrl.text.trim();
    if (description.isEmpty) {
      setState(() => _draftError = 'Describe the persona you want.');
      return;
    }
    setState(() {
      _drafting = true;
      _draftError = null;
    });
    final client = TokenClient(widget.settings);
    try {
      final draft = await client.draftPersona(description);
      if (!mounted) return;
      _nameCtrl.text = draft.name;
      _descCtrl.text = draft.description;
      _promptCtrl.text = draft.systemPrompt;
      setState(() {
        _draft = draft;
        _mode = _Mode.manual;
        _drafting = false;
      });
      ScaffoldMessenger.of(context)
        ..clearSnackBars()
        ..showSnackBar(const SnackBar(content: Text('Draft ready — review and save.')));
    } catch (e) {
      if (mounted) {
        setState(() {
          _drafting = false;
          _draftError = _draftMessage(e);
        });
      }
    } finally {
      client.close();
    }
  }

  /// A user-facing line for a failed draft. A 4xx carries the server's reason (e.g. "could not
  /// draft a persona: …"); a transport failure hides the raw socket noise.
  String _draftMessage(Object e) {
    if (e is TokenClientException) {
      if (e.statusCode == null) {
        return 'Could not reach the server. Check your connection and try again.';
      }
      return e.message;
    }
    return 'Something went wrong drafting the persona. Try again.';
  }

  Future<void> _bootstrap() async {
    final client = TokenClient(widget.settings);
    try {
      // LoRA catalog is best-effort — an older server / non-LoRA backend just yields an empty
      // picker; it shouldn't block authoring.
      var loras = LoraCatalog.empty;
      try {
        loras = await client.fetchLoras();
      } catch (_) {}
      final draft = widget.isEditing
          ? await client.fetchPersona(widget.personaId!)
          : PersonaDraft();
      if (!mounted) return;
      _nameCtrl.text = draft.name;
      _descCtrl.text = draft.description;
      _promptCtrl.text = draft.systemPrompt;
      setState(() {
        _draft = draft;
        _loras = loras;
        _state = _Load.ready;
      });
    } catch (e) {
      if (mounted) {
        setState(() {
          _state = _Load.error;
          _error = '$e';
        });
      }
    } finally {
      client.close();
    }
  }

  Future<void> _save() async {
    if (!_formKey.currentState!.validate()) return;
    _draft
      ..name = _nameCtrl.text
      ..description = _descCtrl.text
      ..systemPrompt = _promptCtrl.text;

    setState(() => _saving = true);
    final client = TokenClient(widget.settings);
    try {
      if (widget.isEditing) {
        await client.updatePersona(widget.personaId!, _draft);
      } else {
        await client.createPersona(_draft);
      }
      if (mounted) Navigator.of(context).pop(true);
    } catch (e) {
      if (mounted) {
        setState(() => _saving = false);
        ScaffoldMessenger.of(context)
          ..clearSnackBars()
          ..showSnackBar(SnackBar(content: Text('Could not save: $e')));
      }
    } finally {
      client.close();
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text(widget.isEditing ? 'Edit persona' : 'New persona'),
        actions: [
          // No Save while drafting from a description: there's nothing to save until the draft
          // lands in (and flips back to) the manual form. Generate lives in the body instead.
          if (_state == _Load.ready && _mode == _Mode.manual)
            TextButton(
              onPressed: _saving ? null : _save,
              child: _saving
                  ? const SizedBox(
                      width: 18, height: 18, child: CircularProgressIndicator(strokeWidth: 2))
                  : const Text('Save'),
            ),
        ],
      ),
      body: switch (_state) {
        _Load.loading => const Center(child: CircularProgressIndicator()),
        _Load.error => _ErrorBody(message: _error ?? 'Failed to load', onRetry: _bootstrap),
        _Load.ready => _readyBody(context),
      },
    );
  }

  /// The ready body. Editing is always the manual form; a new persona gets a mode toggle on top so
  /// the user can author by hand or generate a draft that then prefills the same form.
  Widget _readyBody(BuildContext context) {
    if (widget.isEditing) return _form(context);
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        Padding(
          padding: const EdgeInsets.fromLTRB(16, 16, 16, 0),
          child: SegmentedButton<_Mode>(
            segments: const [
              ButtonSegment(
                value: _Mode.manual,
                label: Text('Manual'),
                icon: Icon(Icons.edit_outlined),
              ),
              ButtonSegment(
                value: _Mode.fromDraft,
                label: Text('From description'),
                icon: Icon(Icons.auto_awesome_outlined),
              ),
            ],
            selected: {_mode},
            // Lock the toggle mid-draft so a switch can't strand the in-flight request.
            onSelectionChanged:
                _drafting ? null : (s) => setState(() => _mode = s.first),
          ),
        ),
        Expanded(
          child: _mode == _Mode.manual ? _form(context) : _draftSubview(context),
        ),
      ],
    );
  }

  Widget _draftSubview(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    return ListView(
      padding: const EdgeInsets.fromLTRB(16, 16, 16, 32),
      children: [
        Text(
          'Describe the persona in plain language and the assistant drafts it for you. '
          "You'll review and tweak everything before it's saved.",
          style: TextStyle(fontSize: 13, color: scheme.onSurfaceVariant),
        ),
        const SizedBox(height: 16),
        TextField(
          controller: _draftDescCtrl,
          enabled: !_drafting,
          minLines: 3,
          maxLines: 6,
          textInputAction: TextInputAction.newline,
          decoration: const InputDecoration(
            labelText: 'Description',
            hintText: 'e.g. a patient French tutor who only speaks in B1',
            border: OutlineInputBorder(),
            alignLabelWithHint: true,
          ),
        ),
        const SizedBox(height: 12),
        Wrap(
          spacing: 8,
          runSpacing: 8,
          children: [
            for (final ex in _examples)
              ActionChip(
                label: Text(ex),
                onPressed:
                    _drafting ? null : () => setState(() => _draftDescCtrl.text = ex),
              ),
          ],
        ),
        if (_draftError != null) ...[
          const SizedBox(height: 16),
          Text(_draftError!, style: TextStyle(color: scheme.error)),
        ],
        const SizedBox(height: 24),
        FilledButton.icon(
          onPressed: _drafting ? null : _generate,
          icon: _drafting
              ? const SizedBox(
                  width: 18, height: 18, child: CircularProgressIndicator(strokeWidth: 2))
              : const Icon(Icons.auto_awesome),
          label: Text(_drafting ? 'Drafting…' : 'Generate persona'),
        ),
      ],
    );
  }

  Widget _form(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    return Form(
      key: _formKey,
      child: ListView(
        padding: const EdgeInsets.fromLTRB(16, 16, 16, 32),
        children: [
          TextFormField(
            controller: _nameCtrl,
            decoration: const InputDecoration(
              labelText: 'Name',
              border: OutlineInputBorder(),
            ),
            textInputAction: TextInputAction.next,
            validator: (v) =>
                (v == null || v.trim().isEmpty) ? 'Give your persona a name' : null,
          ),
          const SizedBox(height: 16),
          TextFormField(
            controller: _descCtrl,
            decoration: const InputDecoration(
              labelText: 'Description (optional)',
              helperText: 'One line shown on the persona shelf',
              border: OutlineInputBorder(),
            ),
          ),
          const SizedBox(height: 16),
          TextFormField(
            controller: _promptCtrl,
            decoration: const InputDecoration(
              labelText: 'System prompt',
              helperText: 'How the persona should behave and speak',
              border: OutlineInputBorder(),
              alignLabelWithHint: true,
            ),
            minLines: 4,
            maxLines: 10,
            validator: (v) =>
                (v == null || v.trim().isEmpty) ? 'A system prompt is required' : null,
          ),
          const SizedBox(height: 24),

          _Label('Voice'),
          _voiceField(scheme),
          const SizedBox(height: 20),

          _Label('Turn style'),
          _decoratedDropdown<TurnStyle>(
            icon: Icons.chat_bubble_outline,
            value: _draft.turnStyle,
            items: [
              for (final t in TurnStyle.values)
                DropdownMenuItem(value: t, child: Text(t.label)),
            ],
            onChanged: (v) => setState(() => _draft.turnStyle = v ?? TurnStyle.balanced),
          ),
          const SizedBox(height: 20),

          SwitchListTile(
            contentPadding: EdgeInsets.zero,
            title: const Text('Remember past conversations'),
            subtitle: Text(
              'Recall earlier sessions with this persona (needs consent on the device).',
              style: TextStyle(fontSize: 12, color: scheme.onSurfaceVariant),
            ),
            value: _draft.memoryEnabled,
            onChanged: (v) => setState(() => _draft.memoryEnabled = v),
          ),
          const Divider(height: 32),

          Text('Session defaults', style: Theme.of(context).textTheme.titleSmall),
          const SizedBox(height: 4),
          Text(
            'Baked into the persona; a per-call customization still overrides them.',
            style: TextStyle(fontSize: 12, color: scheme.onSurfaceVariant),
          ),
          const SizedBox(height: 16),
          _Label('Proficiency (CEFR)'),
          _decoratedDropdown<CefrLevel?>(
            icon: Icons.school_outlined,
            value: _draft.sessionDefaults.cefr,
            items: [
              const DropdownMenuItem(value: null, child: Text('None')),
              for (final level in CefrLevel.values)
                DropdownMenuItem(value: level, child: Text(level.label)),
            ],
            onChanged: (v) => setState(() => _draft.sessionDefaults =
                _draft.sessionDefaults.copyWith(cefr: v, clearCefr: v == null)),
          ),
          const SizedBox(height: 20),
          _Label('Demeanor'),
          _decoratedDropdown<Demeanor?>(
            icon: Icons.mood_outlined,
            value: _draft.sessionDefaults.demeanor,
            items: [
              const DropdownMenuItem(value: null, child: Text('Natural')),
              for (final d in Demeanor.values)
                DropdownMenuItem(value: d, child: Text(d.label)),
            ],
            onChanged: (v) => setState(() => _draft.sessionDefaults =
                _draft.sessionDefaults.copyWith(demeanor: v, clearDemeanor: v == null)),
          ),
          const Divider(height: 32),

          _Label('LoRA adapter'),
          _loraField(scheme),
        ],
      ),
    );
  }

  Widget _voiceField(ColorScheme scheme) {
    final available = widget.catalog.available;
    final values = <String?>[null, ...available.map((v) => v.id)];
    final chosen = _draft.voiceRef.isEmpty ? null : _draft.voiceRef;
    // Keep the persona's current voice selectable even if the catalog doesn't list it now.
    if (chosen != null && !values.contains(chosen)) values.add(chosen);

    if (available.isEmpty && chosen == null) {
      return Text(
        'No voice catalog from the server — the server picks a default voice.',
        style: TextStyle(fontSize: 13, color: scheme.outline),
      );
    }
    return _decoratedDropdown<String?>(
      icon: Icons.graphic_eq,
      value: chosen,
      items: [
        for (final v in values)
          DropdownMenuItem(value: v, child: Text(_voiceLabel(v), overflow: TextOverflow.ellipsis)),
      ],
      onChanged: (v) => setState(() => _draft.voiceRef = v ?? ''),
    );
  }

  String _voiceLabel(String? id) {
    if (id == null) return 'Server default';
    final option = widget.catalog.byId(id);
    if (option == null) return id;
    final kind = option.kind == 'preset' ? '' : ' · ${option.kindLabel}';
    return '${option.name}$kind';
  }

  Widget _loraField(ColorScheme scheme) {
    // Empty on Mac / LM Studio: a LoRA is merged into the base model at train time, not served.
    if (!_loras.supportsLora || _loras.loras.isEmpty) {
      return Text(
        _loras.reason ??
            'No served LoRA adapters on this backend — using the base model.',
        style: TextStyle(fontSize: 13, color: scheme.outline),
      );
    }
    final values = <String?>[null, ..._loras.loras.map((l) => l.id)];
    final chosen = _draft.lora;
    if (chosen != null && !values.contains(chosen)) values.add(chosen);
    return _decoratedDropdown<String?>(
      icon: Icons.memory_outlined,
      value: chosen,
      items: [
        const DropdownMenuItem(value: null, child: Text('Base model (no LoRA)')),
        for (final l in _loras.loras)
          DropdownMenuItem(value: l.id, child: Text(l.name, overflow: TextOverflow.ellipsis)),
        if (chosen != null && !_loras.loras.any((l) => l.id == chosen))
          DropdownMenuItem(value: chosen, child: Text('$chosen (not served)')),
      ],
      onChanged: (v) => setState(() => _draft.lora = v),
    );
  }

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

class _ErrorBody extends StatelessWidget {
  const _ErrorBody({required this.message, required this.onRetry});
  final String message;
  final VoidCallback onRetry;

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    return Padding(
      padding: const EdgeInsets.all(32),
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Icon(Icons.cloud_off_outlined, size: 48, color: scheme.outline),
          const SizedBox(height: 16),
          Text(message, textAlign: TextAlign.center),
          const SizedBox(height: 16),
          FilledButton(onPressed: onRetry, child: const Text('Retry')),
        ],
      ),
    );
  }
}
