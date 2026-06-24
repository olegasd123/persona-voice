import 'dart:async';
import 'dart:typed_data';

import 'package:file_picker/file_picker.dart';
import 'package:flutter/material.dart';

import '../models/connection_settings.dart';
import '../models/voice_option.dart';
import '../services/token_client.dart';
import '../services/voice_recorder.dart';

/// Server name rule for a clone (mirrors `_VOICE_NAME_RE` in token_server.py).
final _voiceNameRe = RegExp(r'^[A-Za-z0-9_-]+$');

/// Upload cap the server enforces (`_DEFAULT_MAX_CLONE_BYTES`).
const _maxSampleBytes = 10 * 1024 * 1024;

enum _LoadState { loading, ok, error }

/// The voice library (N2): lists the server's voice catalog grouped by kind, and lets the user
/// add a clone from a file or the device mic, or delete one they've added.
class VoiceLibraryScreen extends StatefulWidget {
  const VoiceLibraryScreen({super.key, required this.settings});

  final ConnectionSettings settings;

  @override
  State<VoiceLibraryScreen> createState() => _VoiceLibraryScreenState();
}

class _VoiceLibraryScreenState extends State<VoiceLibraryScreen> {
  VoiceCatalog _catalog = VoiceCatalog.empty;
  _LoadState _state = _LoadState.loading;
  String? _error;

  @override
  void initState() {
    super.initState();
    _load();
  }

  Future<void> _load() async {
    if (mounted) setState(() => _state = _LoadState.loading);
    final client = TokenClient(widget.settings);
    try {
      final catalog = await client.fetchVoices();
      if (!mounted) return;
      setState(() {
        _catalog = catalog;
        _state = _LoadState.ok;
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

  Future<void> _addVoice() async {
    final added = await showModalBottomSheet<bool>(
      context: context,
      isScrollControlled: true,
      showDragHandle: true,
      builder: (_) => _EnrollVoiceSheet(settings: widget.settings),
    );
    if (added == true) {
      _snack('Voice added.');
      await _load();
    }
  }

  Future<void> _deleteVoice(VoiceOption voice) async {
    final confirmed = await showDialog<bool>(
      context: context,
      builder: (_) => AlertDialog(
        title: Text('Delete "${voice.name}"?'),
        content: const Text(
          'This removes the cloned voice and unassigns it from any persona using it.',
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.of(context).pop(false),
            child: const Text('Cancel'),
          ),
          FilledButton(
            onPressed: () => Navigator.of(context).pop(true),
            child: const Text('Delete'),
          ),
        ],
      ),
    );
    if (confirmed != true) return;
    final client = TokenClient(widget.settings);
    try {
      await client.deleteVoice(voice.id);
      _snack('Deleted "${voice.name}".');
      await _load();
    } catch (e) {
      _snack('Could not delete: $e');
    } finally {
      client.close();
    }
  }

  void _snack(String message) {
    if (!mounted) return;
    ScaffoldMessenger.of(context)
      ..clearSnackBars()
      ..showSnackBar(SnackBar(content: Text(message)));
  }

  @override
  Widget build(BuildContext context) {
    final canClone = _catalog.supportsCloning;
    return Scaffold(
      appBar: AppBar(
        title: const Text('Voice library'),
        actions: [
          IconButton(
            icon: const Icon(Icons.refresh),
            tooltip: 'Reload',
            onPressed: _state == _LoadState.loading ? null : _load,
          ),
        ],
      ),
      floatingActionButton: _state == _LoadState.ok && canClone
          ? FloatingActionButton.extended(
              onPressed: _addVoice,
              icon: const Icon(Icons.add),
              label: const Text('Add voice'),
            )
          : null,
      body: RefreshIndicator(onRefresh: _load, child: _body()),
    );
  }

  Widget _body() {
    switch (_state) {
      case _LoadState.loading:
        return const Center(child: CircularProgressIndicator());
      case _LoadState.error:
        return ListView(
          children: [
            const SizedBox(height: 80),
            Icon(Icons.cloud_off_outlined,
                size: 56, color: Theme.of(context).colorScheme.outline),
            const SizedBox(height: 16),
            Center(
              child: Text(_error ?? 'Could not load the voice catalog.',
                  textAlign: TextAlign.center),
            ),
            const SizedBox(height: 16),
            Center(child: FilledButton(onPressed: _load, child: const Text('Retry'))),
          ],
        );
      case _LoadState.ok:
        return ListView(
          padding: const EdgeInsets.only(bottom: 96),
          children: [
            if (!_catalog.supportsCloning) _NoCloningBanner(tts: _catalog.tts),
            ..._sections(),
          ],
        );
    }
  }

  List<Widget> _sections() {
    // Stable kind order matching the server catalog.
    const order = ['finetuned', 'clone', 'preset'];
    final widgets = <Widget>[];
    for (final kind in order) {
      // Hide "Your clones" *and* "Fine-tuned" entirely on a backend that can't clone — both
      // need a cloning backend (f5_mlx / chatterbox), so on a preset-only backend (kokoro /
      // orpheus) everything there would be greyed-out as unavailable. Presets stay: they're
      // the only voices actionable there.
      if ((kind == 'clone' || kind == 'finetuned') && !_catalog.supportsCloning) continue;
      final group = _catalog.voices.where((v) => v.kind == kind).toList();
      if (group.isEmpty) continue;
      widgets.add(_SectionHeader('${_kindHeading(kind)} (${group.length})'));
      for (final v in group) {
        widgets.add(_VoiceTile(
          voice: v,
          onDelete: v.removable ? () => _deleteVoice(v) : null,
        ));
      }
    }
    if (widgets.isEmpty) {
      widgets.add(const Padding(
        padding: EdgeInsets.all(32),
        child: Center(child: Text('No voices available from this server.')),
      ));
    }
    return widgets;
  }

  static String _kindHeading(String kind) => switch (kind) {
        'finetuned' => 'Fine-tuned',
        'clone' => 'Your clones',
        'preset' => 'Presets',
        _ => kind,
      };
}

class _SectionHeader extends StatelessWidget {
  const _SectionHeader(this.text);
  final String text;

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.fromLTRB(16, 20, 16, 6),
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

class _VoiceTile extends StatelessWidget {
  const _VoiceTile({required this.voice, this.onDelete});
  final VoiceOption voice;
  final VoidCallback? onDelete;

  IconData get _icon => switch (voice.kind) {
        'clone' => Icons.person_outline,
        'finetuned' => Icons.tune,
        _ => Icons.graphic_eq,
      };

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    final subtitleParts = <String>[
      if (voice.emotion != null) voice.emotion!,
      if (!voice.available && voice.reason != null) voice.reason!,
    ];
    return ListTile(
      leading: CircleAvatar(
        backgroundColor:
            voice.available ? scheme.secondaryContainer : scheme.surfaceContainerHighest,
        child: Icon(_icon,
            color: voice.available ? scheme.onSecondaryContainer : scheme.outline),
      ),
      title: Text(voice.name),
      subtitle: subtitleParts.isEmpty ? null : Text(subtitleParts.join(' · ')),
      trailing: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          if (!voice.available)
            Padding(
              padding: const EdgeInsets.only(right: 4),
              child: Tooltip(
                message: voice.reason ?? 'Not available on this backend',
                child: Icon(Icons.block, size: 18, color: scheme.outline),
              ),
            ),
          if (onDelete != null)
            IconButton(
              icon: const Icon(Icons.delete_outline),
              tooltip: 'Delete',
              onPressed: onDelete,
            ),
        ],
      ),
    );
  }
}

class _NoCloningBanner extends StatelessWidget {
  const _NoCloningBanner({required this.tts});
  final String tts;

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    final backend = tts.isEmpty ? 'the active backend' : tts;
    return Container(
      margin: const EdgeInsets.all(12),
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: scheme.surfaceContainerHighest,
        borderRadius: BorderRadius.circular(8),
      ),
      child: Row(
        children: [
          Icon(Icons.info_outline, size: 18, color: scheme.onSurfaceVariant),
          const SizedBox(width: 10),
          Expanded(
            child: Text(
              "$backend can't speak or record clones. Switch to a cloning backend "
              '(f5_mlx on Mac, chatterbox on CUDA) to add and use your own voices.',
              style: TextStyle(fontSize: 13, color: scheme.onSurfaceVariant),
            ),
          ),
        ],
      ),
    );
  }
}

/// Bottom sheet to enroll a clone: name + consent + a sample (file upload or mic recording).
class _EnrollVoiceSheet extends StatefulWidget {
  const _EnrollVoiceSheet({required this.settings});
  final ConnectionSettings settings;

  @override
  State<_EnrollVoiceSheet> createState() => _EnrollVoiceSheetState();
}

class _EnrollVoiceSheetState extends State<_EnrollVoiceSheet> {
  final _nameCtrl = TextEditingController();
  final _recorder = VoiceRecorder();

  Uint8List? _sample;
  String _sampleLabel = '';
  bool _authorized = false;
  bool _recording = false;
  bool _submitting = false;
  String? _error;

  // Live recording timer.
  Timer? _ticker;
  final _stopwatch = Stopwatch();

  bool get _nameValid => _voiceNameRe.hasMatch(_nameCtrl.text.trim());
  bool get _canSubmit =>
      _nameValid && _authorized && _sample != null && !_submitting && !_recording;

  @override
  void dispose() {
    _ticker?.cancel();
    _nameCtrl.dispose();
    _recorder.dispose();
    super.dispose();
  }

  Future<void> _pickFile() async {
    setState(() => _error = null);
    try {
      final result = await FilePicker.platform.pickFiles(
        type: FileType.custom,
        allowedExtensions: ['wav'],
        withData: true,
      );
      final file = result?.files.single;
      final bytes = file?.bytes;
      if (bytes == null) return; // cancelled, or path-only result
      if (bytes.length > _maxSampleBytes) {
        setState(() => _error = 'That file is too large (max 10 MB).');
        return;
      }
      setState(() {
        _sample = bytes;
        _sampleLabel = '${file!.name} · ${_kb(bytes.length)}';
      });
    } catch (e) {
      setState(() => _error = 'Could not read the file: $e');
    }
  }

  Future<void> _toggleRecording() async {
    setState(() => _error = null);
    if (_recording) {
      final bytes = await _recorder.stop();
      _stopwatch.stop();
      _ticker?.cancel();
      setState(() {
        _recording = false;
        if (bytes != null && bytes.isNotEmpty) {
          _sample = bytes;
          _sampleLabel = 'Recording · ${_stopwatch.elapsed.inSeconds}s · ${_kb(bytes.length)}';
        } else {
          _error = 'Nothing was recorded.';
        }
      });
      return;
    }
    if (!await _recorder.hasPermission()) {
      setState(() => _error = 'Microphone permission is required to record.');
      return;
    }
    await _recorder.start();
    _stopwatch
      ..reset()
      ..start();
    _ticker = Timer.periodic(const Duration(milliseconds: 250), (_) {
      if (mounted) setState(() {});
    });
    setState(() {
      _recording = true;
      _sample = null;
      _sampleLabel = '';
    });
  }

  Future<void> _submit() async {
    setState(() {
      _submitting = true;
      _error = null;
    });
    final client = TokenClient(widget.settings);
    try {
      await client.cloneVoice(_nameCtrl.text.trim(), _sample!, authorized: true);
      if (mounted) Navigator.of(context).pop(true);
    } catch (e) {
      if (mounted) setState(() => _error = '$e');
    } finally {
      client.close();
      if (mounted) setState(() => _submitting = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    return Padding(
      padding: EdgeInsets.fromLTRB(20, 4, 20, 20 + MediaQuery.of(context).viewInsets.bottom),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text('Add a voice', style: Theme.of(context).textTheme.titleLarge),
          const SizedBox(height: 4),
          Text(
            'Clone a voice from a ~10-second sample. Speak naturally, in a quiet room.',
            style: TextStyle(fontSize: 13, color: scheme.onSurfaceVariant),
          ),
          const SizedBox(height: 20),
          TextField(
            controller: _nameCtrl,
            autocorrect: false,
            decoration: InputDecoration(
              labelText: 'Voice name',
              hintText: 'e.g. my_voice',
              border: const OutlineInputBorder(),
              prefixIcon: const Icon(Icons.badge_outlined),
              errorText: _nameCtrl.text.isNotEmpty && !_nameValid
                  ? 'Letters, digits, "-" and "_" only'
                  : null,
            ),
            onChanged: (_) => setState(() {}),
          ),
          const SizedBox(height: 16),
          Row(
            children: [
              Expanded(
                child: OutlinedButton.icon(
                  onPressed: _recording || _submitting ? null : _pickFile,
                  icon: const Icon(Icons.upload_file),
                  label: const Text('Upload .wav'),
                ),
              ),
              const SizedBox(width: 12),
              Expanded(
                child: OutlinedButton.icon(
                  onPressed: _submitting ? null : _toggleRecording,
                  icon: Icon(_recording ? Icons.stop : Icons.mic),
                  label: Text(_recording
                      ? 'Stop (${_stopwatch.elapsed.inSeconds}s)'
                      : 'Record'),
                  style: _recording
                      ? OutlinedButton.styleFrom(
                          foregroundColor: scheme.error,
                          side: BorderSide(color: scheme.error),
                        )
                      : null,
                ),
              ),
            ],
          ),
          if (_sampleLabel.isNotEmpty) ...[
            const SizedBox(height: 12),
            Row(
              children: [
                Icon(Icons.check_circle, size: 18, color: scheme.primary),
                const SizedBox(width: 8),
                Expanded(child: Text(_sampleLabel, style: const TextStyle(fontSize: 13))),
              ],
            ),
          ],
          const SizedBox(height: 8),
          CheckboxListTile(
            contentPadding: EdgeInsets.zero,
            controlAffinity: ListTileControlAffinity.leading,
            value: _authorized,
            onChanged: _submitting ? null : (v) => setState(() => _authorized = v ?? false),
            title: const Text("I'm authorized to use this voice",
                style: TextStyle(fontSize: 14)),
          ),
          if (_error != null) ...[
            const SizedBox(height: 4),
            Text(_error!, style: TextStyle(color: scheme.error, fontSize: 13)),
          ],
          const SizedBox(height: 12),
          SizedBox(
            width: double.infinity,
            child: FilledButton(
              onPressed: _canSubmit ? _submit : null,
              child: _submitting
                  ? const SizedBox(
                      height: 18, width: 18, child: CircularProgressIndicator(strokeWidth: 2))
                  : const Text('Clone voice'),
            ),
          ),
        ],
      ),
    );
  }

  static String _kb(int bytes) => '${(bytes / 1024).round()} KB';
}
