import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';

import '../models/connection_settings.dart';
import '../models/persona.dart';
import '../models/session_options.dart';
import '../services/queue_controller.dart';
import '../services/token_client.dart';
import '../services/voice_session.dart';
import 'call_screen.dart';

/// Shown when every session is busy: the user waits here (no LiveKit connection yet) while the
/// client quietly re-asks for a slot, then is dropped straight into the call the moment one frees.
/// This is the "queue" — a pre-connection wait, so no session is held while waiting.
class QueueScreen extends StatefulWidget {
  const QueueScreen({
    super.key,
    required this.settings,
    required this.persona,
    required this.personas,
    this.options = const SessionOptions(),
    this.initialMicMode = MicMode.openMic,
    this.initialRetryAfter,
  });

  final ConnectionSettings settings;
  final Persona persona;
  final List<Persona> personas;
  final SessionOptions options;
  final MicMode initialMicMode;

  /// The `Retry-After` from the busy response that sent us here (seconds); null → a default.
  final int? initialRetryAfter;

  @override
  State<QueueScreen> createState() => _QueueScreenState();
}

class _QueueScreenState extends State<QueueScreen> {
  late final TokenClient _client = TokenClient(widget.settings);
  QueueController? _controller;
  String? _error;

  @override
  void initState() {
    super.initState();
    _start();
  }

  @override
  void dispose() {
    _controller?.dispose();
    _client.close();
    super.dispose();
  }

  void _start() {
    _controller?.dispose();
    final controller = _controller = QueueController(
      request: () =>
          _client.requestToken(persona: widget.persona.id, options: widget.options),
      initialRetryAfter: widget.initialRetryAfter ?? 10,
    );
    setState(() => _error = null);
    _drive(controller);
  }

  Future<void> _drive(QueueController controller) async {
    try {
      final grant = await controller.run();
      if (!mounted || grant == null) return; // cancelled (run returns null)
      // Replace the queue route so Back from the call returns home, not to the queue.
      Navigator.of(context).pushReplacement(MaterialPageRoute(
        builder: (_) => CallScreen(
          session: VoiceSession(),
          grant: grant,
          personas: widget.personas,
          options: widget.options,
          initialMicMode: widget.initialMicMode,
        ),
      ));
    } catch (e) {
      if (mounted) setState(() => _error = friendlyTokenError(e));
    }
  }

  void _cancel() {
    _controller?.cancel();
    Navigator.of(context).pop();
  }

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    return Scaffold(
      appBar: AppBar(title: Text(widget.persona.name)),
      body: Center(
        child: Padding(
          padding: const EdgeInsets.all(24),
          child: _error != null
              ? _ErrorBody(message: _error!, onRetry: _start, onClose: _cancel)
              : _WaitingBody(
                  persona: widget.persona,
                  status: _controller!.status,
                  scheme: scheme,
                  onCancel: _cancel,
                ),
        ),
      ),
    );
  }
}

class _WaitingBody extends StatelessWidget {
  const _WaitingBody({
    required this.persona,
    required this.status,
    required this.scheme,
    required this.onCancel,
  });

  final Persona persona;
  final ValueListenable<QueueStatus> status;
  final ColorScheme scheme;
  final VoidCallback onCancel;

  @override
  Widget build(BuildContext context) {
    return Column(
      mainAxisSize: MainAxisSize.min,
      children: [
        const SizedBox(
          width: 48,
          height: 48,
          child: CircularProgressIndicator(strokeWidth: 3),
        ),
        const SizedBox(height: 24),
        Text(
          'All lines are busy',
          style: Theme.of(context).textTheme.titleMedium,
          textAlign: TextAlign.center,
        ),
        const SizedBox(height: 8),
        Text(
          "You're in line for ${persona.name}. We'll connect you the moment a slot frees — "
          'no need to keep tapping.',
          textAlign: TextAlign.center,
          style: TextStyle(color: scheme.onSurfaceVariant),
        ),
        const SizedBox(height: 16),
        ValueListenableBuilder<QueueStatus>(
          valueListenable: status,
          builder: (_, s, _) => Text(
            s.waited == Duration.zero
                ? 'Waiting…'
                : 'Waiting ${_format(s.waited)} · checking every ${s.retryAfter}s',
            style: TextStyle(color: scheme.onSurfaceVariant),
          ),
        ),
        const SizedBox(height: 24),
        OutlinedButton.icon(
          onPressed: onCancel,
          icon: const Icon(Icons.close),
          label: const Text('Cancel'),
        ),
      ],
    );
  }

  static String _format(Duration d) {
    final m = d.inMinutes;
    final s = d.inSeconds % 60;
    return m > 0 ? '${m}m ${s}s' : '${s}s';
  }
}

class _ErrorBody extends StatelessWidget {
  const _ErrorBody({required this.message, required this.onRetry, required this.onClose});

  final String message;
  final VoidCallback onRetry;
  final VoidCallback onClose;

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    return Column(
      mainAxisSize: MainAxisSize.min,
      children: [
        Icon(Icons.error_outline, size: 48, color: scheme.error),
        const SizedBox(height: 16),
        Text(message, textAlign: TextAlign.center),
        const SizedBox(height: 24),
        Row(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            OutlinedButton(onPressed: onClose, child: const Text('Back')),
            const SizedBox(width: 12),
            FilledButton(onPressed: onRetry, child: const Text('Try again')),
          ],
        ),
      ],
    );
  }
}
