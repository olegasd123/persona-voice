import 'package:flutter/material.dart';

import '../models/persona.dart';
import '../services/token_client.dart';
import '../services/voice_session.dart';

/// The live call: connection status, mic toggle, a transcript, mid-call persona switch,
/// and hang-up. Drives a [VoiceSession] and rebuilds as it notifies.
class CallScreen extends StatefulWidget {
  const CallScreen({
    super.key,
    required this.session,
    required this.grant,
    required this.personas,
  });

  final VoiceSession session;
  final JoinGrant grant;
  final List<Persona> personas;

  @override
  State<CallScreen> createState() => _CallScreenState();
}

class _CallScreenState extends State<CallScreen> {
  VoiceSession get _session => widget.session;

  @override
  void initState() {
    super.initState();
    _session.addListener(_onChange);
    _session.connect(widget.grant);
  }

  void _onChange() {
    if (mounted) setState(() {});
  }

  Future<void> _hangUp() async {
    await _session.disconnect();
    if (mounted) Navigator.of(context).pop();
  }

  @override
  void dispose() {
    _session.removeListener(_onChange);
    _session.dispose();
    super.dispose();
  }

  String get _statusLabel => switch (_session.status) {
        SessionStatus.connecting => 'Connecting…',
        SessionStatus.connected => _session.agentSpeaking ? 'Speaking…' : 'Listening',
        SessionStatus.disconnected => 'Disconnected',
        SessionStatus.error => 'Error',
        SessionStatus.idle => 'Idle',
      };

  @override
  Widget build(BuildContext context) {
    final transcript = _session.transcript;
    return Scaffold(
      appBar: AppBar(
        title: Text(_session.persona.isEmpty ? 'Call' : _session.persona),
        actions: [
          if (widget.personas.length > 1)
            PopupMenuButton<String>(
              icon: const Icon(Icons.switch_account),
              tooltip: 'Switch persona',
              onSelected: _session.switchPersona,
              itemBuilder: (_) => widget.personas
                  .map((p) => PopupMenuItem<String>(value: p.id, child: Text(p.name)))
                  .toList(),
            ),
        ],
      ),
      body: Column(
        children: [
          _StatusBar(label: _statusLabel, speaking: _session.agentSpeaking),
          if (_session.errorMessage != null)
            Padding(
              padding: const EdgeInsets.all(12),
              child: Text(_session.errorMessage!,
                  style: TextStyle(color: Theme.of(context).colorScheme.error)),
            ),
          Expanded(
            child: transcript.isEmpty
                ? const Center(child: Text('Say something to start the conversation.'))
                : ListView.builder(
                    padding: const EdgeInsets.all(12),
                    itemCount: transcript.length,
                    itemBuilder: (_, i) => _TranscriptBubble(line: transcript[i]),
                  ),
          ),
          _Controls(
            micEnabled: _session.micEnabled,
            connected: _session.isConnected,
            onToggleMic: _session.toggleMic,
            onHangUp: _hangUp,
          ),
        ],
      ),
    );
  }
}

class _StatusBar extends StatelessWidget {
  const _StatusBar({required this.label, required this.speaking});
  final String label;
  final bool speaking;

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    return Container(
      width: double.infinity,
      color: speaking ? scheme.primaryContainer : scheme.surfaceContainerHighest,
      padding: const EdgeInsets.symmetric(vertical: 10, horizontal: 16),
      child: Row(
        children: [
          Icon(speaking ? Icons.graphic_eq : Icons.hearing, size: 18),
          const SizedBox(width: 8),
          Text(label),
        ],
      ),
    );
  }
}

class _TranscriptBubble extends StatelessWidget {
  const _TranscriptBubble({required this.line});
  final TranscriptLine line;

  @override
  Widget build(BuildContext context) {
    final isUser = line.speaker == 'You';
    final scheme = Theme.of(context).colorScheme;
    return Align(
      alignment: isUser ? Alignment.centerRight : Alignment.centerLeft,
      child: Container(
        margin: const EdgeInsets.symmetric(vertical: 4),
        padding: const EdgeInsets.symmetric(vertical: 8, horizontal: 12),
        decoration: BoxDecoration(
          color: isUser ? scheme.primaryContainer : scheme.secondaryContainer,
          borderRadius: BorderRadius.circular(12),
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(line.speaker, style: const TextStyle(fontSize: 11, fontWeight: FontWeight.bold)),
            Text(line.text + (line.isFinal ? '' : ' …')),
          ],
        ),
      ),
    );
  }
}

class _Controls extends StatelessWidget {
  const _Controls({
    required this.micEnabled,
    required this.connected,
    required this.onToggleMic,
    required this.onHangUp,
  });

  final bool micEnabled;
  final bool connected;
  final Future<bool> Function() onToggleMic;
  final Future<void> Function() onHangUp;

  @override
  Widget build(BuildContext context) {
    return SafeArea(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Row(
          mainAxisAlignment: MainAxisAlignment.spaceEvenly,
          children: [
            FloatingActionButton(
              heroTag: 'mic',
              onPressed: connected ? () => onToggleMic() : null,
              backgroundColor: micEnabled ? null : Colors.grey,
              child: Icon(micEnabled ? Icons.mic : Icons.mic_off),
            ),
            FloatingActionButton(
              heroTag: 'hangup',
              backgroundColor: Theme.of(context).colorScheme.error,
              onPressed: () => onHangUp(),
              child: const Icon(Icons.call_end),
            ),
          ],
        ),
      ),
    );
  }
}
