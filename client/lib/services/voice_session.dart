import 'dart:convert';

import 'package:flutter/foundation.dart';
import 'package:livekit_client/livekit_client.dart';

import 'token_client.dart';

enum SessionStatus { idle, connecting, connected, reconnecting, disconnected, error }

/// How the mic is driven during a call.
///
/// [openMic] keeps the mic live so the server's VAD decides turns (hands-free); the user can
/// still mute. [pushToTalk] keeps the mic muted and only un-mutes while the talk button is
/// held — useful in noisy rooms or to avoid the server hearing background speech.
enum MicMode { openMic, pushToTalk }

/// Human-readable status text. Pure (no [VoiceSession] needed) so it's unit-tested directly.
String sessionStatusLabel(SessionStatus status, {required bool agentSpeaking}) =>
    switch (status) {
      SessionStatus.connecting => 'Connecting…',
      SessionStatus.reconnecting => 'Reconnecting…',
      SessionStatus.connected => agentSpeaking ? 'Speaking…' : 'Listening',
      SessionStatus.disconnected => 'Disconnected',
      SessionStatus.error => 'Error',
      SessionStatus.idle => 'Idle',
    };

/// One line of conversation transcript, keyed by the LiveKit segment id so streamed
/// (interim → final) updates replace in place rather than appending duplicates.
class TranscriptLine {
  TranscriptLine({
    required this.id,
    required this.speaker,
    required this.text,
    required this.isFinal,
  });

  final String id;
  final String speaker; // "You" | "Assistant"
  String text;
  bool isFinal;
}

/// Wraps a LiveKit [Room] for one voice conversation: connect with a [JoinGrant], publish
/// the mic, surface transcripts + who's speaking, and switch persona mid-call via a data
/// message (the server agent's `on("data_received")` handler swaps to it — see
/// `orchestrator/agent.py`). Exposed as a [ChangeNotifier] so the UI rebuilds on changes.
class VoiceSession extends ChangeNotifier {
  Room? _room;
  EventsListener<RoomEvent>? _listener;

  SessionStatus status = SessionStatus.idle;
  String? errorMessage;
  bool micEnabled = false;
  bool agentSpeaking = false;
  String persona = '';
  MicMode micMode = MicMode.openMic;
  bool talking = false; // push-to-talk: true while the talk button is held

  final List<TranscriptLine> _transcript = [];
  final Map<String, TranscriptLine> _byId = {};
  List<TranscriptLine> get transcript => List.unmodifiable(_transcript);

  bool get isConnected => status == SessionStatus.connected;

  Future<void> connect(JoinGrant grant) async {
    if (status == SessionStatus.connecting || status == SessionStatus.connected) return;
    _setStatus(SessionStatus.connecting);
    persona = grant.persona;
    try {
      final room = Room(
        roomOptions: const RoomOptions(
          adaptiveStream: true,
          dynacast: true,
          // It's a voice call: route audio to the loudspeaker by default (a wired or
          // Bluetooth headset still overrides this at the OS level).
          defaultAudioOutputOptions: AudioOutputOptions(speakerOn: true),
        ),
      );
      final listener = room.createListener()
        ..on<RoomConnectedEvent>((_) => _setStatus(SessionStatus.connected))
        ..on<RoomReconnectingEvent>((_) => _setStatus(SessionStatus.reconnecting))
        ..on<RoomResumingEvent>((_) => _setStatus(SessionStatus.reconnecting))
        ..on<RoomReconnectedEvent>((_) => _setStatus(SessionStatus.connected))
        ..on<RoomDisconnectedEvent>((_) => _setStatus(SessionStatus.disconnected))
        ..on<TranscriptionEvent>(_onTranscription)
        ..on<ActiveSpeakersChangedEvent>(_onActiveSpeakers);

      _room = room;
      _listener = listener;

      await room.connect(grant.url, grant.token);
      // Open-mic starts live; push-to-talk starts muted until the user holds to talk.
      micEnabled = micMode == MicMode.openMic;
      await room.localParticipant?.setMicrophoneEnabled(micEnabled);
      // The agent selects its persona from a data message on connect.
      await _sendPersona(grant.persona);
      _setStatus(SessionStatus.connected);
    } catch (e) {
      errorMessage = e.toString();
      _setStatus(SessionStatus.error);
      await _teardown();
    }
  }

  /// Toggle the mic (mute / unmute) in open-mic mode. Returns the new enabled state.
  Future<bool> toggleMic() async {
    await _setMicEnabled(!micEnabled);
    return micEnabled;
  }

  /// Switch between open-mic (VAD) and push-to-talk. Open-mic goes live immediately;
  /// push-to-talk mutes until the user holds the talk button ([setTalking]).
  Future<void> setMicMode(MicMode mode) async {
    if (mode == micMode) return;
    micMode = mode;
    talking = false;
    await _setMicEnabled(mode == MicMode.openMic);
  }

  /// Push-to-talk: enable the mic while the talk button is held, mute on release.
  /// A no-op in open-mic mode.
  Future<void> setTalking(bool held) async {
    if (micMode != MicMode.pushToTalk || held == talking) return;
    talking = held;
    await _setMicEnabled(held);
  }

  Future<void> _setMicEnabled(bool enabled) async {
    micEnabled = enabled;
    notifyListeners();
    await _room?.localParticipant?.setMicrophoneEnabled(enabled);
  }

  /// Switch persona without dropping the call (keeps conversation history server-side).
  Future<void> switchPersona(String personaId) async {
    if (personaId == persona) return;
    persona = personaId;
    await _sendPersona(personaId);
    notifyListeners();
  }

  Future<void> _sendPersona(String personaId) async {
    final lp = _room?.localParticipant;
    if (lp == null || personaId.isEmpty) return;
    final data = utf8.encode(jsonEncode({'persona': personaId}));
    await lp.publishData(data, reliable: true);
  }

  void _onTranscription(TranscriptionEvent event) {
    final speaker =
        event.participant.identity == _room?.localParticipant?.identity ? 'You' : 'Assistant';
    for (final seg in event.segments) {
      final existing = _byId[seg.id];
      if (existing == null) {
        final line = TranscriptLine(
          id: seg.id,
          speaker: speaker,
          text: seg.text,
          isFinal: seg.isFinal,
        );
        _byId[seg.id] = line;
        _transcript.add(line);
      } else {
        existing.text = seg.text;
        existing.isFinal = seg.isFinal;
      }
    }
    notifyListeners();
  }

  void _onActiveSpeakers(ActiveSpeakersChangedEvent event) {
    final localId = _room?.localParticipant?.identity;
    final speaking = event.speakers.any((p) => p.identity != localId);
    if (speaking != agentSpeaking) {
      agentSpeaking = speaking;
      notifyListeners();
    }
  }

  Future<void> disconnect() async {
    await _teardown();
    _setStatus(SessionStatus.disconnected);
  }

  Future<void> _teardown() async {
    try {
      await _listener?.dispose();
      await _room?.disconnect();
      await _room?.dispose();
    } catch (e) {
      debugPrint('voice session teardown error: $e');
    } finally {
      _listener = null;
      _room = null;
      micEnabled = false;
      agentSpeaking = false;
      talking = false;
    }
  }

  void _setStatus(SessionStatus s) {
    status = s;
    notifyListeners();
  }

  @override
  void dispose() {
    _teardown();
    super.dispose();
  }
}
