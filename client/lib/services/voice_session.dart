import 'dart:async';
import 'dart:convert';

import 'package:flutter/foundation.dart';
import 'package:livekit_client/livekit_client.dart';

import 'audio_session.dart';
import 'telephony.dart';
import 'token_client.dart';

enum SessionStatus { idle, connecting, connected, reconnecting, disconnected, error }

/// How the mic is driven during a call.
///
/// [openMic] keeps the mic live so the server's VAD decides turns (hands-free); the user can
/// still mute. [pushToTalk] keeps the mic muted and only un-mutes while the talk button is
/// held — useful in noisy rooms or to avoid the server hearing background speech.
enum MicMode { openMic, pushToTalk }

/// Human-readable status text. Pure (no [VoiceSession] needed) so it's unit-tested directly.
/// When [interrupted] (an incoming call / Siri grabbed the audio session), that takes
/// precedence over the normal connected labels since the mic is muted until it clears.
String sessionStatusLabel(
  SessionStatus status, {
  required bool agentSpeaking,
  bool interrupted = false,
}) {
  if (interrupted && status == SessionStatus.connected) return 'Paused (interrupted)';
  return switch (status) {
    SessionStatus.connecting => 'Connecting…',
    SessionStatus.reconnecting => 'Reconnecting…',
    SessionStatus.connected => agentSpeaking ? 'Speaking…' : 'Listening',
    SessionStatus.disconnected => 'Disconnected',
    SessionStatus.error => 'Error',
    SessionStatus.idle => 'Idle',
  };
}

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
  /// [audioEvents] lets tests feed synthetic interruption/route events; in the app it
  /// defaults to the native monitor ([AudioInterruptions.instance]). [systemCall] drives the
  /// native system-call (CallKit / ConnectionService) presentation; injected in tests, else
  /// lazily defaults to [PlatformSystemCallController.instance] on first connect.
  VoiceSession({Stream<AudioEvent>? audioEvents, SystemCallController? systemCall})
      : _audioEvents = audioEvents,
        _systemCall = systemCall;

  final Stream<AudioEvent>? _audioEvents;
  StreamSubscription<AudioEvent>? _audioSub;

  SystemCallController? _systemCall;
  StreamSubscription<CallControlEvent>? _callSub;
  String? _callId;
  // Set when the system call UI ended the call, so teardown doesn't redundantly re-report it.
  bool _systemEndedCall = false;

  Room? _room;
  EventsListener<RoomEvent>? _listener;

  SessionStatus status = SessionStatus.idle;
  String? errorMessage;
  bool micEnabled = false;
  bool agentSpeaking = false;
  String persona = '';
  MicMode micMode = MicMode.openMic;
  bool talking = false; // push-to-talk: true while the talk button is held

  /// An OS audio interruption (incoming call, Siri, alarm) is in effect; the mic is muted
  /// until it clears. Surfaced in the status line.
  bool interrupted = false;
  // Whether the mic was live just before the interruption, so we only auto-resume a mic the
  // user actually had on (not one they'd muted).
  bool _micBeforeInterruption = false;

  /// Where the OS is currently routing audio (loudspeaker, headset, Bluetooth…). Observed,
  /// not controlled — useful for the UI and diagnostics.
  AudioRoute route = AudioRoute.speaker;

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
      // Start watching for OS audio interruptions (incoming calls) + route changes for the
      // life of the call. Lazily listened, so a never-connected session never touches the
      // platform channel.
      _audioSub ??= (_audioEvents ?? AudioInterruptions.instance.events).listen(onAudioEvent);

      // Present this conversation as a native system call (CallKit / ConnectionService): it
      // appears in the OS call UI and its end/mute buttons drive the session via
      // [onCallControlEvent]. Best-effort — a host without the native layer no-ops.
      final systemCall = _systemCall ??= PlatformSystemCallController.instance;
      _callSub ??= systemCall.events.listen(onCallControlEvent);
      _systemEndedCall = false;
      final callId = _callId = generateCallId();
      final callName = grant.persona.isEmpty ? 'Persona Voice' : grant.persona;
      await systemCall.startCall(
        callId: callId,
        displayName: callName,
        handle: grant.persona.isEmpty ? 'persona' : grant.persona,
      );

      await room.connect(grant.url, grant.token);
      // Open-mic starts live; push-to-talk starts muted until the user holds to talk.
      micEnabled = micMode == MicMode.openMic;
      await room.localParticipant?.setMicrophoneEnabled(micEnabled);
      // The agent selects its persona from a data message on connect.
      await _sendPersona(grant.persona);
      await systemCall.reportConnected(callId);
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

  Future<void> _setMicEnabled(bool enabled, {bool pushToSystem = true}) async {
    micEnabled = enabled;
    notifyListeners();
    await _room?.localParticipant?.setMicrophoneEnabled(enabled);
    // Keep the system call UI's mute button in sync with the actual mic. Skipped when the
    // change *originated* from that UI ([onCallControlEvent]) so the two can't ping-pong.
    if (pushToSystem) {
      await _systemCall?.setMuted(_callId ?? '', !enabled);
    }
  }

  /// React to a user action on the native system call UI. Public for unit testing (the
  /// [connect] subscription funnels here):
  ///  - [EndCallRequested] (system end button / lock screen) hangs up the whole session;
  ///  - [MuteRequested] mirrors the mute onto the mic — un-muting re-engages the mic only in
  ///    open-mic mode; in push-to-talk the user re-engages by holding the talk button.
  @visibleForTesting
  Future<void> onCallControlEvent(CallControlEvent event) async {
    switch (event) {
      case EndCallRequested():
        _systemEndedCall = true;
        await disconnect();
      case MuteRequested(:final muted):
        if (muted) {
          talking = false;
          await _setMicEnabled(false, pushToSystem: false);
        } else if (micMode == MicMode.openMic) {
          await _setMicEnabled(true, pushToSystem: false);
        }
    }
  }

  /// React to a native audio-session event. Public for unit testing (the subscription in
  /// [connect] funnels here):
  ///  - [InterruptionBegan] mutes the mic (remembering whether it was live) and marks the
  ///    call paused;
  ///  - [InterruptionEnded] auto-resumes only a mic that was live, in open-mic mode, when the
  ///    OS says it's appropriate — push-to-talk stays muted so the user re-engages by holding;
  ///  - [RouteChanged] just records the new output route.
  @visibleForTesting
  Future<void> onAudioEvent(AudioEvent event) async {
    switch (event) {
      case InterruptionBegan():
        if (interrupted) return;
        interrupted = true;
        _micBeforeInterruption = micEnabled;
        talking = false;
        await _setMicEnabled(false);
      case InterruptionEnded(:final shouldResume):
        if (!interrupted) return;
        interrupted = false;
        if (shouldResume && _micBeforeInterruption && micMode == MicMode.openMic) {
          await _setMicEnabled(true);
        } else {
          notifyListeners();
        }
      case RouteChanged(:final route):
        if (route == this.route) return;
        this.route = route;
        notifyListeners();
    }
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
      // End the system call — unless its own UI already ended it (no redundant report).
      if (_callId != null && !_systemEndedCall) {
        await _systemCall?.endCall(_callId!);
      }
      await _callSub?.cancel();
      await _audioSub?.cancel();
      await _listener?.dispose();
      await _room?.disconnect();
      await _room?.dispose();
    } catch (e) {
      debugPrint('voice session teardown error: $e');
    } finally {
      _callSub = null;
      _callId = null;
      _systemEndedCall = false;
      _audioSub = null;
      _listener = null;
      _room = null;
      micEnabled = false;
      agentSpeaking = false;
      talking = false;
      interrupted = false;
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
