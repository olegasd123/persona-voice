import 'dart:async';
import 'dart:convert';

import 'package:flutter/foundation.dart';
// livekit_client also exports a `SessionOptions` (for its agent API); hide it so the name
// refers to our per-call overrides model.
import 'package:livekit_client/livekit_client.dart' hide SessionOptions;

import '../models/session_options.dart';
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
///
/// [agentReady] distinguishes "the LiveKit room connected" from "the assistant has actually
/// joined and is ready to respond". The agent only joins (publishing its audio track) once the
/// server worker has finished warming up its models — so until then we say the assistant is
/// still connecting, rather than the misleading "Listening".
String sessionStatusLabel(
  SessionStatus status, {
  required bool agentSpeaking,
  bool interrupted = false,
  bool agentReady = true,
}) {
  if (interrupted && status == SessionStatus.connected) return 'Paused (interrupted)';
  return switch (status) {
    SessionStatus.connecting => 'Connecting…',
    SessionStatus.reconnecting => 'Reconnecting…',
    SessionStatus.connected =>
      agentReady ? (agentSpeaking ? 'Speaking…' : 'Listening') : 'Connecting to assistant…',
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

  // --- Warm-up watchdog -------------------------------------------------------------------
  // The agent only joins (and publishes its track, flipping [agentReady]) once the server
  // worker is warm. If we join a room while the worker is still prewarming, automatic dispatch
  // can miss us and isn't retried for an *existing* room — so the room would sit there with no
  // assistant forever (the user's only recourse being to leave and re-enter). This watchdog
  // closes that gap: if the assistant doesn't show within [_agentJoinGrace], we rejoin a *fresh*
  // room (a new token = a new room, which re-triggers dispatch), up to [_maxAgentRetries] times.

  /// How long after the room connects we wait for the assistant before assuming dispatch was
  /// missed and rejoining. A warm agent joins in ~1 s; this tolerates a slow join without nagging.
  static const _agentJoinGrace = Duration(seconds: 12);

  /// How many automatic fresh-room rejoins to attempt before giving up and surfacing a note.
  static const _maxAgentRetries = 2;

  /// Re-mints a fresh grant (new room) for a rejoin; null disables the watchdog (e.g. unit
  /// tests, or a caller that didn't supply one). Stored from [connect], survives a rejoin,
  /// cleared on full teardown.
  Future<JoinGrant> Function()? _regrant;
  Timer? _agentWaitTimer;
  int _agentRetries = 0;

  SessionStatus status = SessionStatus.idle;
  String? errorMessage;
  bool micEnabled = false;
  bool agentSpeaking = false;

  /// Whether the assistant has joined the room and published its audio track — i.e. the server
  /// finished warming up and is ready to respond. The room can be [SessionStatus.connected]
  /// while this is still false (the worker is loading models / waiting for vLLM); the UI shows
  /// "Connecting to assistant…" and the open mic is held muted until this flips true.
  bool agentReady = false;
  String persona = '';

  /// The per-session overrides (voice / CEFR / demeanor) chosen for this call. Already applied
  /// server-side from the token metadata; re-sent in the persona data message so a mid-call
  /// persona switch keeps them.
  SessionOptions options = const SessionOptions();
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

  /// Connect to the room described by [grant]. [regrant] (optional) re-mints a fresh grant so
  /// the warm-up watchdog can rejoin a new room if the assistant never shows; omit it to keep
  /// the old behaviour (wait indefinitely for the agent).
  Future<void> connect(JoinGrant grant,
      {SessionOptions options = const SessionOptions(),
      Future<JoinGrant> Function()? regrant}) async {
    if (status == SessionStatus.connecting || status == SessionStatus.connected) return;
    // A fresh user-initiated connect: reset the rejoin budget and remember how to re-mint.
    _agentRetries = 0;
    _regrant = regrant;
    this.options = options;
    await _openRoom(grant);
  }

  /// Open (or re-open) the LiveKit room for [grant] and bring the call to `connected`. Shared by
  /// [connect] and the watchdog rejoin: the session-scoped pieces (the native system call and the
  /// audio-interruption monitor) are started once and survive a rejoin, so swapping rooms doesn't
  /// flicker the OS call UI — only the LiveKit room itself is replaced.
  Future<void> _openRoom(JoinGrant grant) async {
    _setStatus(SessionStatus.connecting);
    persona = grant.persona;
    agentReady = false;
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
      // The agent joins as a remote participant and publishes its "assistant-voice" track only
      // once the server worker is warm (see orchestrator/agent.py). We treat that track
      // subscription as "assistant ready"; losing the track / participant flips it back.
      final listener = room.createListener()
        ..on<RoomConnectedEvent>((_) => _setStatus(SessionStatus.connected))
        ..on<RoomReconnectingEvent>((_) {
          setAgentReady(false);
          _setStatus(SessionStatus.reconnecting);
        })
        ..on<RoomResumingEvent>((_) {
          setAgentReady(false);
          _setStatus(SessionStatus.reconnecting);
        })
        ..on<RoomReconnectedEvent>((_) => _setStatus(SessionStatus.connected))
        ..on<RoomDisconnectedEvent>((_) {
          setAgentReady(false);
          _setStatus(SessionStatus.disconnected);
        })
        ..on<TrackSubscribedEvent>((_) => setAgentReady(true))
        ..on<TrackUnsubscribedEvent>((_) => setAgentReady(false))
        ..on<ParticipantDisconnectedEvent>((_) => setAgentReady(false))
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
      // [onCallControlEvent]. Best-effort — a host without the native layer no-ops. Started only
      // on the first open; a watchdog rejoin swaps the room underneath the same system call.
      final systemCall = _systemCall ??= PlatformSystemCallController.instance;
      _callSub ??= systemCall.events.listen(onCallControlEvent);
      final firstOpen = _callId == null;
      if (firstOpen) {
        _systemEndedCall = false;
        final callName = grant.persona.isEmpty ? 'Persona Voice' : grant.persona;
        await systemCall.startCall(
          callId: _callId = generateCallId(),
          displayName: callName,
          handle: grant.persona.isEmpty ? 'persona' : grant.persona,
        );
      }

      await room.connect(grant.url, grant.token);
      // Hold the mic muted until the assistant is actually ready (see [setAgentReady]). Open-mic
      // would otherwise go live before the agent joins, so the user's first words would be lost
      // while the server is still warming up. Push-to-talk is muted until the user holds anyway.
      micEnabled = false;
      await room.localParticipant?.setMicrophoneEnabled(false);
      // The agent selects its persona from a data message on connect.
      await _sendPersona(grant.persona);
      if (firstOpen) await systemCall.reportConnected(_callId!);
      _setStatus(SessionStatus.connected);
      // Start the warm-up watchdog: if the assistant doesn't join before the grace elapses, the
      // dispatch was likely missed (worker still prewarming) — rejoin a fresh room. No-op once
      // [agentReady] is already true (a warm agent that joined during connect) or without a regrant.
      _armAgentWait();
    } catch (e) {
      errorMessage = e.toString();
      _setStatus(SessionStatus.error);
      await _teardown();
    }
  }

  /// Arm the warm-up watchdog (idempotent — replaces any pending timer). A no-op when the
  /// assistant is already present or no [_regrant] was supplied.
  void _armAgentWait() {
    _agentWaitTimer?.cancel();
    _agentWaitTimer = null;
    if (agentReady || _regrant == null) return;
    _agentWaitTimer = Timer(_agentJoinGrace, () => unawaited(_onAgentWaitElapsed()));
  }

  /// The grace window elapsed with no assistant on the line. Rejoin a fresh room (which
  /// re-triggers agent dispatch), up to [_maxAgentRetries]; after that, surface a gentle note and
  /// stop — the call stays up so the user can keep waiting or hang up.
  Future<void> _onAgentWaitElapsed() async {
    _agentWaitTimer = null;
    // Only act if we're still sitting connected with no assistant (it may have just joined, or
    // the call may have moved to reconnecting/disconnected meanwhile).
    if (status != SessionStatus.connected || agentReady) return;
    final regrant = _regrant;
    if (regrant == null) return;
    if (_agentRetries >= _maxAgentRetries) {
      errorMessage = "The assistant isn't responding. Try hanging up and calling again.";
      notifyListeners();
      return;
    }
    _agentRetries++;
    errorMessage = null;
    // Drop just the dead room (keep the OS call + audio monitor alive) and rejoin a fresh one.
    await _teardownRoom();
    _setStatus(SessionStatus.reconnecting);
    try {
      final grant = await regrant();
      // The user may have hung up while we were re-minting; bail if we're no longer rejoining.
      if (status != SessionStatus.reconnecting) return;
      await _openRoom(grant);
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
    // Carry the session overrides alongside the persona so a mid-call persona switch keeps the
    // chosen voice / CEFR / demeanor (the agent's data handler applies both — agent.py).
    final payload = <String, dynamic>{'persona': personaId, ...options.toWireMap()};
    final data = utf8.encode(jsonEncode(payload));
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

  /// Record whether the assistant is present and ready (its audio track is live). Public for
  /// unit testing (the room's track/participant events funnel here). On the transition to
  /// ready, an open mic that was held muted during warm-up is brought live so the user can
  /// start talking — but not while an OS interruption is active, and never in push-to-talk
  /// (which stays muted until the user holds). Idempotent.
  @visibleForTesting
  void setAgentReady(bool ready) {
    if (ready == agentReady) return;
    agentReady = ready;
    if (ready) {
      // The assistant joined — stop the warm-up watchdog, reset the rejoin budget, and clear any
      // "not responding" note left from a prior wait.
      _agentWaitTimer?.cancel();
      _agentWaitTimer = null;
      _agentRetries = 0;
      errorMessage = null;
    }
    if (ready && micMode == MicMode.openMic && !interrupted && !micEnabled) {
      unawaited(_setMicEnabled(true));
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

  /// Tear down only the LiveKit room (listener + room), leaving the native system call and the
  /// audio-interruption monitor intact. Used by the watchdog rejoin so swapping rooms doesn't end
  /// and restart the OS call. The full [_teardown] is for ending the whole session.
  Future<void> _teardownRoom() async {
    _agentWaitTimer?.cancel();
    _agentWaitTimer = null;
    try {
      await _listener?.dispose();
      await _room?.disconnect();
      await _room?.dispose();
    } catch (e) {
      debugPrint('voice session room teardown error: $e');
    } finally {
      _listener = null;
      _room = null;
      micEnabled = false;
      agentSpeaking = false;
      agentReady = false;
    }
  }

  Future<void> _teardown() async {
    _agentWaitTimer?.cancel();
    _agentWaitTimer = null;
    _regrant = null;
    _agentRetries = 0;
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
      agentReady = false;
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
