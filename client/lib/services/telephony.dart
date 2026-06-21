import 'dart:async';
import 'dart:math';

import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';

/// Presents a live [VoiceSession] *as a native system call* — the "fat server, thin client"
/// idea taken to the OS: CallKit on iOS and a self-managed ConnectionService on Android, so the
/// conversation shows up in the system call UI (lock-screen controls, the OS call list, audio
/// routing arbitration) and the system end/mute buttons drive the session.
///
/// This is distinct from the audio-session monitor (see `audio_session.dart`), which reacts to
/// being interrupted *by* another call. Here we register *our* conversation as the call.
///
/// Two channels, mirroring the audio-session split:
///  - a [MethodChannel] (`persona_voice/telephony`) for app → OS commands
///    (`startCall`/`reportConnected`/`endCall`/`setMuted`);
///  - an [EventChannel] (`persona_voice/telephony_events`) for OS → app actions the user took on
///    the system UI ([EndCallRequested], [MuteRequested]).

/// An action the user took on the OS call UI, surfaced over the telephony event channel.
sealed class CallControlEvent {
  const CallControlEvent();
}

/// The user ended the call from the system UI (lock screen, call list, CarPlay). The session
/// should hang up.
class EndCallRequested extends CallControlEvent {
  const EndCallRequested();

  @override
  bool operator ==(Object other) => other is EndCallRequested;

  @override
  int get hashCode => (EndCallRequested).hashCode;
}

/// The user toggled mute on the system call UI. [muted] is the requested state; the session
/// mirrors it onto the mic.
class MuteRequested extends CallControlEvent {
  const MuteRequested({required this.muted});

  final bool muted;

  @override
  bool operator ==(Object other) => other is MuteRequested && other.muted == muted;

  @override
  int get hashCode => Object.hash(MuteRequested, muted);
}

/// Decode one telephony event-channel payload into a [CallControlEvent], or `null` for an
/// unrecognized shape (defensive: a future native event we don't model is dropped, not fatal).
/// Pure, so it's unit-tested without the platform channel.
CallControlEvent? parseCallControlEvent(Object? raw) {
  if (raw is! Map) return null;
  switch (raw['event']) {
    case 'endCall':
      return const EndCallRequested();
    case 'setMuted':
      return MuteRequested(muted: raw['muted'] == true);
    default:
      return null;
  }
}

/// Generate a UUID-v4 string used as the call id. CallKit keys calls by `UUID`, and Android's
/// Telecom stack is happy with any stable id, so producing the id in Dart keeps both native
/// sides trivial (`UUID(uuidString:)` / pass-through) and lets the session own the lifecycle.
String generateCallId([Random? random]) {
  final r = random ?? Random.secure();
  final b = List<int>.generate(16, (_) => r.nextInt(256));
  b[6] = (b[6] & 0x0f) | 0x40; // version 4
  b[8] = (b[8] & 0x3f) | 0x80; // RFC 4122 variant
  String hex(int x) => x.toRadixString(16).padLeft(2, '0');
  final h = b.map(hex).toList();
  return '${h.sublist(0, 4).join()}-${h.sublist(4, 6).join()}-'
      '${h.sublist(6, 8).join()}-${h.sublist(8, 10).join()}-${h.sublist(10, 16).join()}';
}

/// Drives the native system-call presentation. An interface so [VoiceSession] can be unit-tested
/// with a fake (the real impl touches platform channels that don't exist on the test host).
abstract interface class SystemCallController {
  /// OS → app actions (end / mute) the user took on the system call UI.
  Stream<CallControlEvent> get events;

  /// Tell the OS a call has started (outgoing/self-managed) so it appears in the call UI.
  Future<void> startCall({
    required String callId,
    required String displayName,
    required String handle,
  });

  /// Mark the call connected (stops the "connecting…" state in the system UI).
  Future<void> reportConnected(String callId);

  /// Tear the system call down when the app hangs up.
  Future<void> endCall(String callId);

  /// Reflect the app's mic mute state onto the system call UI's mute button.
  Future<void> setMuted(String callId, bool muted);
}

/// Real [SystemCallController] over the platform channels. Commands swallow
/// [MissingPluginException] so a host without the native layer (unit-test VM, an unsupported
/// platform) degrades to a no-op rather than throwing.
class PlatformSystemCallController implements SystemCallController {
  PlatformSystemCallController._(this._methods, this._eventChannel);

  /// Shared instance over the agreed channel names. Tests inject a fake into [VoiceSession]
  /// instead of touching this.
  static final PlatformSystemCallController instance = PlatformSystemCallController._(
    const MethodChannel('persona_voice/telephony'),
    const EventChannel('persona_voice/telephony_events'),
  );

  final MethodChannel _methods;
  final EventChannel _eventChannel;
  Stream<CallControlEvent>? _events;

  @override
  Stream<CallControlEvent> get events => _events ??= _eventChannel
      .receiveBroadcastStream()
      .map(parseCallControlEvent)
      .where((e) => e != null)
      .cast<CallControlEvent>();

  @override
  Future<void> startCall({
    required String callId,
    required String displayName,
    required String handle,
  }) =>
      _invoke('startCall', {'callId': callId, 'displayName': displayName, 'handle': handle});

  @override
  Future<void> reportConnected(String callId) => _invoke('reportConnected', {'callId': callId});

  @override
  Future<void> endCall(String callId) => _invoke('endCall', {'callId': callId});

  @override
  Future<void> setMuted(String callId, bool muted) =>
      _invoke('setMuted', {'callId': callId, 'muted': muted});

  Future<void> _invoke(String method, Map<String, Object?> args) async {
    try {
      await _methods.invokeMethod<void>(method, args);
    } on MissingPluginException {
      // No native telephony layer on this host (e.g. test VM) — best-effort, ignore.
    } on PlatformException catch (e) {
      debugPrint('telephony $method failed: $e');
    }
  }
}
