import 'dart:async';

import 'package:flutter/services.dart';

/// Where the OS is currently sending call audio. Surfaced for the UI (and so push-to-talk /
/// open-mic decisions can stay sensible when a headset comes and goes); the actual routing is
/// done by LiveKit/WebRTC + the OS, we only observe it.
enum AudioRoute { speaker, receiver, headphones, bluetooth, carAudio, unknown }

/// A native audio-session event surfaced over the platform channel
/// (`persona_voice/audio_session`). The native side (iOS `AVAudioSession` notifications /
/// Android `AudioManager` focus + device callbacks) normalizes its events into these.
sealed class AudioEvent {
  const AudioEvent();
}

/// The OS took over the audio session — an incoming phone/VoIP call, Siri, an alarm. The mic
/// is effectively dead and playback is ducked/stopped until an [InterruptionEnded] arrives.
class InterruptionBegan extends AudioEvent {
  const InterruptionBegan();

  @override
  bool operator ==(Object other) => other is InterruptionBegan;

  @override
  int get hashCode => (InterruptionBegan).hashCode;
}

/// The interruption is over. [shouldResume] mirrors the OS hint that resuming audio is
/// appropriate (iOS `AVAudioSession.InterruptionOptions.shouldResume`; Android maps a regained
/// `AUDIOFOCUS_GAIN` to `true`). When `false`, leave the mic muted and let the user re-engage.
class InterruptionEnded extends AudioEvent {
  const InterruptionEnded({required this.shouldResume});

  final bool shouldResume;

  @override
  bool operator ==(Object other) =>
      other is InterruptionEnded && other.shouldResume == shouldResume;

  @override
  int get hashCode => Object.hash(InterruptionEnded, shouldResume);
}

/// The audio output route changed: a wired/Bluetooth headset was connected or removed, or the
/// phone switched between earpiece and loudspeaker.
class RouteChanged extends AudioEvent {
  const RouteChanged(this.route);

  final AudioRoute route;

  @override
  bool operator ==(Object other) => other is RouteChanged && other.route == route;

  @override
  int get hashCode => Object.hash(RouteChanged, route);
}

/// Map the native `route` string onto [AudioRoute]. Unknown / unseen values fall back to
/// [AudioRoute.unknown] rather than throwing, so a new OS route name can't crash a call.
AudioRoute parseAudioRoute(String? raw) => switch (raw) {
      'speaker' => AudioRoute.speaker,
      'receiver' => AudioRoute.receiver,
      'headphones' => AudioRoute.headphones,
      'bluetooth' => AudioRoute.bluetooth,
      'carAudio' => AudioRoute.carAudio,
      _ => AudioRoute.unknown,
    };

/// Decode one platform-channel payload into an [AudioEvent], or `null` for an unrecognized
/// shape (defensive: a future native event we don't model is dropped, not fatal). Pure, so
/// it's unit-tested directly without the platform channel.
AudioEvent? parseAudioEvent(Object? raw) {
  if (raw is! Map) return null;
  switch (raw['event']) {
    case 'interruptionBegan':
      return const InterruptionBegan();
    case 'interruptionEnded':
      return InterruptionEnded(shouldResume: raw['shouldResume'] == true);
    case 'routeChanged':
      return RouteChanged(parseAudioRoute(raw['route'] as String?));
    default:
      return null;
  }
}

/// Bridges the native audio-session monitor (see `ios/Runner/AudioSessionMonitor.swift` and
/// `android/.../AudioSessionMonitor.kt`) to a typed Dart [Stream]. The native side only starts
/// emitting once the broadcast stream is first listened to, so an unused [AudioInterruptions]
/// costs nothing.
class AudioInterruptions {
  AudioInterruptions._(this._channel);

  /// Shared instance over the agreed channel name. Tests inject their own stream into
  /// [VoiceSession] instead of touching this.
  static final AudioInterruptions instance =
      AudioInterruptions._(const EventChannel('persona_voice/audio_session'));

  final EventChannel _channel;
  Stream<AudioEvent>? _events;

  /// Hot-ish broadcast of normalized audio events; null payloads (unmodeled) are filtered out.
  Stream<AudioEvent> get events => _events ??= _channel
      .receiveBroadcastStream()
      .map(parseAudioEvent)
      .where((e) => e != null)
      .cast<AudioEvent>();
}
