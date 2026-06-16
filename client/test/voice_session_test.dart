import 'package:flutter_test/flutter_test.dart';
import 'package:personavoice_client/services/audio_session.dart';
import 'package:personavoice_client/services/voice_session.dart';

void main() {
  group('sessionStatusLabel', () {
    test('maps each status to its label', () {
      String label(SessionStatus s, {bool speaking = false}) =>
          sessionStatusLabel(s, agentSpeaking: speaking);

      expect(label(SessionStatus.idle), 'Idle');
      expect(label(SessionStatus.connecting), 'Connecting…');
      expect(label(SessionStatus.reconnecting), 'Reconnecting…');
      expect(label(SessionStatus.disconnected), 'Disconnected');
      expect(label(SessionStatus.error), 'Error');
    });

    test('connected reflects who is speaking', () {
      expect(sessionStatusLabel(SessionStatus.connected, agentSpeaking: false), 'Listening');
      expect(sessionStatusLabel(SessionStatus.connected, agentSpeaking: true), 'Speaking…');
    });

    test('interruption overrides the connected label', () {
      expect(
        sessionStatusLabel(SessionStatus.connected, agentSpeaking: true, interrupted: true),
        'Paused (interrupted)',
      );
      // Only while connected — a non-connected status keeps its own label.
      expect(
        sessionStatusLabel(SessionStatus.connecting, agentSpeaking: false, interrupted: true),
        'Connecting…',
      );
    });
  });

  group('VoiceSession audio interruptions (no room)', () {
    test('open-mic: interruption mutes, then auto-resumes on a resumable end', () async {
      final s = VoiceSession();
      await s.toggleMic(); // mic live in open-mic
      expect(s.micEnabled, true);

      await s.onAudioEvent(const InterruptionBegan());
      expect(s.interrupted, true);
      expect(s.micEnabled, false);

      await s.onAudioEvent(const InterruptionEnded(shouldResume: true));
      expect(s.interrupted, false);
      expect(s.micEnabled, true);
    });

    test('does not resume when the OS says not to', () async {
      final s = VoiceSession();
      await s.toggleMic();

      await s.onAudioEvent(const InterruptionBegan());
      await s.onAudioEvent(const InterruptionEnded(shouldResume: false));
      expect(s.interrupted, false);
      expect(s.micEnabled, false);
    });

    test('does not resume a mic the user had already muted', () async {
      final s = VoiceSession();
      // mic starts disabled in a room-less session; leave it muted.
      expect(s.micEnabled, false);

      await s.onAudioEvent(const InterruptionBegan());
      await s.onAudioEvent(const InterruptionEnded(shouldResume: true));
      expect(s.micEnabled, false);
    });

    test('push-to-talk stays muted after an interruption ends', () async {
      final s = VoiceSession();
      await s.setMicMode(MicMode.pushToTalk);
      await s.setTalking(true);
      expect(s.micEnabled, true);

      await s.onAudioEvent(const InterruptionBegan());
      expect(s.talking, false);
      expect(s.micEnabled, false);

      await s.onAudioEvent(const InterruptionEnded(shouldResume: true));
      expect(s.micEnabled, false); // user must hold again
    });

    test('a duplicate interruptionBegan is ignored', () async {
      final s = VoiceSession();
      await s.toggleMic();
      await s.onAudioEvent(const InterruptionBegan());
      var notifications = 0;
      s.addListener(() => notifications++);
      await s.onAudioEvent(const InterruptionBegan());
      expect(notifications, 0);
    });

    test('route changes are recorded and de-duplicated', () async {
      final s = VoiceSession();
      expect(s.route, AudioRoute.speaker);

      var notifications = 0;
      s.addListener(() => notifications++);
      await s.onAudioEvent(const RouteChanged(AudioRoute.bluetooth));
      expect(s.route, AudioRoute.bluetooth);
      expect(notifications, 1);

      await s.onAudioEvent(const RouteChanged(AudioRoute.bluetooth));
      expect(notifications, 1); // same route → no extra notify
    });
  });

  group('VoiceSession mic modes (no room)', () {
    test('defaults to open-mic', () {
      expect(VoiceSession().micMode, MicMode.openMic);
    });

    test('switching to push-to-talk mutes and clears talking', () async {
      final s = VoiceSession();
      await s.setTalking(true); // ignored in open-mic
      expect(s.talking, false);

      await s.setMicMode(MicMode.pushToTalk);
      expect(s.micMode, MicMode.pushToTalk);
      expect(s.micEnabled, false);
      expect(s.talking, false);
    });

    test('push-to-talk enables the mic only while held', () async {
      final s = VoiceSession();
      await s.setMicMode(MicMode.pushToTalk);

      await s.setTalking(true);
      expect(s.talking, true);
      expect(s.micEnabled, true);

      await s.setTalking(false);
      expect(s.talking, false);
      expect(s.micEnabled, false);
    });

    test('switching back to open-mic re-enables the mic', () async {
      final s = VoiceSession();
      await s.setMicMode(MicMode.pushToTalk);
      await s.setMicMode(MicMode.openMic);
      expect(s.micMode, MicMode.openMic);
      expect(s.micEnabled, true);
    });

    test('setMicMode to the same mode is a no-op', () async {
      final s = VoiceSession();
      var notifications = 0;
      s.addListener(() => notifications++);
      await s.setMicMode(MicMode.openMic);
      expect(notifications, 0);
    });

    test('toggleMic flips the enabled state', () async {
      final s = VoiceSession();
      expect(await s.toggleMic(), true);
      expect(s.micEnabled, true);
      expect(await s.toggleMic(), false);
      expect(s.micEnabled, false);
    });
  });
}
