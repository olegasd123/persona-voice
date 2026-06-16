import 'package:flutter_test/flutter_test.dart';
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
