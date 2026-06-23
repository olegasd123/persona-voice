import 'dart:io';
import 'dart:typed_data';

import 'package:path_provider/path_provider.dart';
import 'package:record/record.dart';

/// Records a short voice sample to a WAV file and returns its bytes — the enrollment source for
/// cloning a voice from the device mic (`POST /voices/clone`). Mono PCM WAV is what the server's
/// shared audio codec decodes (`src/personavoice/audio.py`); the server resamples, so the exact
/// rate here doesn't matter. Thin wrapper so the Voice Library screen stays testable.
class VoiceRecorder {
  VoiceRecorder({AudioRecorder? recorder}) : _rec = recorder ?? AudioRecorder();

  final AudioRecorder _rec;
  String? _path;

  /// Whether the OS has granted mic access (prompts on first call).
  Future<bool> hasPermission() => _rec.hasPermission();

  Future<bool> isRecording() => _rec.isRecording();

  /// Begin recording to a temp WAV file.
  Future<void> start() async {
    final dir = await getTemporaryDirectory();
    final path = '${dir.path}/pv_clone_${DateTime.now().millisecondsSinceEpoch}.wav';
    _path = path;
    await _rec.start(
      const RecordConfig(encoder: AudioEncoder.wav, sampleRate: 24000, numChannels: 1),
      path: path,
    );
  }

  /// Stop recording and return the captured WAV bytes (null if nothing was captured). The temp
  /// file is deleted once read.
  Future<Uint8List?> stop() async {
    final stopped = await _rec.stop();
    final path = stopped ?? _path;
    _path = null;
    if (path == null) return null;
    final file = File(path);
    if (!await file.exists()) return null;
    final bytes = await file.readAsBytes();
    try {
      await file.delete();
    } catch (_) {
      // Best-effort cleanup; a leftover temp file is harmless.
    }
    return bytes;
  }

  Future<void> dispose() => _rec.dispose();
}
