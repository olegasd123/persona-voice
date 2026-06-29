/// One selectable voice from the server catalog (`GET /voices`). Mirrors the server's
/// `VoiceOption` (`src/personavoice/voice/registry.py`).
class VoiceOption {
  const VoiceOption({
    required this.id,
    required this.name,
    required this.kind,
    this.emotion,
    this.available = true,
    this.reason,
    this.removable = false,
  });

  final String id;
  final String name;

  /// "preset" | "clone" | "finetuned".
  final String kind;
  final String? emotion;

  /// Whether this voice is speakable on the *active* TTS backend (clones/fine-tunes need a
  /// cloning backend). When false, [reason] explains why so the UI can grey it out.
  final bool available;
  final String? reason;

  /// Whether the user may delete this voice. Only user-enrolled clones are removable; presets,
  /// fine-tunes and the bundled "inbox" seed clones are not. The UI hides the delete control
  /// when false (and the server rejects the delete anyway).
  final bool removable;

  bool get isClone => kind == 'clone';

  /// A built-in preset voice (vs. a clone / fine-tune). Presets carry a human descriptor in
  /// [name] ("warm, soft, feminine"); clones/fine-tunes carry only their library name.
  bool get isPreset => kind == 'preset';

  /// Title-cased kind for section headers ("Clone", "Preset", "Finetuned").
  String get kindLabel =>
      kind.isEmpty ? 'Voice' : '${kind[0].toUpperCase()}${kind.substring(1)}';

  factory VoiceOption.fromJson(Map<String, dynamic> json) {
    final id = json['id'] as String? ?? '';
    final kind = json['kind'] as String? ?? 'preset';
    String? clean(String? s) => (s != null && s.trim().isNotEmpty) ? s.trim() : null;
    return VoiceOption(
      id: id,
      name: clean(json['name'] as String?) ?? id,
      kind: kind,
      emotion: clean(json['emotion'] as String?),
      available: json['available'] as bool? ?? true,
      reason: clean(json['reason'] as String?),
      // Older servers don't send `removable`; fall back to "any clone is removable".
      removable: json['removable'] as bool? ?? (kind == 'clone'),
    );
  }
}

/// The `GET /voices` response: the catalog plus the active-backend capability flags.
class VoiceCatalog {
  const VoiceCatalog({
    required this.voices,
    this.backend = '',
    this.tts = '',
    this.supportsCloning = false,
  });

  final List<VoiceOption> voices;
  final String backend;
  final String tts;

  /// Whether the active TTS backend can speak clones/fine-tunes (and accept enrollment).
  final bool supportsCloning;

  /// Voices the user can actually select right now.
  List<VoiceOption> get available => voices.where((v) => v.available).toList();

  /// Voices listed but not speakable on this backend (shown greyed, with a reason).
  List<VoiceOption> get unavailable => voices.where((v) => !v.available).toList();

  VoiceOption? byId(String? id) {
    if (id == null || id.isEmpty) return null;
    for (final v in voices) {
      if (v.id == id) return v;
    }
    return null;
  }

  factory VoiceCatalog.fromJson(Map<String, dynamic> json) => VoiceCatalog(
        voices: (json['voices'] as List<dynamic>? ?? const [])
            .map((e) => VoiceOption.fromJson(e as Map<String, dynamic>))
            .toList(),
        backend: json['backend'] as String? ?? '',
        tts: json['tts'] as String? ?? '',
        supportsCloning: json['supports_cloning'] as bool? ?? false,
      );

  static const empty = VoiceCatalog(voices: []);
}
