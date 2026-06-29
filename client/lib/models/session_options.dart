import 'dart:convert';

/// CEFR proficiency level for language-learner personas (maps to the server's `cefr`).
enum CefrLevel { a1, a2, b1, b2, c1, c2 }

extension CefrLevelX on CefrLevel {
  /// The wire value the server expects (`A1`…`C2`).
  String get wire => name.toUpperCase();

  /// Short label for the picker (same as the wire value).
  String get label => wire;
}

/// Parse a CEFR wire value (`A1`…`C2`, case-insensitive); null when unknown/absent.
CefrLevel? cefrFromWire(String? value) {
  if (value == null) return null;
  final s = value.trim().toUpperCase();
  for (final level in CefrLevel.values) {
    if (level.wire == s) return level;
  }
  return null;
}

/// Conversational demeanor overlaid on the persona (maps to the server's `demeanor`).
enum Demeanor { kind, natural, rude }

extension DemeanorX on Demeanor {
  /// The wire value the server expects (`kind` | `natural` | `rude`).
  String get wire => name;

  /// Human label for the picker.
  String get label => switch (this) {
        Demeanor.kind => 'Kind',
        Demeanor.natural => 'Natural',
        Demeanor.rude => 'Blunt',
      };
}

/// Parse a demeanor wire value (case-insensitive); null when unknown/absent.
Demeanor? demeanorFromWire(String? value) {
  if (value == null) return null;
  final s = value.trim().toLowerCase();
  for (final d in Demeanor.values) {
    if (d.wire == s) return d;
  }
  return null;
}

/// Per-conversation overrides layered on top of a persona — chosen before a call and
/// (optionally) changed mid-call. Every field is nullable; null = the persona's own default,
/// so an all-null [SessionOptions] changes nothing. Mirrors the server's `SessionOptions`
/// (`src/personavoice/models.py`); the same three keys ride the `/token` body and the mid-call
/// data message.
class SessionOptions {
  const SessionOptions({this.voice, this.cefr, this.demeanor});

  /// A voice-library id (preset | clone | finetuned); null keeps the persona's voice.
  final String? voice;
  final CefrLevel? cefr;
  final Demeanor? demeanor;

  bool get isEmpty => voice == null && cefr == null && demeanor == null;
  bool get isNotEmpty => !isEmpty;

  /// Number of fields actually overridden (for a compact "2 customized" hint).
  int get count =>
      (voice != null ? 1 : 0) + (cefr != null ? 1 : 0) + (demeanor != null ? 1 : 0);

  /// Copy with selected fields changed. Pass a `clear*` flag to force a field back to null
  /// (a plain null argument means "leave unchanged", so clearing needs an explicit flag).
  SessionOptions copyWith({
    String? voice,
    CefrLevel? cefr,
    Demeanor? demeanor,
    bool clearVoice = false,
    bool clearCefr = false,
    bool clearDemeanor = false,
  }) =>
      SessionOptions(
        voice: clearVoice ? null : (voice ?? this.voice),
        cefr: clearCefr ? null : (cefr ?? this.cefr),
        demeanor: clearDemeanor ? null : (demeanor ?? this.demeanor),
      );

  /// The set fields as the server's wire keys — for the `/token` body and the data message.
  /// Empty when nothing is overridden.
  Map<String, String> toWireMap() => {
        if (voice != null && voice!.isNotEmpty) 'voice': voice!,
        if (cefr != null) 'cefr': cefr!.wire,
        if (demeanor != null) 'demeanor': demeanor!.wire,
      };

  Map<String, dynamic> toJson() => toWireMap();

  factory SessionOptions.fromJson(Map<String, dynamic> json) => SessionOptions(
        voice: (json['voice'] as String?)?.trim().isNotEmpty == true
            ? (json['voice'] as String).trim()
            : null,
        cefr: cefrFromWire(json['cefr'] as String?),
        demeanor: demeanorFromWire(json['demeanor'] as String?),
      );

  /// JSON-encode for storage in [AppPreferences].
  String encode() => jsonEncode(toJson());

  /// Decode a stored options string; tolerant of null/garbage (returns empty options).
  static SessionOptions decode(String? stored) {
    if (stored == null || stored.trim().isEmpty) return const SessionOptions();
    try {
      final obj = jsonDecode(stored);
      return obj is Map<String, dynamic>
          ? SessionOptions.fromJson(obj)
          : const SessionOptions();
    } catch (_) {
      return const SessionOptions();
    }
  }

  @override
  bool operator ==(Object other) =>
      other is SessionOptions &&
      other.voice == voice &&
      other.cefr == cefr &&
      other.demeanor == demeanor;

  @override
  int get hashCode => Object.hash(voice, cefr, demeanor);
}
