import 'session_options.dart';

class Persona {
  const Persona({
    required this.id,
    required this.name,
    this.description = '',
    this.voice = '',
    this.cefr,
    this.demeanor,
    this.custom = false,
  });

  final String id;
  final String name;

  /// One-line blurb from the server (`config/personas/*.yaml`); may be empty.
  final String description;

  /// Human description of the persona's voice (e.g. "warm, soft, feminine"); may be empty.
  final String voice;

  /// The persona's baked-in session defaults (`session_defaults`), so the card can show what
  /// it speaks like before any per-call override. Null = unset (the curated personas, today).
  final CefrLevel? cefr;
  final Demeanor? demeanor;

  /// True for a user-authored persona (editable / deletable). Curated personas are false.
  final bool custom;

  factory Persona.fromJson(Map<String, dynamic> json) {
    final id = json['id'] as String? ?? '';
    return Persona(
      id: id,
      name: (json['name'] as String?)?.trim().isNotEmpty == true
          ? json['name'] as String
          : id,
      description: (json['description'] as String?)?.trim() ?? '',
      voice: (json['voice'] as String?)?.trim() ?? '',
      cefr: cefrFromWire(json['cefr'] as String?),
      demeanor: demeanorFromWire(json['demeanor'] as String?),
      custom: json['custom'] as bool? ?? false,
    );
  }

  /// First letter for the avatar circle; falls back to '?' for an empty name.
  String get initial =>
      name.trim().isNotEmpty ? name.trim()[0].toUpperCase() : '?';
}

/// Turn style baked into a persona (maps to the server's `behavior.turn_style`).
enum TurnStyle { concise, balanced, verbose }

extension TurnStyleX on TurnStyle {
  String get wire => name;
  String get label => switch (this) {
        TurnStyle.concise => 'Concise',
        TurnStyle.balanced => 'Balanced',
        TurnStyle.verbose => 'Verbose',
      };
}

TurnStyle turnStyleFromWire(String? value) {
  final s = (value ?? '').trim().toLowerCase();
  for (final t in TurnStyle.values) {
    if (t.wire == s) return t;
  }
  return TurnStyle.balanced;
}

/// An editable custom-persona draft — the fields the New/Edit form exposes, plus a snapshot of
/// the original server body so advanced fields we don't surface (temperature, memory.top_k, …)
/// round-trip on edit instead of resetting to defaults. Mirrors the server's `Persona`
/// (`src/personavoice/models.py`), which forbids unknown keys, so [toJson] sends a clean body.
class PersonaDraft {
  PersonaDraft({
    this.id,
    this.name = '',
    this.description = '',
    this.systemPrompt = '',
    this.voiceRef = '',
    this.voiceEmotion = 'neutral',
    this.lora,
    this.turnStyle = TurnStyle.balanced,
    this.memoryEnabled = false,
    this.sessionDefaults = const SessionOptions(),
    Map<String, dynamic>? original,
  }) : _original = original;

  /// Null for a new persona; set when editing one of the user's own.
  final String? id;
  String name;
  String description;
  String systemPrompt;

  /// The persona's base TTS voice (`voice.ref`) — a voice-library id from `GET /voices`.
  String voiceRef;
  String voiceEmotion;

  /// A served LoRA adapter id (`llm.lora`); null = the backend's base model. Empty on Mac.
  String? lora;
  TurnStyle turnStyle;
  bool memoryEnabled;

  /// Per-session defaults baked into the persona (CEFR / demeanor); applied unless a call
  /// overrides them. The base voice above covers voice, so [SessionOptions.voice] is unused here.
  SessionOptions sessionDefaults;

  /// The full body the server last returned, so unedited advanced fields survive a round-trip.
  final Map<String, dynamic>? _original;

  bool get isEditing => id != null;

  /// Build a draft to prefill an Edit form from a `GET /personas/{id}` (or POST/PUT) body.
  factory PersonaDraft.fromBody(Map<String, dynamic> body) {
    final voice = (body['voice'] as Map?)?.cast<String, dynamic>() ?? const {};
    final llm = (body['llm'] as Map?)?.cast<String, dynamic>() ?? const {};
    final behavior = (body['behavior'] as Map?)?.cast<String, dynamic>() ?? const {};
    final memory = (body['memory'] as Map?)?.cast<String, dynamic>() ?? const {};
    final defaults = (body['session_defaults'] as Map?)?.cast<String, dynamic>() ?? const {};
    return PersonaDraft(
      id: body['id'] as String?,
      name: (body['name'] as String?) ?? '',
      description: (body['description'] as String?) ?? '',
      systemPrompt: (body['system_prompt'] as String?) ?? '',
      voiceRef: (voice['ref'] as String?) ?? '',
      voiceEmotion: (voice['emotion'] as String?) ?? 'neutral',
      lora: llm['lora'] as String?,
      turnStyle: turnStyleFromWire(behavior['turn_style'] as String?),
      memoryEnabled: memory['enabled'] as bool? ?? false,
      sessionDefaults: SessionOptions.fromJson(defaults),
      original: body,
    );
  }

  /// The persona draft body for `POST` / `PUT`. Starts from the original (so advanced fields
  /// round-trip), overlays the edited fields, and drops server-owned keys (`id`, `user`).
  Map<String, dynamic> toJson() {
    final original = _original;
    final body =
        original != null ? Map<String, dynamic>.from(original) : <String, dynamic>{};
    body.remove('id');
    body.remove('user');

    body['name'] = name.trim();
    body['description'] = description.trim();
    body['system_prompt'] = systemPrompt.trim();

    final voice = Map<String, dynamic>.from((body['voice'] as Map?) ?? const {});
    if (voiceRef.trim().isNotEmpty) voice['ref'] = voiceRef.trim();
    voice['emotion'] = voiceEmotion;
    if (voice['ref'] != null) body['voice'] = voice;

    // base_model is server-owned: omit it on create (the server fills the curated default) and
    // let it round-trip on edit via the original body. Only `lora` is user-selectable here.
    final llm = Map<String, dynamic>.from((body['llm'] as Map?) ?? const {});
    llm['lora'] = (lora != null && lora!.trim().isNotEmpty) ? lora!.trim() : null;
    if (llm.isNotEmpty) body['llm'] = llm;

    final behavior = Map<String, dynamic>.from((body['behavior'] as Map?) ?? const {});
    behavior['turn_style'] = turnStyle.wire;
    body['behavior'] = behavior;

    final memory = Map<String, dynamic>.from((body['memory'] as Map?) ?? const {});
    memory['enabled'] = memoryEnabled;
    body['memory'] = memory;

    body['session_defaults'] = sessionDefaults.toWireMap();
    return body;
  }
}
