/// One selectable LoRA adapter for a custom persona's `llm.lora` (`GET /loras`). Mirrors the
/// server's `LoraOption` (`src/personavoice/persona/lora.py`).
class LoraOption {
  const LoraOption({
    required this.id,
    required this.name,
    this.available = true,
    this.reason,
  });

  final String id;
  final String name;

  /// Whether the active LLM backend can actually serve this adapter (False on a non-LoRA
  /// backend); [reason] explains a False so the picker can grey it out.
  final bool available;
  final String? reason;

  factory LoraOption.fromJson(Map<String, dynamic> json) {
    final id = json['id'] as String? ?? '';
    String? clean(String? s) => (s != null && s.trim().isNotEmpty) ? s.trim() : null;
    return LoraOption(
      id: id,
      name: clean(json['name'] as String?) ?? id,
      available: json['available'] as bool? ?? true,
      reason: clean(json['reason'] as String?),
    );
  }
}

/// The `GET /loras` response: the served adapters plus the active-backend capability flags.
/// Empty + [supportsLora] false on Mac / LM Studio, where a LoRA is merged into the base model
/// at train time rather than hot-swapped ([reason] says so).
class LoraCatalog {
  const LoraCatalog({
    this.loras = const [],
    this.llm = '',
    this.supportsLora = false,
    this.reason,
  });

  final List<LoraOption> loras;
  final String llm;
  final bool supportsLora;
  final String? reason;

  static const empty = LoraCatalog();

  factory LoraCatalog.fromJson(Map<String, dynamic> json) => LoraCatalog(
        loras: (json['loras'] as List<dynamic>? ?? const [])
            .map((e) => LoraOption.fromJson(e as Map<String, dynamic>))
            .toList(),
        llm: json['llm'] as String? ?? '',
        supportsLora: json['supports_lora'] as bool? ?? false,
        reason: (json['reason'] as String?)?.trim().isNotEmpty == true
            ? (json['reason'] as String).trim()
            : null,
      );
}
