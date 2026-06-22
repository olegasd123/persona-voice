class Persona {
  const Persona({
    required this.id,
    required this.name,
    this.description = '',
    this.voice = '',
  });

  final String id;
  final String name;

  /// One-line blurb from the server (`config/personas/*.yaml`); may be empty.
  final String description;

  /// Human description of the persona's voice (e.g. "warm, soft, feminine"); may be empty.
  final String voice;

  factory Persona.fromJson(Map<String, dynamic> json) {
    final id = json['id'] as String? ?? '';
    return Persona(
      id: id,
      name: (json['name'] as String?)?.trim().isNotEmpty == true
          ? json['name'] as String
          : id,
      description: (json['description'] as String?)?.trim() ?? '',
      voice: (json['voice'] as String?)?.trim() ?? '',
    );
  }

  /// First letter for the avatar circle; falls back to '?' for an empty name.
  String get initial =>
      name.trim().isNotEmpty ? name.trim()[0].toUpperCase() : '?';
}
