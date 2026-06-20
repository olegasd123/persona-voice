class Persona {
  const Persona({
    required this.id,
    required this.name,
  });

  final String id;
  final String name;

  factory Persona.fromJson(Map<String, dynamic> json) {
    final id = json['id'] as String? ?? '';
    return Persona(
      id: id,
      name: (json['name'] as String?)?.trim().isNotEmpty == true
          ? json['name'] as String
          : id,
    );
  }
}
