import 'package:flutter_test/flutter_test.dart';
import 'package:personavoice_client/models/persona.dart';

void main() {
  test('parses all fields', () {
    final p = Persona.fromJson({
      'id': 'companion',
      'name': 'Companion',
      'description': 'A warm friend.',
      'voice': 'warm, soft, feminine',
    });
    expect(p.id, 'companion');
    expect(p.name, 'Companion');
    expect(p.description, 'A warm friend.');
    expect(p.voice, 'warm, soft, feminine');
    expect(p.initial, 'C');
  });

  test('description and voice default to empty (older server)', () {
    final p = Persona.fromJson({'id': 'x', 'name': 'X'});
    expect(p.description, '');
    expect(p.voice, '');
  });

  test('falls back to the id when name is blank', () {
    final p = Persona.fromJson({'id': 'companion', 'name': '   '});
    expect(p.name, 'companion');
    expect(p.initial, 'C');
  });

  test('initial is "?" when name is empty', () {
    expect(Persona.fromJson({'id': '', 'name': ''}).initial, '?');
  });
}
