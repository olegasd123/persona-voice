import 'package:flutter_test/flutter_test.dart';
import 'package:personavoice_client/models/persona.dart';
import 'package:personavoice_client/models/session_options.dart';

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

  test('parses baked-in session defaults (cefr / demeanor)', () {
    final p = Persona.fromJson({
      'id': 'tutor',
      'name': 'Tutor',
      'cefr': 'b1',
      'demeanor': 'kind',
    });
    expect(p.cefr, CefrLevel.b1);
    expect(p.demeanor, Demeanor.kind);
  });

  test('description, voice and session defaults default to empty/null (older server)', () {
    final p = Persona.fromJson({'id': 'x', 'name': 'X'});
    expect(p.description, '');
    expect(p.voice, '');
    expect(p.cefr, isNull);
    expect(p.demeanor, isNull);
  });

  test('falls back to the id when name is blank', () {
    final p = Persona.fromJson({'id': 'companion', 'name': '   '});
    expect(p.name, 'companion');
    expect(p.initial, 'C');
  });

  test('initial is "?" when name is empty', () {
    expect(Persona.fromJson({'id': '', 'name': ''}).initial, '?');
  });

  test('custom flag defaults false and parses when present', () {
    expect(Persona.fromJson({'id': 'x', 'name': 'X'}).custom, isFalse);
    expect(Persona.fromJson({'id': 'x', 'name': 'X', 'custom': true}).custom, isTrue);
  });

  _draftTests();
}

void _draftTests() {
  group('PersonaDraft', () {
    final fullBody = {
      'id': 'french-tutor',
      'name': 'French Tutor',
      'description': 'Gentle French practice.',
      'system_prompt': 'Teach French.',
      'llm': {'base_model': 'qwen', 'lora': 'adapters/fr', 'temperature': 0.3},
      'voice': {'ref': 'my_voice', 'emotion': 'neutral'},
      'behavior': {'turn_style': 'concise', 'follow_up_probability': 0.9},
      'memory': {'enabled': true, 'top_k': 8},
      'session_defaults': {'cefr': 'A1', 'demeanor': 'kind'},
    };

    test('fromBody reads the edited fields', () {
      final d = PersonaDraft.fromBody(fullBody);
      expect(d.id, 'french-tutor');
      expect(d.systemPrompt, 'Teach French.');
      expect(d.voiceRef, 'my_voice');
      expect(d.lora, 'adapters/fr');
      expect(d.turnStyle, TurnStyle.concise);
      expect(d.memoryEnabled, isTrue);
      expect(d.sessionDefaults.cefr, CefrLevel.a1);
      expect(d.isEditing, isTrue);
    });

    test('toJson round-trips advanced fields it does not surface', () {
      final d = PersonaDraft.fromBody(fullBody);
      d.turnStyle = TurnStyle.balanced; // edit one field
      final out = d.toJson();
      // Server-owned id is dropped; edited field overlaid…
      expect(out.containsKey('id'), isFalse);
      expect((out['behavior'] as Map)['turn_style'], 'balanced');
      // …while advanced fields we never showed survive untouched.
      expect((out['llm'] as Map)['temperature'], 0.3);
      expect((out['behavior'] as Map)['follow_up_probability'], 0.9);
      expect((out['memory'] as Map)['top_k'], 8);
    });

    test('a fresh draft omits id and base_model so the server fills defaults', () {
      final out = PersonaDraft(name: 'New', systemPrompt: 'hi').toJson();
      expect(out.containsKey('id'), isFalse);
      expect((out['llm'] as Map?)?.containsKey('base_model') ?? false, isFalse);
      expect(out['name'], 'New');
      // No base voice chosen → no ref sent; the server defaults it from the curated persona.
      expect(out.containsKey('voice'), isFalse);
      expect(out['session_defaults'], isEmpty);
    });

    test('clearing the LoRA sends an explicit null', () {
      final d = PersonaDraft.fromBody(fullBody)..lora = null;
      expect((d.toJson()['llm'] as Map)['lora'], isNull);
    });
  });
}
