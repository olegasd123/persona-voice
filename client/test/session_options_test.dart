import 'package:flutter_test/flutter_test.dart';
import 'package:personavoice_client/models/session_options.dart';

void main() {
  test('empty options serialize to nothing and report empty', () {
    const o = SessionOptions();
    expect(o.isEmpty, isTrue);
    expect(o.count, 0);
    expect(o.toWireMap(), isEmpty);
  });

  test('only set fields appear in the wire map', () {
    const o = SessionOptions(voice: 'my_voice', cefr: CefrLevel.b1);
    expect(o.isNotEmpty, isTrue);
    expect(o.count, 2);
    expect(o.toWireMap(), {'voice': 'my_voice', 'cefr': 'B1'});
    // demeanor was never set, so it's absent (not null-valued)
    expect(o.toWireMap().containsKey('demeanor'), isFalse);
  });

  test('demeanor uses lowercase wire values', () {
    const o = SessionOptions(demeanor: Demeanor.rude);
    expect(o.toWireMap(), {'demeanor': 'rude'});
  });

  test('copyWith sets and clears fields independently', () {
    const base = SessionOptions(voice: 'v', cefr: CefrLevel.a2, demeanor: Demeanor.kind);
    final clearedVoice = base.copyWith(clearVoice: true);
    expect(clearedVoice.voice, isNull);
    expect(clearedVoice.cefr, CefrLevel.a2); // untouched
    final changed = base.copyWith(cefr: CefrLevel.c1);
    expect(changed.cefr, CefrLevel.c1);
    expect(changed.voice, 'v');
  });

  test('fromJson parses wire values and ignores blanks', () {
    final o = SessionOptions.fromJson({'voice': '  ', 'cefr': 'b2', 'demeanor': 'NATURAL'});
    expect(o.voice, isNull); // blank dropped
    expect(o.cefr, CefrLevel.b2); // case-insensitive
    expect(o.demeanor, Demeanor.natural);
  });

  test('encode/decode round-trips and tolerates garbage', () {
    const o = SessionOptions(voice: 'my_voice', demeanor: Demeanor.kind);
    expect(SessionOptions.decode(o.encode()), o);
    expect(SessionOptions.decode(null), const SessionOptions());
    expect(SessionOptions.decode('not json'), const SessionOptions());
    expect(SessionOptions.decode('[1,2,3]'), const SessionOptions());
  });

  test('wire parsers reject unknown values', () {
    expect(cefrFromWire('z9'), isNull);
    expect(cefrFromWire(null), isNull);
    expect(demeanorFromWire('grumpy'), isNull);
  });

  test('equality is by value', () {
    expect(const SessionOptions(voice: 'v'), const SessionOptions(voice: 'v'));
    expect(const SessionOptions(voice: 'v'), isNot(const SessionOptions(voice: 'w')));
  });
}
