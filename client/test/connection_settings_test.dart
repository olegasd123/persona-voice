import 'package:flutter_test/flutter_test.dart';
import 'package:personavoice_client/models/connection_settings.dart';
import 'package:shared_preferences/shared_preferences.dart';

void main() {
  group('normalizeUserId', () {
    test('passes a clean id through unchanged (case preserved)', () {
      expect(ConnectionSettings.normalizeUserId('oleg'), 'oleg');
      expect(ConnectionSettings.normalizeUserId('Oleg'), 'Oleg');
      expect(ConnectionSettings.normalizeUserId('work.2'), 'work.2');
      expect(ConnectionSettings.normalizeUserId('a@b.com'), 'a@b.com');
    });

    test('trims, maps illegal chars to "-", and collapses repeats', () {
      expect(ConnectionSettings.normalizeUserId('  Oleg Smith  '), 'Oleg-Smith');
      expect(ConnectionSettings.normalizeUserId('language practice'), 'language-practice');
      expect(ConnectionSettings.normalizeUserId('a / b'), 'a-b');
    });

    test('drops leading non-alphanumerics so the first char is valid', () {
      expect(ConnectionSettings.normalizeUserId('  @work'), 'work');
      expect(ConnectionSettings.normalizeUserId('---x'), 'x');
    });

    test('returns empty when nothing usable remains', () {
      expect(ConnectionSettings.normalizeUserId(''), '');
      expect(ConnectionSettings.normalizeUserId('   '), '');
      expect(ConnectionSettings.normalizeUserId('@@@'), '');
    });

    test('caps the length at 128 chars', () {
      expect(ConnectionSettings.normalizeUserId('a' * 200).length, 128);
    });
  });

  group('effectiveUser', () {
    test('normalizes the account id', () {
      expect(ConnectionSettings(userId: 'Oleg Smith').effectiveUser, 'Oleg-Smith');
    });

    test('falls back to the default bucket when unset', () {
      expect(ConnectionSettings().effectiveUser, ConnectionSettings.defaultUser);
      expect(ConnectionSettings(userId: '  ').effectiveUser, ConnectionSettings.defaultUser);
    });
  });

  group('load', () {
    test('migrates the legacy identity field into both split fields', () async {
      SharedPreferences.setMockInitialValues({'connection.identity': 'oleg'});
      final s = await ConnectionSettings.load();
      expect(s.displayName, 'oleg');
      expect(s.userId, 'oleg');
    });

    test('prefers the new split fields over the legacy one when present', () async {
      SharedPreferences.setMockInitialValues({
        'connection.identity': 'old',
        'connection.displayName': 'Oleg',
        'connection.userId': 'personal',
      });
      final s = await ConnectionSettings.load();
      expect(s.displayName, 'Oleg');
      expect(s.userId, 'personal');
    });
  });
}
