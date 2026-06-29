import 'package:flutter_test/flutter_test.dart';
import 'package:personavoice_client/services/queue_controller.dart';
import 'package:personavoice_client/services/token_client.dart';

const _grant = JoinGrant(
  url: 'wss://lk',
  token: 't',
  room: 'r',
  identity: 'me',
  persona: 'companion',
);

TokenClientException _busy({int? retryAfter}) =>
    TokenClientException('busy', statusCode: 503, retryAfter: retryAfter);

void main() {
  // Instant sleep so the loop runs without real delays.
  Future<void> noSleep(Duration _) async {}

  test('retries while busy, then resolves with the grant once a slot frees', () async {
    var calls = 0;
    final controller = QueueController(
      request: () async {
        calls++;
        if (calls < 3) throw _busy();
        return _grant;
      },
      sleep: noSleep,
    );

    final grant = await controller.run();

    expect(grant, same(_grant));
    expect(calls, 3); // two busy responses, then success
    expect(controller.status.value.attempts, 3);
  });

  test('adopts the server\'s updated Retry-After between attempts', () async {
    var calls = 0;
    final controller = QueueController(
      request: () async {
        calls++;
        if (calls == 1) throw _busy(retryAfter: 25);
        return _grant;
      },
      initialRetryAfter: 5,
      sleep: noSleep,
    );

    await controller.run();
    // The 503 carried retry_after=25, so the status reflects the server's hint, not the initial 5.
    expect(controller.status.value.retryAfter, 25);
  });

  test('returns null when cancelled (no token), without calling request again', () async {
    var calls = 0;
    late QueueController controller;
    controller = QueueController(
      request: () async {
        calls++;
        controller.cancel(); // cancel mid-flight, on the first busy attempt
        throw _busy();
      },
      sleep: noSleep,
    );

    final grant = await controller.run();

    expect(grant, isNull);
    expect(calls, 1); // stopped after the cancel; didn't loop again
  });

  test('rethrows a non-busy error so the page can show it', () async {
    final controller = QueueController(
      request: () async => throw TokenClientException('nope', statusCode: 401),
      sleep: noSleep,
    );

    await expectLater(controller.run(), throwsA(isA<TokenClientException>()));
  });
}
