import 'package:flutter/foundation.dart';

import 'token_client.dart';

/// A snapshot of the wait, for the queue page to render.
@immutable
class QueueStatus {
  const QueueStatus({this.attempts = 0, this.waited = Duration.zero, this.retryAfter = 10});

  /// How many times we've re-asked the server for a slot.
  final int attempts;

  /// Total time spent waiting so far (sum of the retry intervals elapsed).
  final Duration waited;

  /// The interval (seconds) before the next attempt — the server's latest `Retry-After`.
  final int retryAfter;
}

/// Drives the lightweight "wait for a free slot" loop behind the queue page.
///
/// The server has no queue: it just returns `503` (busy) or mints a token. So this polls
/// [request] on the server's suggested interval, retrying while the response is busy (`503`)
/// and resolving as soon as one succeeds. A non-busy failure (auth, server down) is surfaced so
/// the page can show it rather than spin forever; [cancel] stops the loop (the user left).
///
/// `sleep` is injected so tests can run the loop instantly; production uses a real delay.
class QueueController {
  QueueController({
    required Future<JoinGrant> Function() request,
    int initialRetryAfter = 10,
    Future<void> Function(Duration) sleep = _realSleep,
  })  : _request = request,
        _sleep = sleep,
        _retryAfter = initialRetryAfter < 1 ? 1 : initialRetryAfter;

  final Future<JoinGrant> Function() _request;
  final Future<void> Function(Duration) _sleep;

  int _retryAfter;
  int _attempts = 0;
  Duration _waited = Duration.zero;
  bool _cancelled = false;

  final ValueNotifier<QueueStatus> status = ValueNotifier(const QueueStatus());

  /// Poll until a slot frees (returns the grant), the caller [cancel]s (returns null), or a
  /// non-busy error occurs (rethrown). Waits *before* the first retry — the caller only enters the
  /// queue after already getting one busy response, so an immediate re-ask would just bounce.
  Future<JoinGrant?> run() async {
    _emit();
    while (!_cancelled) {
      await _sleep(Duration(seconds: _retryAfter));
      if (_cancelled) return null;
      _attempts++;
      _waited += Duration(seconds: _retryAfter);
      _emit();
      try {
        return await _request();
      } on TokenClientException catch (e) {
        if (!e.isBusy) rethrow; // a real error — let the page show it
        if (e.retryAfter != null && e.retryAfter! >= 1) _retryAfter = e.retryAfter!;
        _emit();
      }
    }
    return null;
  }

  /// Stop waiting (the user backed out). A `run()` in flight resolves to null at the next tick.
  void cancel() => _cancelled = true;

  void dispose() {
    _cancelled = true;
    status.dispose();
  }

  void _emit() => status.value = QueueStatus(
        attempts: _attempts,
        waited: _waited,
        retryAfter: _retryAfter,
      );

  static Future<void> _realSleep(Duration d) => Future<void>.delayed(d);
}
