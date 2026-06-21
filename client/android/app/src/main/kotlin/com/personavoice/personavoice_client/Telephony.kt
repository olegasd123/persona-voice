package com.personavoice.personavoice_client

import android.os.Handler
import android.os.Looper
import io.flutter.plugin.common.EventChannel

/**
 * Process-wide bridge between the OS-instantiated [PersonaConnectionService] / [PersonaConnection]
 * and the Flutter telephony EventChannel. The Telecom framework creates the ConnectionService
 * itself (we never get to hand it the Flutter messenger), so it reaches Dart through this
 * singleton's [eventSink], which [TelephonyController] owns.
 */
object Telephony {
  private val mainHandler = Handler(Looper.getMainLooper())

  /** Owned by [TelephonyController.onListen] / [TelephonyController.onCancel]. */
  @Volatile var eventSink: EventChannel.EventSink? = null

  /** The live self-managed connection, if any (set by the service, cleared on disconnect). */
  @Volatile var activeConnection: PersonaConnection? = null

  /** Caller display name stashed by [TelephonyController.startCall] for the next connection. */
  @Volatile var pendingDisplayName: String? = null

  /** Post a normalized event to Dart on the main thread (sinks are not thread-safe). */
  fun emit(event: Map<String, Any>) {
    mainHandler.post { eventSink?.success(event) }
  }
}
