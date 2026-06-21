package com.personavoice.personavoice_client

import android.os.Build
import android.telecom.CallAudioState
import android.telecom.Connection
import android.telecom.DisconnectCause
import androidx.annotation.RequiresApi

/**
 * One self-managed Telecom [Connection] representing a Persona-Voice conversation. The system
 * in-call UI (lock screen, call list) drives it; this surfaces the user's end + mute actions
 * back to Dart via [Telephony.emit], mirroring the iOS `CXProviderDelegate`.
 *
 * Self-managed connections are API 26+; below that the telephony layer is a no-op (see
 * [TelephonyController]).
 */
@RequiresApi(Build.VERSION_CODES.O)
class PersonaConnection : Connection() {

  override fun onAnswer() {
    setActive()
  }

  override fun onDisconnect() {
    // User tapped "end" on the system call UI.
    finish(DisconnectCause.LOCAL)
    Telephony.emit(mapOf("event" to "endCall"))
  }

  override fun onAbort() {
    finish(DisconnectCause.CANCELED)
    Telephony.emit(mapOf("event" to "endCall"))
  }

  /** The system in-call UI owns mute for self-managed calls; relay its state to the app. */
  override fun onCallAudioStateChanged(state: CallAudioState) {
    Telephony.emit(mapOf("event" to "setMuted", "muted" to state.isMuted))
  }

  private fun finish(cause: Int) {
    setDisconnected(DisconnectCause(cause))
    destroy()
    if (Telephony.activeConnection === this) Telephony.activeConnection = null
  }
}
