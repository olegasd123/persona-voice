package com.personavoice.personavoice_client

import android.os.Build
import android.telecom.Connection
import android.telecom.ConnectionRequest
import android.telecom.ConnectionService
import android.telecom.PhoneAccountHandle
import android.telecom.TelecomManager
import androidx.annotation.RequiresApi

/**
 * Self-managed [ConnectionService] (API 26+) that the Telecom framework instantiates when we
 * place a call via [TelecomManager.placeCall] in [TelephonyController]. It builds a
 * [PersonaConnection], marks it self-managed + VoIP, and hands it back so the conversation shows
 * up in the system call UI. Declared in the manifest with `BIND_TELECOM_CONNECTION_SERVICE`.
 */
@RequiresApi(Build.VERSION_CODES.O)
class PersonaConnectionService : ConnectionService() {

  override fun onCreateOutgoingConnection(
      connectionManagerPhoneAccount: PhoneAccountHandle?,
      request: ConnectionRequest?
  ): Connection {
    val connection = PersonaConnection()
    connection.connectionProperties = Connection.PROPERTY_SELF_MANAGED
    connection.connectionCapabilities = Connection.CAPABILITY_MUTE
    connection.audioModeIsVoip = true
    request?.address?.let { connection.setAddress(it, TelecomManager.PRESENTATION_ALLOWED) }
    Telephony.pendingDisplayName?.let {
      connection.setCallerDisplayName(it, TelecomManager.PRESENTATION_ALLOWED)
    }
    // The session is already establishing over LiveKit; reportConnected() flips it to active.
    connection.setDialing()
    Telephony.activeConnection = connection
    return connection
  }

  override fun onCreateOutgoingConnectionFailed(
      connectionManagerPhoneAccount: PhoneAccountHandle?,
      request: ConnectionRequest?
  ) {
    Telephony.activeConnection = null
    Telephony.emit(mapOf("event" to "endCall"))
  }
}
