package com.personavoice.personavoice_client

import android.content.ComponentName
import android.content.Context
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.telecom.DisconnectCause
import android.telecom.PhoneAccount
import android.telecom.PhoneAccountHandle
import android.telecom.TelecomManager
import io.flutter.plugin.common.EventChannel
import io.flutter.plugin.common.MethodCall
import io.flutter.plugin.common.MethodChannel

/**
 * Flutter-facing telephony bridge — the Android half of M6's "present the conversation as a system
 * call" layer, the analog of iOS [CallKitController]. Handles `persona_voice/telephony`
 * MethodChannel commands (app → OS) and owns the `persona_voice/telephony_events` EventChannel
 * sink (OS → app, via [Telephony]).
 *
 * Presents the call as a self-managed Telecom call on **API 26+** (the only way a VoIP app gets
 * the system call UI without being a default dialer). On API 23–25 every command is a no-op —
 * the conversation still works, it just isn't surfaced as a system call there.
 */
class TelephonyController(context: Context) :
    MethodChannel.MethodCallHandler, EventChannel.StreamHandler {

  private val appContext = context.applicationContext
  private val telecomManager =
      appContext.getSystemService(Context.TELECOM_SERVICE) as TelecomManager
  private val supported = Build.VERSION.SDK_INT >= Build.VERSION_CODES.O
  private var accountRegistered = false

  private val phoneAccountHandle: PhoneAccountHandle
    get() =
        PhoneAccountHandle(
            ComponentName(appContext, PersonaConnectionService::class.java), ACCOUNT_ID)

  override fun onMethodCall(call: MethodCall, result: MethodChannel.Result) {
    if (!supported) {
      result.success(null)
      return
    }
    try {
      when (call.method) {
        "startCall" ->
            startCall(call.argument<String>("displayName") ?: "Persona Voice",
                call.argument<String>("handle") ?: "persona")
        "reportConnected" -> Telephony.activeConnection?.setActive()
        "endCall" -> endCall()
        // Self-managed connections can't set mute programmatically — the system in-call UI owns
        // it (reported back via PersonaConnection.onCallAudioStateChanged), and the mic is
        // already muted at the WebRTC track. Accept the call and no-op.
        "setMuted" -> {}
        else -> {
          result.notImplemented()
          return
        }
      }
      result.success(null)
    } catch (e: SecurityException) {
      // Missing MANAGE_OWN_CALLS, or the OEM blocked self-managed calls.
      result.error("telephony_permission", e.message, null)
    } catch (e: IllegalArgumentException) {
      result.error("telephony_error", e.message, null)
    }
  }

  private fun ensureAccountRegistered() {
    if (accountRegistered) return
    val account =
        PhoneAccount.builder(phoneAccountHandle, ACCOUNT_LABEL)
            .setCapabilities(PhoneAccount.CAPABILITY_SELF_MANAGED)
            .addSupportedUriScheme(PhoneAccount.SCHEME_SIP)
            .build()
    telecomManager.registerPhoneAccount(account)
    accountRegistered = true
  }

  private fun startCall(displayName: String, handle: String) {
    ensureAccountRegistered()
    Telephony.pendingDisplayName = displayName
    val address = Uri.fromParts(PhoneAccount.SCHEME_SIP, handle, null)
    val extras =
        Bundle().apply {
          putParcelable(TelecomManager.EXTRA_PHONE_ACCOUNT_HANDLE, phoneAccountHandle)
        }
    telecomManager.placeCall(address, extras)
  }

  private fun endCall() {
    Telephony.activeConnection?.let {
      it.setDisconnected(DisconnectCause(DisconnectCause.LOCAL))
      it.destroy()
    }
    Telephony.activeConnection = null
  }

  override fun onListen(arguments: Any?, events: EventChannel.EventSink?) {
    Telephony.eventSink = events
  }

  override fun onCancel(arguments: Any?) {
    Telephony.eventSink = null
  }

  companion object {
    private const val ACCOUNT_ID = "persona_voice"
    private const val ACCOUNT_LABEL = "Persona Voice"
  }
}
