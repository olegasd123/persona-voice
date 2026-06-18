package com.personavoice.personavoice_client

import android.content.Context
import android.media.AudioAttributes
import android.media.AudioDeviceCallback
import android.media.AudioDeviceInfo
import android.media.AudioFocusRequest
import android.media.AudioManager
import android.os.Build
import android.os.Handler
import android.os.Looper
import io.flutter.plugin.common.EventChannel

/**
 * Bridges Android audio-focus changes + output-device changes to the Flutter
 * `persona_voice/audio_session` EventChannel — the Android half of the deep audio-session
 * plumbing. An incoming call (or any app grabbing audio focus) surfaces as
 * `interruptionBegan/Ended`; a headset/Bluetooth device connecting or dropping surfaces as
 * `routeChanged`, so the thin client can mute/resume the mic and show the active route.
 *
 * Note: LiveKit/WebRTC also manages audio focus during a call, so the focus-based interruption
 * path is the part most in need of on-device tuning (it may need to ride LiveKit's own focus
 * handling instead of a competing request). Route observation is independent and safe.
 */
class AudioSessionMonitor(context: Context) : EventChannel.StreamHandler {
  private val audioManager =
      context.applicationContext.getSystemService(Context.AUDIO_SERVICE) as AudioManager
  private val mainHandler = Handler(Looper.getMainLooper())

  private var sink: EventChannel.EventSink? = null
  private var focusRequest: AudioFocusRequest? = null
  private var deviceCallback: AudioDeviceCallback? = null
  private var interrupted = false

  private val focusListener = AudioManager.OnAudioFocusChangeListener { change ->
    when (change) {
      AudioManager.AUDIOFOCUS_LOSS,
      AudioManager.AUDIOFOCUS_LOSS_TRANSIENT,
      AudioManager.AUDIOFOCUS_LOSS_TRANSIENT_CAN_DUCK -> {
        if (!interrupted) {
          interrupted = true
          send(mapOf("event" to "interruptionBegan"))
        }
      }
      AudioManager.AUDIOFOCUS_GAIN -> {
        if (interrupted) {
          interrupted = false
          send(mapOf("event" to "interruptionEnded", "shouldResume" to true))
        }
      }
    }
  }

  override fun onListen(arguments: Any?, events: EventChannel.EventSink?) {
    sink = events
    registerFocusListener()
    registerDeviceCallback()
    emitCurrentRoute()
  }

  override fun onCancel(arguments: Any?) {
    abandonFocus()
    deviceCallback?.let { audioManager.unregisterAudioDeviceCallback(it) }
    deviceCallback = null
    sink = null
    interrupted = false
  }

  private fun registerFocusListener() {
    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
      val request = AudioFocusRequest.Builder(AudioManager.AUDIOFOCUS_GAIN)
          .setAudioAttributes(
              AudioAttributes.Builder()
                  .setUsage(AudioAttributes.USAGE_VOICE_COMMUNICATION)
                  .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
                  .build())
          .setOnAudioFocusChangeListener(focusListener, mainHandler)
          .build()
      focusRequest = request
      audioManager.requestAudioFocus(request)
    } else {
      @Suppress("DEPRECATION")
      audioManager.requestAudioFocus(
          focusListener, AudioManager.STREAM_VOICE_CALL, AudioManager.AUDIOFOCUS_GAIN)
    }
  }

  private fun abandonFocus() {
    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
      focusRequest?.let { audioManager.abandonAudioFocusRequest(it) }
      focusRequest = null
    } else {
      @Suppress("DEPRECATION")
      audioManager.abandonAudioFocus(focusListener)
    }
  }

  private fun registerDeviceCallback() {
    val callback = object : AudioDeviceCallback() {
      override fun onAudioDevicesAdded(addedDevices: Array<out AudioDeviceInfo>?) = emitCurrentRoute()
      override fun onAudioDevicesRemoved(removedDevices: Array<out AudioDeviceInfo>?) = emitCurrentRoute()
    }
    deviceCallback = callback
    audioManager.registerAudioDeviceCallback(callback, mainHandler)
  }

  private fun emitCurrentRoute() {
    send(mapOf("event" to "routeChanged", "route" to currentRouteName()))
  }

  /**
   * Best-effort active output: there's no portable "active output device" before API 31, so we
   * pick the most external connected output (Bluetooth > wired > car > speaker), matching how
   * Android would actually route during a voice call.
   */
  private fun currentRouteName(): String {
    val outputs = audioManager.getDevices(AudioManager.GET_DEVICES_OUTPUTS)
    var route = "speaker"
    for (device in outputs) {
      when (device.type) {
        AudioDeviceInfo.TYPE_BLUETOOTH_A2DP,
        AudioDeviceInfo.TYPE_BLUETOOTH_SCO -> return "bluetooth"
        AudioDeviceInfo.TYPE_WIRED_HEADPHONES,
        AudioDeviceInfo.TYPE_WIRED_HEADSET,
        AudioDeviceInfo.TYPE_USB_HEADSET -> route = "headphones"
        AudioDeviceInfo.TYPE_BUS -> if (route == "speaker") route = "carAudio"
      }
    }
    return route
  }

  private fun send(payload: Map<String, Any>) {
    mainHandler.post { sink?.success(payload) }
  }
}
