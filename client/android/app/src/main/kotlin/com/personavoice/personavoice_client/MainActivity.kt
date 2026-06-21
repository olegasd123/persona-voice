package com.personavoice.personavoice_client

import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.plugin.common.EventChannel
import io.flutter.plugin.common.MethodChannel

class MainActivity : FlutterActivity() {
  private var audioMonitor: AudioSessionMonitor? = null
  private var telephony: TelephonyController? = null

  override fun configureFlutterEngine(flutterEngine: FlutterEngine) {
    super.configureFlutterEngine(flutterEngine)
    val messenger = flutterEngine.dartExecutor.binaryMessenger

    val monitor = AudioSessionMonitor(applicationContext)
    audioMonitor = monitor
    EventChannel(messenger, "persona_voice/audio_session").setStreamHandler(monitor)

    // Telephony: present the conversation as a self-managed system call (API 26+).
    val controller = TelephonyController(applicationContext)
    telephony = controller
    MethodChannel(messenger, "persona_voice/telephony").setMethodCallHandler(controller)
    EventChannel(messenger, "persona_voice/telephony_events").setStreamHandler(controller)
  }
}
