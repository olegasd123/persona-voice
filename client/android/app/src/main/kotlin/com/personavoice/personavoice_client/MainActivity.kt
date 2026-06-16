package com.personavoice.personavoice_client

import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.plugin.common.EventChannel

class MainActivity : FlutterActivity() {
  private var audioMonitor: AudioSessionMonitor? = null

  override fun configureFlutterEngine(flutterEngine: FlutterEngine) {
    super.configureFlutterEngine(flutterEngine)
    val monitor = AudioSessionMonitor(applicationContext)
    audioMonitor = monitor
    EventChannel(flutterEngine.dartExecutor.binaryMessenger, "persona_voice/audio_session")
        .setStreamHandler(monitor)
  }
}
