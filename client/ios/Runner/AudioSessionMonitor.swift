import AVFoundation
import Flutter

/// Bridges `AVAudioSession` interruption + route-change notifications to the Flutter
/// `persona_voice/audio_session` EventChannel. This is the "deep audio-session plumbing"
/// half of M6: the thin client reacts to incoming phone/VoIP calls, Siri, alarms, and
/// headset/Bluetooth changes that the bare WebRTC stack doesn't surface to Dart.
///
/// We only *observe* — LiveKit/WebRTC still owns configuring and activating the session, so we
/// never reconfigure the category here and can't fight its audio management.
final class AudioSessionMonitor: NSObject, FlutterStreamHandler {
  private var sink: FlutterEventSink?

  /// Wire the channel up against the engine's binary messenger. Call once at launch.
  static func register(with messenger: FlutterBinaryMessenger) -> AudioSessionMonitor {
    let monitor = AudioSessionMonitor()
    let channel = FlutterEventChannel(name: "persona_voice/audio_session", binaryMessenger: messenger)
    channel.setStreamHandler(monitor)
    return monitor
  }

  // MARK: FlutterStreamHandler

  func onListen(withArguments arguments: Any?, eventSink events: @escaping FlutterEventSink) -> FlutterError? {
    sink = events
    let center = NotificationCenter.default
    center.addObserver(
      self, selector: #selector(handleInterruption(_:)),
      name: AVAudioSession.interruptionNotification, object: nil)
    center.addObserver(
      self, selector: #selector(handleRouteChange(_:)),
      name: AVAudioSession.routeChangeNotification, object: nil)
    // Emit the current route immediately so the UI starts in sync.
    emitCurrentRoute()
    return nil
  }

  func onCancel(withArguments arguments: Any?) -> FlutterError? {
    NotificationCenter.default.removeObserver(self)
    sink = nil
    return nil
  }

  // MARK: Notifications

  @objc private func handleInterruption(_ note: Notification) {
    guard
      let info = note.userInfo,
      let raw = info[AVAudioSessionInterruptionTypeKey] as? UInt,
      let type = AVAudioSession.InterruptionType(rawValue: raw)
    else { return }

    switch type {
    case .began:
      send(["event": "interruptionBegan"])
    case .ended:
      var shouldResume = false
      if let optsRaw = info[AVAudioSessionInterruptionOptionKey] as? UInt {
        shouldResume = AVAudioSession.InterruptionOptions(rawValue: optsRaw).contains(.shouldResume)
      }
      send(["event": "interruptionEnded", "shouldResume": shouldResume])
    @unknown default:
      break
    }
  }

  @objc private func handleRouteChange(_ note: Notification) {
    emitCurrentRoute()
  }

  // MARK: Helpers

  private func emitCurrentRoute() {
    let route = AVAudioSession.sharedInstance().currentRoute
    send(["event": "routeChanged", "route": Self.routeName(route)])
  }

  /// Collapse the active output to the coarse `AudioRoute` the Dart side models.
  private static func routeName(_ route: AVAudioSessionRouteDescription) -> String {
    guard let port = route.outputs.first?.portType else { return "unknown" }
    switch port {
    case .builtInSpeaker:
      return "speaker"
    case .builtInReceiver:
      return "receiver"
    case .headphones, .usbAudio:
      return "headphones"
    case .bluetoothA2DP, .bluetoothLE, .bluetoothHFP:
      return "bluetooth"
    case .carAudio:
      return "carAudio"
    default:
      return "unknown"
    }
  }

  private func send(_ payload: [String: Any]) {
    // Interruption/route notifications can arrive off the main thread; the event sink must be
    // called on the platform thread.
    if Thread.isMainThread {
      sink?(payload)
    } else {
      DispatchQueue.main.async { [weak self] in self?.sink?(payload) }
    }
  }
}
