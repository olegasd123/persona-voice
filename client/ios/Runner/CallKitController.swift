import CallKit
import Flutter
import Foundation

/// Presents a Persona-Voice conversation as a native system call via **CallKit**, so it shows in
/// the iOS in-call / lock-screen UI and the OS call list, and the system end/mute buttons drive
/// the Flutter session. This is the iOS half of the telephony layer (`persona_voice/telephony`
/// MethodChannel for app→OS commands, `persona_voice/telephony_events` EventChannel for the
/// reverse).
///
/// We model the conversation as an **outgoing** call (the user initiates by connecting): a
/// `CXStartCallAction` puts it on the system call list, `reportOutgoingCall(...connectedAt:)`
/// flips it to "connected" once LiveKit is up. End/mute requests from the system UI arrive
/// through the `CXProviderDelegate` and are forwarded to Dart.
///
/// We let LiveKit/WebRTC keep owning the actual `AVAudioSession`; CallKit here is presentation +
/// control surface, not audio configuration.
final class CallKitController: NSObject, CXProviderDelegate, FlutterStreamHandler {
  private let provider: CXProvider
  private let callController = CXCallController()
  private var sink: FlutterEventSink?
  // Calls are keyed by the UUID minted in Dart (a UUID-v4 string), so every command and every
  // delegate callback refers to the same call.
  private var activeCallId: UUID?

  /// Wire up both channels against the engine's binary messenger. Call once at launch.
  static func register(with messenger: FlutterBinaryMessenger) -> CallKitController {
    let controller = CallKitController()
    let methods = FlutterMethodChannel(name: "persona_voice/telephony", binaryMessenger: messenger)
    methods.setMethodCallHandler { [weak controller] call, result in
      controller?.handle(call, result: result)
    }
    let events = FlutterEventChannel(
      name: "persona_voice/telephony_events", binaryMessenger: messenger)
    events.setStreamHandler(controller)
    return controller
  }

  override init() {
    // `CXProviderConfiguration(localizedName:)` keeps the iOS 13.0 deployment target (the no-arg
    // initializer is iOS 14+); the iOS-14 deprecation is a harmless warning.
    let config = CXProviderConfiguration(localizedName: "Persona Voice")
    config.supportsVideo = false
    config.maximumCallsPerCallGroup = 1
    config.maximumCallGroups = 1
    config.supportedHandleTypes = [.generic]
    provider = CXProvider(configuration: config)
    super.init()
    provider.setDelegate(self, queue: nil)
  }

  // MARK: - MethodChannel (app → OS)

  private func handle(_ call: FlutterMethodCall, result: @escaping FlutterResult) {
    let args = call.arguments as? [String: Any] ?? [:]
    switch call.method {
    case "startCall":
      startCall(args)
      result(nil)
    case "reportConnected":
      reportConnected(args)
      result(nil)
    case "endCall":
      endCall(args)
      result(nil)
    case "setMuted":
      setMuted(args)
      result(nil)
    default:
      result(FlutterMethodNotImplemented)
    }
  }

  private func startCall(_ args: [String: Any]) {
    guard let uuid = callId(from: args) else { return }
    activeCallId = uuid
    let displayName = args["displayName"] as? String ?? "Persona Voice"
    let handleValue = args["handle"] as? String ?? "persona"
    let handle = CXHandle(type: .generic, value: handleValue)

    let startAction = CXStartCallAction(call: uuid, handle: handle)
    startAction.contactIdentifier = displayName
    callController.request(CXTransaction(action: startAction)) { [weak self] error in
      if let error = error {
        NSLog("CallKit startCall failed: \(error.localizedDescription)")
        return
      }
      let update = CXCallUpdate()
      update.remoteHandle = handle
      update.localizedCallerName = displayName
      update.hasVideo = false
      update.supportsHolding = false
      update.supportsGrouping = false
      update.supportsUngrouping = false
      self?.provider.reportCall(with: uuid, updated: update)
      self?.provider.reportOutgoingCall(with: uuid, startedConnectingAt: nil)
    }
  }

  private func reportConnected(_ args: [String: Any]) {
    guard let uuid = callId(from: args) ?? activeCallId else { return }
    provider.reportOutgoingCall(with: uuid, connectedAt: nil)
  }

  private func endCall(_ args: [String: Any]) {
    guard let uuid = callId(from: args) ?? activeCallId else { return }
    let action = CXEndCallAction(call: uuid)
    callController.request(CXTransaction(action: action)) { error in
      if let error = error {
        NSLog("CallKit endCall failed: \(error.localizedDescription)")
      }
    }
    if uuid == activeCallId { activeCallId = nil }
  }

  private func setMuted(_ args: [String: Any]) {
    guard let uuid = callId(from: args) ?? activeCallId else { return }
    let muted = args["muted"] as? Bool ?? false
    let action = CXSetMutedCallAction(call: uuid, muted: muted)
    callController.request(CXTransaction(action: action)) { error in
      if let error = error {
        NSLog("CallKit setMuted failed: \(error.localizedDescription)")
      }
    }
  }

  private func callId(from args: [String: Any]) -> UUID? {
    guard let raw = args["callId"] as? String else { return nil }
    return UUID(uuidString: raw)
  }

  // MARK: - CXProviderDelegate (OS → app)

  func providerDidReset(_ provider: CXProvider) {
    activeCallId = nil
    send(["event": "endCall"])
  }

  func provider(_ provider: CXProvider, perform action: CXStartCallAction) {
    // We initiated this, so just acknowledge; the actual connect is driven from Dart/LiveKit.
    action.fulfill()
  }

  func provider(_ provider: CXProvider, perform action: CXEndCallAction) {
    if action.callUUID == activeCallId { activeCallId = nil }
    send(["event": "endCall"])
    action.fulfill()
  }

  func provider(_ provider: CXProvider, perform action: CXSetMutedCallAction) {
    send(["event": "setMuted", "muted": action.isMuted])
    action.fulfill()
  }

  // MARK: - FlutterStreamHandler

  func onListen(withArguments arguments: Any?, eventSink events: @escaping FlutterEventSink)
    -> FlutterError?
  {
    sink = events
    return nil
  }

  func onCancel(withArguments arguments: Any?) -> FlutterError? {
    sink = nil
    return nil
  }

  private func send(_ payload: [String: Any]) {
    // CXProviderDelegate callbacks arrive on the provider's queue; hop to main for the sink.
    if Thread.isMainThread {
      sink?(payload)
    } else {
      DispatchQueue.main.async { [weak self] in self?.sink?(payload) }
    }
  }
}
