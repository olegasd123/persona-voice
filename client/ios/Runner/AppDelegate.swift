import Flutter
import UIKit

@main
@objc class AppDelegate: FlutterAppDelegate, FlutterImplicitEngineDelegate {
  // Held strong for the app's lifetime so its NotificationCenter observers stay live.
  private var audioMonitor: AudioSessionMonitor?
  // Held strong so its CXProvider/CXCallController and channel handlers stay live.
  private var callKitController: CallKitController?

  override func application(
    _ application: UIApplication,
    didFinishLaunchingWithOptions launchOptions: [UIApplication.LaunchOptionsKey: Any]?
  ) -> Bool {
    return super.application(application, didFinishLaunchingWithOptions: launchOptions)
  }

  func didInitializeImplicitFlutterEngine(_ engineBridge: FlutterImplicitEngineBridge) {
    let registry = engineBridge.pluginRegistry
    GeneratedPluginRegistrant.register(with: registry)
    if let messenger = registry.registrar(forPlugin: "AudioSessionMonitor")?.messenger() {
      audioMonitor = AudioSessionMonitor.register(with: messenger)
    }
    if let messenger = registry.registrar(forPlugin: "CallKitController")?.messenger() {
      callKitController = CallKitController.register(with: messenger)
    }
  }
}
