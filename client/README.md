# Persona Voice — Flutter client (M6)

A thin cross-platform (iOS + Android) client for the Persona-Voice server, built on the
[LiveKit Flutter SDK](https://pub.dev/packages/livekit_client). It captures the mic, plays
the assistant's voice, shows a transcript, and lets you pick / switch personas — all model
work stays on the server.

## How it connects

The client never holds LiveKit secrets. It talks to two things:

1. **Token server** (`personavoice --token-server`, default `http://<host>:8080`) — the app
   fetches `/personas`, then `POST /token` for a room join token. The response also carries
   the **LiveKit URL** to dial.
2. **LiveKit SFU** — the app connects there with that URL + token and publishes its mic. The
   server's agent worker (`personavoice --serve`) auto-joins the room and drives
   STT → LLM → TTS, streaming the reply back as audio (with barge-in).

Persona selection rides a LiveKit **data message**: on connect (and on switch) the client
publishes `{"persona": "<id>"}`, which the agent's `data_received` handler swaps to without
dropping the call (conversation history is preserved server-side).

```
Flutter app ──HTTP──▶ token server  ──▶ {url, token, persona}
     │
     └──WebRTC(mic/audio + data msg)──▶ LiveKit SFU ◀──── agent worker (STT→LLM→TTS)
```

## Run it

Bring up the server side (see the repo README "Self-hosted LiveKit (M6)"):

```bash
# from the repo root
docker compose -f docker-compose.livekit.yml up --build      # LiveKit SFU + token server
LIVEKIT_URL=ws://localhost:7880 LIVEKIT_API_KEY=devkey \
  LIVEKIT_API_SECRET=secret BACKEND=mac personavoice --serve  # the conversation agent
```

Then the app:

```bash
cd client
flutter pub get
flutter run                # pick an attached device / simulator
```

In the app: set the **token server URL** (use the machine's LAN IP from a real phone, not
`localhost`), tap **Load personas**, pick one, then **Connect & talk**.

In a call you can switch between **Open mic** (the server's VAD decides turns; tap the mic to
mute) and **Push to talk** (hold the button to speak, release to send). The assistant's words
stream into the transcript as it speaks, and the status line shows reconnects. If a phone call
or Siri interrupts, the mic auto-mutes and the status shows **Paused (interrupted)**, then
resumes (in open-mic) when the interruption clears; the route indicator reflects a headset or
Bluetooth output. The conversation also registers as a **system call** (CallKit / Android
self-managed ConnectionService), so it appears on the lock screen / OS call list — ending or
muting it from there controls the session.

## Layout

```
lib/
├── main.dart                       # app + theme
├── models/
│   ├── persona.dart                # {id, name} from /personas
│   └── connection_settings.dart    # token-server URL / api token / identity (persisted)
├── services/
│   ├── token_client.dart           # HTTP client for /personas and /token
│   ├── audio_session.dart          # native interruption/route events (platform channel)
│   ├── telephony.dart              # present the call as a system call (CallKit / ConnectionService)
│   └── voice_session.dart          # LiveKit Room wrapper (connect, mic, transcript, switch)
└── screens/
    ├── home_screen.dart            # settings + persona picker + connect
    └── call_screen.dart            # live call: status, mic, transcript, switch, hang up
```

### Native audio-session plumbing

`audio_session.dart` bridges a platform `EventChannel` (`persona_voice/audio_session`) to typed
Dart events. The native sides observe the OS audio session and normalize it:

- **iOS** (`ios/Runner/AudioSessionMonitor.swift`, registered in `AppDelegate`): `AVAudioSession`
  `interruptionNotification` (incoming call / Siri → began/ended with the `shouldResume` hint)
  and `routeChangeNotification` (headset / Bluetooth → the active output).
- **Android** (`android/.../AudioSessionMonitor.kt`, registered in `MainActivity`): an
  `OnAudioFocusChangeListener` (focus loss/gain → interruption began/ended) and an
  `AudioDeviceCallback` (output device add/remove → route).

`VoiceSession` consumes these: an interruption mutes the mic (and auto-resumes a previously-live
mic in open-mic mode once it clears); a route change updates the indicator. The handler is pure
and unit-tested room-less; the native monitors only *observe* (LiveKit/WebRTC still owns
configuring + activating the session).

### Native telephony (present the call as a system call)

`telephony.dart` registers the live conversation *as* an OS call, so it shows in the system
in-call / lock-screen UI and the OS call list, and the system **end** + **mute** buttons drive
the session. Two channels: a `MethodChannel` (`persona_voice/telephony`) for app → OS commands
(`startCall` / `reportConnected` / `endCall` / `setMuted`) and an `EventChannel`
(`persona_voice/telephony_events`) for the reverse (`endCall`, `setMuted`). The call id is a
UUID minted in Dart so both native sides stay trivial.

- **iOS** (`ios/Runner/CallKitController.swift`, registered in `AppDelegate`): **CallKit** —
  models the conversation as an outgoing `CXStartCallAction`, flips to connected with
  `reportOutgoingCall(...connectedAt:)`, and forwards `CXEndCallAction` / `CXSetMutedCallAction`
  from the `CXProviderDelegate` to Dart.
- **Android** (`android/.../PersonaConnectionService.kt` + `TelephonyController.kt`, registered in
  `MainActivity` + the manifest): a **self-managed `ConnectionService`** (API 26+; a no-op on
  23–25). `TelecomManager.placeCall` creates a self-managed `Connection` whose `onDisconnect` /
  `onCallAudioStateChanged` are relayed to Dart via a small bridge singleton.

`VoiceSession` starts/ends the system call across connect/teardown, hangs up on a system
`EndCallRequested`, mirrors a system `MuteRequested` onto the mic, and pushes its own mic mutes
back to the system button — with an echo guard so the two can't ping-pong. All of this is pure
and unit-tested room-less with a fake controller; the native CallKit/Telecom compile + behavior
is **device-verify pending** (no Android SDK / device in dev). On Android, app→system *mute* is a
no-op (self-managed connections can't set mute programmatically — the WebRTC track is already
muted, and the system UI owns the toggle).

## Status & remaining work

- ✅ Connect via the token server, publish mic, persona picker + mid-call switch, hang up.
- ✅ **Mic modes**: open-mic (server VAD) and push-to-talk (hold to talk). Mute in open-mic.
- ✅ **Live transcript both ways**: renders LiveKit `TranscriptionEvent`s; the agent now
  publishes its spoken reply as a growing transcript (`orchestrator/agent.py`
  `make_transcript_publisher`), so the assistant's words appear, not just the user's.
- ✅ **Audio routing & reconnection**: audio defaults to the loudspeaker
  (`AudioOutputOptions(speakerOn: true)`); the status line surfaces LiveKit reconnect/resume.
- ✅ **Audio-session interruption + route handling** (platform channel): incoming-call/Siri
  interruption auto-mutes + auto-resumes the mic; headset/Bluetooth route changes surface to the
  UI. iOS `AVAudioSession` + Android `AudioManager` monitors → `audio_session.dart` → a pure,
  unit-tested `VoiceSession` handler. *Native Swift/Kotlin compile + behavior is device-verify
  pending* (no Android SDK / device in dev); the Dart layer is fully analyzed + tested.
- ✅ **Native telephony** (platform channel): present the conversation as a system call via
  iOS **CallKit** / Android self-managed **ConnectionService** — system end/mute buttons drive
  the session; app mic mutes sync back. Dart layer pure + unit-tested; native compile is
  device-verify pending. (Distinct from the interruption handling above, which is being
  *interrupted by* a call — this is presenting *our* session *as* one.)
- ✅ Mic permission (iOS `NSMicrophoneUsageDescription`, Android `RECORD_AUDIO`), background
  audio modes (`audio` + `voip`), `MANAGE_OWN_CALLS`, cleartext for local dev, `minSdk 23` /
  iOS 13 for `flutter_webrtc`.
- `flutter analyze` clean; `flutter test` green (37 tests: token client, models, status-label,
  mic-mode + interruption/route + system-call logic via a mock HTTP client and a room-less
  `VoiceSession`, and the audio-event / call-control parsers). The native Swift/Kotlin compile +
  behavior is **device-verify pending** (no Android SDK / device in dev).
- Remaining (device-only): the **live spoken run** from a real iPhone + Android device against a
  running LiveKit server + the agent (the analog of the server's other "live LiveKit" steps),
  which also exercises/tunes the native audio-session + CallKit/ConnectionService monitors.
