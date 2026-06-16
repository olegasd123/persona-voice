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

## Layout

```
lib/
├── main.dart                       # app + theme
├── models/
│   ├── persona.dart                # {id, name} from /personas
│   └── connection_settings.dart    # token-server URL / api token / identity (persisted)
├── services/
│   ├── token_client.dart           # HTTP client for /personas and /token
│   └── voice_session.dart          # LiveKit Room wrapper (connect, mic, transcript, switch)
└── screens/
    ├── home_screen.dart            # settings + persona picker + connect
    └── call_screen.dart            # live call: status, mic, transcript, switch, hang up
```

## Status & remaining work

- ✅ Connect via the token server, publish mic, persona picker + mid-call switch, mute,
  transcript view (renders LiveKit `TranscriptionEvent`s), hang up.
- ✅ Mic permission (iOS `NSMicrophoneUsageDescription`, Android `RECORD_AUDIO`), background
  audio modes, cleartext for local dev, `minSdk 23` / iOS 13 for `flutter_webrtc`.
- `flutter analyze` clean; `flutter test` green (token client + models, via a mock HTTP
  client). The **live device run** needs a running LiveKit server + the agent, the analog of
  the server's other "live LiveKit" steps.
- Remaining (the deep part of M6): CallKit (iOS) / ConnectionService (Android), audio-session
  interruption + route/Bluetooth handling, reconnection, and surfacing the assistant
  transcript (the custom agent logs transcripts today; publishing them over LiveKit lights up
  the transcript view fully).
