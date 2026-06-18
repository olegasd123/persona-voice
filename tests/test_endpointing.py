"""VAD endpointing tuning resolved from the environment."""

from __future__ import annotations

from personavoice.orchestrator.endpointing import (
    VadTuning,
    vad_load_kwargs,
    vad_tuning_from_env,
)


def test_empty_env_leaves_silero_defaults() -> None:
    assert vad_tuning_from_env({}) == VadTuning()
    assert vad_load_kwargs(VadTuning()) == {}


def test_durations_are_converted_ms_to_seconds() -> None:
    tuning = vad_tuning_from_env(
        {
            "PERSONAVOICE_VAD_MIN_SILENCE_MS": "300",
            "PERSONAVOICE_VAD_MIN_SPEECH_MS": "50",
            "PERSONAVOICE_VAD_PREFIX_PADDING_MS": "400",
        }
    )
    assert tuning.min_silence_ms == 300
    kwargs = vad_load_kwargs(tuning)
    assert kwargs == {
        "min_silence_duration": 0.3,
        "min_speech_duration": 0.05,
        "prefix_padding_duration": 0.4,
    }


def test_activation_threshold_in_unit_range() -> None:
    assert vad_tuning_from_env({"PERSONAVOICE_VAD_ACTIVATION_THRESHOLD": "0.6"}).activation_threshold == 0.6
    # Out of [0, 1] is ignored (treated as unset).
    assert vad_tuning_from_env({"PERSONAVOICE_VAD_ACTIVATION_THRESHOLD": "1.5"}).activation_threshold is None


def test_invalid_values_are_ignored() -> None:
    tuning = vad_tuning_from_env(
        {"PERSONAVOICE_VAD_MIN_SILENCE_MS": "soon", "PERSONAVOICE_VAD_MIN_SPEECH_MS": "-5"}
    )
    assert tuning == VadTuning()
    assert vad_load_kwargs(tuning) == {}


def test_partial_config_only_sets_provided_kwargs() -> None:
    kwargs = vad_load_kwargs(vad_tuning_from_env({"PERSONAVOICE_VAD_MIN_SILENCE_MS": "200"}))
    assert kwargs == {"min_silence_duration": 0.2}
