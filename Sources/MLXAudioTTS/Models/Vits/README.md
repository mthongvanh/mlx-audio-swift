# MMS-TTS (VITS)

Meta's [Massively Multilingual Speech](https://huggingface.co/facebook/mms-tts) voices:
one small VITS model (36M parameters, 16 kHz) for each of about 1,100 languages. Any
other VITS checkpoint in transformers' format loads too, as it is, with no conversion.

[Hugging Face: `facebook/mms-tts-*`](https://huggingface.co/models?search=facebook/mms-tts)

**Licence.** The MMS weights are **CC-BY-NC 4.0**: no commercial use, and anything
fine-tuned from them carries the same terms. VITS itself, and this port, are MIT.

## Swift Example

```swift
import MLXAudioTTS

let model = try await VitsModel.fromPretrained("facebook/mms-tts-eng")
let audio = try await model.generate(
    text: "Hello from MLX.", voice: nil, refAudio: nil, refText: nil, language: nil)
```

`TTS.loadModel(modelRepo:)` finds it too: the config's `model_type` is `vits`.

- `speakingRate` on the model sets the speed (above 1 is faster).
- A voice with several speakers takes the speaker's number as `voice`.
- Calling the model directly takes `noiseScale` (0.667) and `noiseScaleDuration`
  (0.8), how much the voice and its timing vary, and can be given the noise itself.

**Spell the text as the voice expects.** Each voice reads only the characters in its
`vocab.json`; others are dropped. `tokenizer.prepare(_:)` returns the text as the voice
will read it and the characters it dropped, so check a voice's vocabulary against real
text before trusting it. Voices whose config asks for romanised (`is_uroman`) or
phonemised text need that done first.

## Notes

- Ported from [mlx-audio](https://github.com/Blaizzy/mlx-audio)'s Python `vits` model,
  with the same weight names and layout. That port matches transformers' `VitsModel`
  within 6e-5 on the waveform, noise off.
- This one matches the Python port stage by stage on `facebook/mms-tts-blt`: the
  tokenizer exactly, the text encoder within 2e-6, durations 2e-5, the flow 2e-6, the
  decoder exactly, and the waveform within 3e-5 (noise off) and 1.3e-4 (the same
  noise). `Tests/VitsTests.swift` says how to run those checks; the fixtures come from
  the Python port's `vits/parity/export_fixtures.py`.
- In a padded batch, items shorter than the longest end slightly differently than
  alone, since the decoder's convolutions see their biases in the padding. transformers
  behaves the same way.
- Speaking only, for now. Fine-tuning exists in the Python port.
