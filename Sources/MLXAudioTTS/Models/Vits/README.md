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

## Fine-tuning

A port of the Python port's trainer, itself
[finetune-hf-vits](https://github.com/ylacombe/finetune-hf-vits) (MIT) step for step:
the same losses and weights, discriminator then generator, AdamW with PyTorch's
defaults, the learning rate decayed once an epoch. It needs a checkpoint carrying the
discriminator, as finetune-hf-vits's `convert_original_discriminator_checkpoint.py`
makes (it needs PyTorch, once per language).

```swift
let (model, discriminator) = try VitsModel.loadForTraining(checkpoint)
let trainer = VitsTrainer(model: model, discriminator: discriminator, config: VitsTrainingConfig())
let clips = try trainer.loadClips(folder: data)  // clips and a metadata.jsonl of {"file_name", "text"}
trainer.train(clips) { epoch, step, losses in print(step, losses.total); return true }
try model.saveTrained(from: checkpoint, to: output)
```

- `trainer.clip(name:audio:text:)` makes a clip from audio already in hand.
- `saveTrained` writes a plain transformers checkpoint (weight norm folded, no
  discriminator): it loads here, in the Python port, and in transformers 4.46 and 5.19.
- One step, against the Python port on the same batch and noise, on `blt`'s training
  checkpoint: every loss within 3e-6, the alignment identical, and the gradients of the
  text encoder, duration predictor and flow within 1e-5. The decoder's and posterior
  encoder's gradients differ by up to 1.6%: given the Python port's exact input, the
  decoder's match within 1.3e-6, and its own input, 1e-7 off, moves them by 0.5%. That
  is rounding amplified by the decoder's leaky ReLUs and the losses' `abs`, as between
  the Python port and PyTorch.
- 15 steps at batch 8 on 40 clips the stock voice made (M2 Max, a debug build): about
  3.3 s a step, 7.4 GB at MLX's peak. The voice reads back at 1.8% character error with
  MMS's Tai Dam recogniser, as before training.
- Weight norm is part of how a model is built here (`forTraining: true`), not switched
  on afterwards: MLX Swift modules can't change their parameters once made.
- Layer drop (0.1 in MMS checkpoints) makes the KL loss spike now and then, in every
  port; `model.layerdrop = 0` stops it.

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
