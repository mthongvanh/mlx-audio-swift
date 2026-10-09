//  VITS (MMS-TTS): offline tests on a tiny random model, and parity tests
//  against fixtures from mlx-audio's Python port.
//
//  The parity tests run when VITS_PARITY_FIXTURES names a folder that
//  mlx-audio's `vits/parity/export_fixtures.py` wrote, and VITS_PARITY_MODEL
//  names the same voice's folder (config.json, vocab.json, safetensors).
//  Through xcodebuild, prefix each with TEST_RUNNER_:
//
//    TEST_RUNNER_VITS_PARITY_FIXTURES=<dir> TEST_RUNNER_VITS_PARITY_MODEL=<dir> \
//    xcodebuild test -scheme MLXAudio-Package -destination 'platform=macOS' \
//      -only-testing:MLXAudioTests/VitsParityTests CODE_SIGNING_ALLOWED=NO
//
//  The training parity tests also need VITS_PARITY_TRAIN_MODEL, the training
//  checkpoint the fixtures were made from (mlx-audio's `parity/setup.sh`
//  makes it, `export_train_fixtures.py` the fixtures). The short fine-tune
//  needs it too, with VITS_FINETUNE_DATA (clips and metadata.jsonl) and
//  VITS_FINETUNE_OUTPUT (where the voice goes).

import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXAudioTTS

// MARK: - Helpers

private func tinyConfig(speakers: Int = 1) -> VitsConfig {
    var config = VitsConfig()
    config.vocabSize = 12
    config.hiddenSize = 16
    config.numHiddenLayers = 2
    config.numAttentionHeads = 2
    config.windowSize = 2
    config.ffnDim = 24
    config.flowSize = 8
    config.spectrogramBins = 9
    config.upsampleInitialChannel = 16
    config.upsampleRates = [4, 2]
    config.upsampleKernelSizes = [8, 4]
    config.resblockKernelSizes = [3, 5]
    config.resblockDilationSizes = [[1, 3], [1, 3]]
    config.durationPredictorFilterChannels = 16
    config.durationPredictorFlowBins = 4
    config.durationPredictorNumFlows = 2
    config.priorEncoderNumFlows = 2
    config.priorEncoderNumWavenetLayers = 2
    config.posteriorEncoderNumWavenetLayers = 2
    config.wavenetKernelSize = 3
    if speakers > 1 {
        config.numSpeakers = speakers
        config.speakerEmbeddingSize = 4
    }
    return config
}

/// Small random weights everywhere, so every layer does something.
private func randomise(_ model: VitsModel, seed: UInt64 = 0) throws {
    MLXRandom.seed(seed)
    var params: [String: MLXArray] = [:]
    for (key, value) in model.parameters().flattened() {
        let noise = MLXRandom.normal(value.shape) * 0.2
        if key.hasSuffix("weight_g") {
            params[key] = abs(noise) + 0.5
        } else if key.contains("norm") && key.hasSuffix("weight") {
            params[key] = noise + 1
        } else {
            params[key] = noise
        }
    }
    try model.update(parameters: ModuleParameters.unflattened(params), verify: .all)
    model.train(false)
}

private func maxDifference(_ a: MLXArray, _ b: MLXArray) -> Float {
    abs(a.asType(.float32) - b.asType(.float32)).max().item(Float.self)
}

private func tinyTokenizer() -> VitsTokenizer {
    let letters = [" ", "a", "b", "c", "d", "e", "ê", "'", "‐", "k", "h"]
    var vocabulary: [String: Int] = ["_": 0]
    for (i, letter) in letters.enumerated() { vocabulary[letter] = i + 1 }
    return VitsTokenizer(vocabulary: vocabulary, order: ["_"] + letters)
}

// MARK: - Offline

@Suite("VITS")
struct VitsTests {
    @Test func tokenizerPutsBlanksBetweenCharacters() throws {
        let ids = try tinyTokenizer().encode("ab")
        #expect(ids == [0, 2, 0, 3, 0])
    }

    @Test func tokenizerLowersAndDropsWhatTheVoiceLacks() throws {
        let tokenizer = tinyTokenizer()
        let prepared = try tokenizer.prepare("ꞌHa#ꞌkê, z")
        #expect(prepared.text == "hakê")
        #expect(prepared.dropped == ["#", ",", "z", "ꞌ"])
        #expect(try tokenizer.encode("") == [0])
    }

    @Test func tokenizerMatchesInTheVocabularysOwnOrder() throws {
        // A token matched first is kept as it is; the rest is lowered.
        let first = VitsTokenizer(vocabulary: ["A": 1, "AB": 2], order: ["A", "AB"])
        let second = VitsTokenizer(vocabulary: ["A": 1, "AB": 2], order: ["AB", "A"])
        #expect(first.normalized("AB") == "Ab")
        #expect(second.normalized("AB") == "AB")
    }

    @Test func tokenizerReadsTheVocabularyInFileOrder() throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("vits-vocab-\(UUID())")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        try #"{"z": 0, "AB": 3, "ê": 1, "A": 2}"#.write(
            to: dir.appendingPathComponent("vocab.json"), atomically: true, encoding: .utf8)
        let (values, order) = try VitsTokenizer.readOrderedObject(dir.appendingPathComponent("vocab.json"))
        #expect(order == ["z", "AB", "ê", "A"])
        #expect(values["ê"] == 1)
    }

    @Test func splineReversesItselfInsideAndIsTheIdentityOutside() {
        MLXRandom.seed(1)
        let shape = [2, 7, 1]
        let x = MLXArray([-6, -4.9, -1, 0, 0.3, 2.5, 6] as [Float]).reshaped(1, 7, 1)
        let inputs = concatenated([x, x * 0.5], axis: 0)
        let widths = MLXRandom.normal(shape + [4])
        let heights = MLXRandom.normal(shape + [4])
        let derivatives = MLXRandom.normal(shape + [3])
        let (y, logDet) = VitsSpline.unconstrained(
            inputs, widths: widths, heights: heights, derivatives: derivatives, reverse: false, tailBound: 5)
        let (back, logDetBack) = VitsSpline.unconstrained(
            y, widths: widths, heights: heights, derivatives: derivatives, reverse: true, tailBound: 5)
        #expect(maxDifference(back, inputs) < 1e-4)
        #expect(maxDifference(logDet + logDetBack, MLXArray.zeros(logDet.shape)) < 1e-4)
        // ±6 is outside the interval: passed through, with no log-determinant.
        #expect(y[0, 0, 0].item(Float.self) == -6)
        #expect(logDet[0, 6, 0].item(Float.self) == 0)
    }

    @Test func modelGivesEachItemItsLength() throws {
        let model = try VitsModel(tinyConfig())
        try randomise(model)
        let ids = MLXArray([0, 3, 0, 5, 0, 7, 0] as [Int32]).reshaped(1, -1)
        let (waveform, lengths) = model(ids, noiseScale: 0, noiseScaleDuration: 0)
        let length = lengths[0].item(Int.self)
        #expect(length > 0 && length % model.config.hopLength == 0)
        #expect(waveform.dim(1) == length)
        #expect(!waveform.asArray(Float.self).contains { !$0.isFinite })
    }

    @Test func paddedBatchMatchesItsLongestItemAlone() throws {
        let model = try VitsModel(tinyConfig())
        try randomise(model)
        let long: [Int32] = [0, 3, 0, 5, 0, 7, 0, 2, 0]
        let short: [Int32] = [0, 4, 0, 6, 0]
        let single = model(MLXArray(long).reshaped(1, -1), noiseScale: 0, noiseScaleDuration: 0)
        let singleShort = model(MLXArray(short).reshaped(1, -1), noiseScale: 0, noiseScaleDuration: 0)

        let batch = MLXArray(long + short + [Int32](repeating: 0, count: long.count - short.count))
            .reshaped(2, -1)
        let mask = MLXArray(
            [Float](repeating: 1, count: long.count) + [Float](repeating: 1, count: short.count)
                + [Float](repeating: 0, count: long.count - short.count)
        ).reshaped(2, -1)
        let (waveform, lengths) = model(batch, attentionMask: mask, noiseScale: 0, noiseScaleDuration: 0)
        let n0 = lengths[0].item(Int.self)
        let n1 = lengths[1].item(Int.self)
        #expect(n0 == single.lengths[0].item(Int.self))
        #expect(n1 == singleShort.lengths[0].item(Int.self))
        #expect(maxDifference(waveform[0, ..<n0], single.waveform[0, ..<n0]) < 1e-4)
        // Not the shorter item's audio: the decoder's convolutions see their
        // own biases in its padding, where alone it would see zeros, so its
        // end differs, as in transformers.
    }

    @Test func speakersChangeTheVoice() throws {
        let model = try VitsModel(tinyConfig(speakers: 3))
        try randomise(model)
        let ids = MLXArray([0, 3, 0, 5, 0, 7, 0] as [Int32]).reshaped(1, -1)
        let a = model(ids, speakerId: 0, noiseScale: 0, noiseScaleDuration: 0)
        let b = model(ids, speakerId: 2, noiseScale: 0, noiseScaleDuration: 0)
        let n = min(a.lengths[0].item(Int.self), b.lengths[0].item(Int.self))
        #expect(maxDifference(a.waveform[0, ..<n], b.waveform[0, ..<n]) > 1e-4)
    }

    @Test func sanitizeTakesPyTorchsLayoutAndItsOwn() throws {
        let model = try VitsModel(tinyConfig())
        try randomise(model)
        let own = Dictionary(uniqueKeysWithValues: model.parameters().flattened())

        // The same weights in transformers' names and PyTorch's layout.
        var torch: [String: MLXArray] = [:]
        for (key, value) in own {
            var key = key
            var value = value
            if key.hasSuffix(".translate") || key.hasSuffix(".log_scale") {
                value = value.reshaped(-1, 1)
            } else if value.ndim == 3, !key.hasSuffix("emb_rel_k"), !key.hasSuffix("emb_rel_v"),
                      !key.hasSuffix("weight_g")
            {
                value = key.hasPrefix("decoder.upsampler.") ? value.transposed(2, 0, 1) : value.transposed(0, 2, 1)
            }
            // Newer PyTorch names weight norm's halves differently.
            key = key.replacingOccurrences(of: ".weight_g", with: ".parametrizations.weight.original0")
                .replacingOccurrences(of: ".weight_v", with: ".parametrizations.weight.original1")
            torch[key] = value
        }
        torch["discriminator.discriminators.0.conv_post.weight"] = MLXArray.zeros([1, 2, 3])

        for weights in [torch, own] {
            let sanitized = model.sanitize(weights: weights)
            #expect(Set(sanitized.keys) == Set(own.keys))
            for (key, value) in own {
                #expect(sanitized[key]?.shape == value.shape, "\(key)")
                #expect(maxDifference(sanitized[key]!, value) == 0, "\(key)")
            }
        }
    }

    @Test func alignmentSearchIsMonotonic() {
        // Every frame on one token, tokens in order, each used.
        let frames = 9, tokens = 4
        var cost = [Float](repeating: 0, count: frames * tokens)
        for y in 0 ..< frames { for x in 0 ..< tokens { cost[y * tokens + x] = -Float(abs(y / 2 - x)) } }
        let path = vitsMaximumPath(cost, shape: (1, frames, tokens), frameLengths: [frames], textLengths: [tokens])
        var last = 0
        for y in 0 ..< frames {
            let row = Array(path[(y * tokens) ..< ((y + 1) * tokens)])
            #expect(row.reduce(0, +) == 1)
            let x = row.firstIndex(of: 1)!
            #expect(x >= last && x <= last + 1)
            last = x
        }
        #expect(last == tokens - 1)
    }

    @Test func factoryKnowsMMSVoices() {
        #expect(TTS.resolveModelType(modelRepo: "facebook/mms-tts-blt") == "vits")
        #expect(TTS.resolveModelType(modelRepo: "x/y", modelType: "vits") == "vits")
    }
}

// MARK: - Parity with the Python port

private let parityFixtures = ProcessInfo.processInfo.environment["VITS_PARITY_FIXTURES"]
private let parityModel = ProcessInfo.processInfo.environment["VITS_PARITY_MODEL"]

@Suite(
    "VITS parity with mlx-audio's Python port",
    .serialized,
    .enabled(if: parityFixtures != nil && parityModel != nil, "set VITS_PARITY_FIXTURES and VITS_PARITY_MODEL")
)
struct VitsParityTests {
    let model: VitsModel
    let fixtures: [String: MLXArray]

    init() throws {
        model = try VitsModel.fromModelDirectory(URL(fileURLWithPath: parityModel!))
        fixtures = try loadArrays(
            url: URL(fileURLWithPath: parityFixtures!).appendingPathComponent("vits_fixtures.safetensors"))
    }

    private func check(_ name: String, _ actual: MLXArray, _ expected: MLXArray, within tolerance: Float) {
        #expect(actual.shape == expected.shape, "\(name): shape")
        guard actual.shape == expected.shape else { return }
        let difference = maxDifference(actual, expected)
        print("VITS parity  \(name.padding(toLength: 32, withPad: " ", startingAt: 0)) max|diff| \(difference)")
        #expect(difference <= tolerance, "\(name): \(difference) > \(tolerance)")
    }

    @Test func tokenizerMatches() throws {
        let url = URL(fileURLWithPath: parityFixtures!).appendingPathComponent("vits_tokenizer.json")
        struct Case: Decodable { let text: String; let ids: [Int] }
        struct Cases: Decodable { let cases: [Case] }
        let cases = try JSONDecoder().decode(Cases.self, from: Data(contentsOf: url)).cases
        #expect(!cases.isEmpty)
        for c in cases {
            #expect(try model.tokenizer!.encode(c.text) == c.ids, "\(c.text)")
        }
    }

    @Test func textEncoderMatches() {
        let ids = fixtures["ids"]!
        let length = ids.dim(1)
        let (hidden, means, logVariances) = model.textEncoder(
            ids, mask: MLXArray.ones([1, length, 1]), attentionMask: MLXArray.ones([1, length]))
        check("text encoder hidden", hidden, fixtures["text_encoder.hidden"]!, within: 1e-4)
        check("prior means", means, fixtures["text_encoder.prior_means"]!, within: 1e-4)
        check("prior log variances", logVariances, fixtures["text_encoder.prior_log_variances"]!, within: 1e-4)
    }

    @Test func durationsMatch() {
        let hidden = fixtures["text_encoder.hidden"]!
        let mask = MLXArray.ones([1, hidden.dim(1), 1])
        let quiet = model.logDurations(hidden, mask: mask, speaker: nil, noiseScale: 0, noise: nil)
        check("log durations, noise off", quiet, fixtures["duration.log_duration_quiet"]!, within: 1e-4)
        let drawn = model.logDurations(
            hidden, mask: mask, speaker: nil, noiseScale: model.config.noiseScaleDuration,
            noise: fixtures["duration.noise"]!)
        check("log durations, fixed noise", drawn, fixtures["duration.log_duration"]!, within: 1e-4)
    }

    @Test func flowMatches() {
        let input = fixtures["flow.input"]!
        let output = model.flow(input, mask: MLXArray.ones([1, input.dim(1), 1]), reverse: true)
        check("flow, reversed", output, fixtures["flow.output"]!, within: 1e-4)
    }

    @Test func decoderMatches() {
        let output = model.decoder(fixtures["flow.output"]!)
        check("decoder", output, fixtures["decoder.output"]!, within: 1e-4)
    }

    @Test func modelMatches() {
        let ids = fixtures["ids"]!
        let quiet = model(ids, noiseScale: 0, noiseScaleDuration: 0)
        #expect(quiet.lengths.asArray(Int32.self) == fixtures["model.lengths_quiet"]!.asArray(Int32.self))
        check("waveform, noise off", quiet.waveform, fixtures["model.waveform_quiet"]!, within: 1e-3)

        let drawn = model(
            ids, durationNoise: fixtures["duration.noise"]!, priorNoise: fixtures["model.prior_noise"]!)
        #expect(drawn.lengths.asArray(Int32.self) == fixtures["model.lengths"]!.asArray(Int32.self))
        check("waveform, fixed noise", drawn.waveform, fixtures["model.waveform"]!, within: 1e-3)
    }

    @Test func paddedBatchMatches() {
        let (waveform, lengths) = model(
            fixtures["batch.ids"]!, attentionMask: fixtures["batch.attention_mask"]!,
            noiseScale: 0, noiseScaleDuration: 0)
        #expect(lengths.asArray(Int32.self) == fixtures["batch.lengths"]!.asArray(Int32.self))
        check("batch of two, padded", waveform, fixtures["batch.waveform"]!, within: 1e-3)
    }

    @Test func generateSpeaks() async throws {
        let audio = try await model.generate(
            text: "té pang 'chạu", voice: nil, refAudio: nil, refText: nil, language: nil)
        #expect(audio.ndim == 1 && audio.dim(0) > model.sampleRate / 4)
        #expect(abs(audio).max().item(Float.self) > 0.01)
    }
}

// MARK: - Training parity with the Python port

private let trainFixtures = ProcessInfo.processInfo.environment["VITS_PARITY_FIXTURES"]
private let trainModel = ProcessInfo.processInfo.environment["VITS_PARITY_TRAIN_MODEL"]

/// One training step against fixtures from mlx-audio's
/// `vits/parity/export_train_fixtures.py`. VITS_PARITY_TRAIN_MODEL names
/// the training checkpoint it used (with the discriminator).
@Suite(
    "VITS training parity with mlx-audio's Python port",
    .serialized,
    .enabled(
        if: trainFixtures != nil && trainModel != nil, "set VITS_PARITY_FIXTURES and VITS_PARITY_TRAIN_MODEL")
)
struct VitsTrainingParityTests {
    let trainer: VitsTrainer
    let fixtures: [String: MLXArray]
    let batch: VitsTrainingBatch
    let noise: VitsTrainingNoise

    init() throws {
        let (model, discriminator) = try VitsModel.loadForTraining(URL(fileURLWithPath: trainModel!))
        // Dropout and layer drop off, as the fixtures were made.
        model.train(false)
        discriminator.train(false)
        var config = VitsTrainingConfig()
        config.batchSize = 2
        trainer = VitsTrainer(model: model, discriminator: discriminator, config: config)
        let f = try loadArrays(
            url: URL(fileURLWithPath: trainFixtures!).appendingPathComponent("vits_train_fixtures.safetensors"))
        fixtures = f
        batch = VitsTrainingBatch(
            inputIds: f["batch.input_ids"]!, attentionMask: f["batch.attention_mask"]!,
            labels: f["batch.labels"]!, labelsMask: f["batch.labels_attention_mask"]!,
            mel: f["batch.mel"]!, waveform: f["batch.waveform"]!)
        noise = VitsTrainingNoise(
            posterior: f["noise.posterior_noise"]!, duration: f["noise.duration_noise"]!,
            sliceStarts: f["noise.slice_starts"]!)
    }

    /// The relative error, |a - b| / |b|, over the whole tensor.
    private func relative(_ a: MLXArray, _ b: MLXArray) -> Float {
        let a = a.asType(.float32)
        let b = b.asType(.float32)
        return (sqrt(sum((a - b).square())) / (sqrt(sum(b.square())) + 1e-30)).item(Float.self)
    }

    private func check(_ name: String, _ actual: MLXArray, _ expected: MLXArray, within tolerance: Float) {
        #expect(actual.shape == expected.shape, "\(name): shape \(actual.shape) vs \(expected.shape)")
        guard actual.shape == expected.shape else { return }
        let error = relative(actual, expected)
        print("VITS training parity  \(name.padding(toLength: 30, withPad: " ", startingAt: 0)) rel \(error)")
        #expect(error <= tolerance, "\(name): \(error) > \(tolerance)")
    }

    @Test func spectrogramMatches() {
        let (magnitudes, mel) = trainer.spectrogram(batch.waveform[0 ..< 1])
        check("magnitudes", magnitudes, fixtures["spectrogram.magnitudes"]!, within: 1e-5)
        check("log-mel", mel, fixtures["spectrogram.mel"]!, within: 1e-5)
    }

    @Test func forwardMatches() {
        let out = vitsTrainingForward(
            trainer.model, batch: batch, segmentFrames: trainer.segmentFrames, noise: noise)
        check("alignment", out.attention, fixtures["forward.attn"]!, within: 0)
        check("log duration", out.logDuration, fixtures["forward.log_duration"]!, within: 1e-4)
        check("prior latents", out.priorLatents, fixtures["forward.prior_latents"]!, within: 1e-4)
        check("prior means", out.priorMeans, fixtures["forward.prior_means"]!, within: 1e-4)
        check("prior log variances", out.priorLogVariances, fixtures["forward.prior_log_variances"]!, within: 1e-4)
        check(
            "posterior log variances", out.posteriorLogVariances,
            fixtures["forward.posterior_log_variances"]!, within: 1e-4)
        check("waveform slice", out.waveform, fixtures["forward.waveform"]!, within: 1e-3)
    }

    @Test func lossesAndGradientsMatch() {
        let out = vitsTrainingForward(
            trainer.model, batch: batch, segmentFrames: trainer.segmentFrames, noise: noise)
        let (_, waveTarget) = trainer.targets(batch, out)
        let (disc, discGrads) = trainer.discriminatorGradients(fake: stopGradient(out.waveform), real: waveTarget)
        check("loss disc", disc[1], fixtures["loss.disc"]!, within: 1e-4)
        check("loss real disc", disc[2], fixtures["loss.real_disc"]!, within: 1e-4)
        check("loss fake disc", disc[3], fixtures["loss.fake_disc"]!, within: 1e-4)

        let (gen, genGrads) = trainer.generatorGradients(batch, noise)
        for (i, name) in ["total", "duration", "mel", "kl", "fmaps", "gen"].enumerated() {
            check("loss \(name)", gen[i], fixtures["loss.\(name)"]!, within: 1e-4)
        }
        func norm(_ g: ModuleParameters) -> MLXArray {
            sqrt(g.flattened().map { $0.1.square().sum() }.reduce(MLXArray(Float(0)), +))
        }
        check("discriminator grad norm", norm(discGrads), fixtures["grad_norm.disc"]!, within: 1e-4)
        // The decoder and posterior encoder's gradients come back through
        // the decoder, whose leaky ReLUs and the losses' `abs` amplify
        // rounding: an input 1e-7 different moves them by about 0.5% (see
        // the next test). Everything else matches to rounding.
        check("generator grad norm", norm(genGrads), fixtures["grad_norm.gen"]!, within: 5e-3)
        for (key, g) in genGrads.flattened() {
            let expected = fixtures["gen_grad_norm.\(key)"]!.item(Float.self)
            guard expected > 1e-6 else { continue }  // e.g. key biases, zero but for rounding
            let error = abs(sqrt(g.square().sum()).item(Float.self) - expected) / expected
            let sensitive = key.hasPrefix("decoder.") || key.hasPrefix("posterior_encoder.")
            #expect(error <= (sensitive ? 3e-2 : 1e-4), "gradient of \(key): \(error)")
        }
    }

    @Test func audioGradientsMatch() {
        let out = vitsTrainingForward(
            trainer.model, batch: batch, segmentFrames: trainer.segmentFrames, noise: noise)
        let (melTarget, waveTarget) = trainer.targets(batch, out)
        let gMel = grad({ (w: MLXArray) in
            mean(abs(melTarget - trainer.spectrogram(w[0..., 0..., 0]).mel))
        })(out.waveform)
        check("audio gradient, mel loss", gMel, fixtures["grad_wave.mel"]!, within: 1e-3)
        let gAdversarial = grad({ (w: MLXArray) in
            let (_, fmapsTarget) = trainer.discriminator(waveTarget)
            let (generated, fmapsGenerated) = trainer.discriminator(w)
            return vitsFeatureLoss(real: fmapsTarget, generated: fmapsGenerated) + vitsGeneratorLoss(generated)
        })(out.waveform)
        check("audio gradient, adversarial", gAdversarial, fixtures["grad_wave.adversarial"]!, within: 2e-3)
    }

    /// Given Python's decoder input and its gradient on the audio, the
    /// decoder's backward pass gives Python's gradients to rounding. Its own
    /// input, 1e-7 off Python's, moves them by about 0.5%: that, not the
    /// port, is the gap the other tests allow.
    @Test func decoderBackwardMatchesGivenTheSameInputs() {
        let model = trainer.model
        let labelsMask = batch.labelsMask[0..., 0..., .newAxis]
        let (latents, _, _) = model.posteriorEncoder(batch.labels, mask: labelsMask, noise: noise.posterior)
        let ownInput = vitsSliceSegments(latents, starts: noise.sliceStarts, size: trainer.segmentFrames)
        let input = fixtures["decoder.input"]!
        check("decoder input", ownInput, input, within: 1e-6)
        let c = trainer.config
        let cotangent = c.weightMel * fixtures["grad_wave.mel"]! + fixtures["grad_wave.adversarial"]!
        let vg = valueAndGrad(model: model.decoder) { (decoder: VitsHifiGan, _: Int) in
            [sum(decoder(input) * cotangent)]
        }
        let (_, grads) = vg(model.decoder, 0)
        var worst: Float = 0
        for (key, g) in grads.flattened() {
            let expected = fixtures["gen_grad_norm.decoder.\(key)"]!.item(Float.self)
            let actual = sqrt(g.square().sum()).item(Float.self)
            worst = max(worst, abs(actual - expected) / expected)
        }
        print("VITS training parity  decoder backward, Python's inputs: worst rel \(worst)")
        #expect(worst < 1e-4)
    }

    @Test func oneStepMatches() {
        // Copies: the optimiser updates the model's arrays in place.
        let before = Dictionary(uniqueKeysWithValues: trainer.model.parameters().flattened().map { ($0.0, $0.1 * 1) })
        eval(Array(before.values))
        trainer.setEpoch(0)
        let losses = trainer.step(batch, noise: noise)
        let after = Dictionary(uniqueKeysWithValues: trainer.model.parameters().flattened())

        let expected: [(String, Float)] = [
            ("duration", losses.duration), ("mel", losses.mel), ("kl", losses.kl), ("fmaps", losses.fmaps),
            ("gen", losses.gen), ("disc", losses.disc),
        ]
        for (name, value) in expected {
            check("step loss \(name)", MLXArray(value), fixtures["step.loss.\(name)"]!, within: 1e-4)
        }
        // Adam's first step is about the learning rate times each
        // gradient's sign, so rounding flips a few tiny gradients' updates:
        // compared as updates, not weights.
        var updateSquares = MLXArray(Float(0))
        for (key, value) in after {
            updateSquares = updateSquares + (value - before[key]!).square().sum()
        }
        check("update norm", sqrt(updateSquares), fixtures["step.update_norm"]!, within: 1e-3)
        for key in [
            "text_encoder.encoder.layers.0.attention.q_proj.weight",
            "duration_predictor.flows.1.conv_proj.weight",
            "flow.flows.0.conv_pre.weight_v",
            "decoder.upsampler.0.weight_v",
            "decoder.conv_post.weight",
        ] {
            // The decoder's gradients carry the rounding above, and Adam's
            // first step turns small differences in tiny gradients into
            // whole steps.
            let tolerance: Float = key.hasPrefix("decoder.") ? 0.2 : 1e-3
            check("update \(key)", after[key]! - before[key]!, fixtures["step.\(key)"]! - before[key]!, within: tolerance)
        }
    }
}

// MARK: - A short fine-tune

private let fineTuneData = ProcessInfo.processInfo.environment["VITS_FINETUNE_DATA"]
private let fineTuneOutput = ProcessInfo.processInfo.environment["VITS_FINETUNE_OUTPUT"]

/// A short fine-tune, saved for checking outside: VITS_PARITY_TRAIN_MODEL
/// is the training checkpoint, VITS_FINETUNE_DATA a folder of clips and
/// `metadata.jsonl`, VITS_FINETUNE_OUTPUT where the voice goes. Read it
/// back with mlx-audio's `vits/parity/asr_check.py --model <output>`.
@Suite(
    "VITS fine-tune",
    .serialized,
    .enabled(
        if: trainModel != nil && fineTuneData != nil && fineTuneOutput != nil,
        "set VITS_PARITY_TRAIN_MODEL, VITS_FINETUNE_DATA and VITS_FINETUNE_OUTPUT")
)
struct VitsFineTuneTests {
    @Test func fifteenStepsTrainAndSave() throws {
        let source = URL(fileURLWithPath: trainModel!)
        let (model, discriminator) = try VitsModel.loadForTraining(source)
        var config = VitsTrainingConfig()
        config.batchSize = 8
        config.epochs = 3
        let trainer = VitsTrainer(model: model, discriminator: discriminator, config: config)
        MLXRandom.seed(config.seed)
        let clips = try trainer.loadClips(folder: URL(fileURLWithPath: fineTuneData!))
        #expect(clips.count >= 8)

        let start = Date()
        var steps = 0
        GPU.resetPeakMemory()
        trainer.train(clips, maxSteps: 15) { epoch, step, losses in
            steps = step
            print(String(
                format: "VITS fine-tune  epoch %d step %d  total %.3f  mel %.3f  kl %.3f  disc %.3f  (%.0fs)",
                epoch, step, losses.total, losses.mel, losses.kl, losses.disc, Date().timeIntervalSince(start)))
            #expect(losses.total.isFinite && losses.disc.isFinite)
            return true
        }
        let seconds = Date().timeIntervalSince(start)
        print(String(
            format: "VITS fine-tune  %d steps in %.1f s, peak %.2f GB", steps, seconds,
            Double(Memory.peakMemory) / 1_073_741_824))
        #expect(steps == 15)

        let output = URL(fileURLWithPath: fineTuneOutput!)
        try model.saveTrained(from: source, to: output)
        // It loads back as a voice, and speaks.
        let voice = try VitsModel.fromModelDirectory(output)
        let (waveform, lengths) = voice(
            MLXArray(try voice.tokenizer!.encode("té pang 'chạu").map(Int32.init)).reshaped(1, -1))
        #expect(lengths[0].item(Int.self) > 0)
        #expect(abs(waveform).max().item(Float.self) > 0.01)
    }
}

// MARK: - PyTorch checkpoints and training checkpoints

/// A checkpoint `torch.save` wrote (torch 2.14): `{"model": OrderedDict,
/// "iteration", "learning_rate", "optimizer"}`, its archive folder named for
/// the file, one tensor a view into a larger storage.
@Suite("VITS PyTorch checkpoints")
struct VitsTorchCheckpointTests {
    let url = Bundle.module.url(forResource: "vits_tiny_checkpoint", withExtension: "pth", subdirectory: "media")!

    @Test func readsAStateDict() throws {
        let tensors = try TorchCheckpoint.tensors(at: url, under: "model")
        #expect(Set(tensors.keys) == ["convs.0.weight_v", "convs.0.weight_g", "convs.0.bias", "half", "steps", "scalar"])
        let v = tensors["convs.0.weight_v"]!
        #expect(v.shape == [2, 3, 2])
        #expect(v.asArray(Float.self) == (0 ..< 12).map { Float($0) / 4 })
        #expect(tensors["convs.0.weight_g"]!.shape == [2, 1, 1])
        // Elements 5 to 7 of a storage of 20.
        #expect(tensors["convs.0.bias"]!.asArray(Float.self) == [5, 6, 7])
        #expect(tensors["half"]!.dtype == .float16)
        #expect(tensors["half"]!.asType(.float32).asArray(Float.self) == [1.5, -2.25])
        #expect(tensors["steps"]!.asArray(Int64.self) == [7, -1])
        #expect(tensors["scalar"]!.shape == [])
        #expect(tensors["scalar"]!.item(Float.self) == 0.5)
    }

    @Test func readsWhatIsBesideIt() throws {
        let checkpoint = try TorchCheckpoint(url: url)
        guard case .int(7693)? = checkpoint.root["iteration"] else {
            Issue.record("iteration")
            return
        }
        guard case .float(let rate)? = checkpoint.root["learning_rate"] else {
            Issue.record("learning_rate")
            return
        }
        #expect(rate == 0.0002)
    }

    @Test func refusesWhatIsNotOne() throws {
        let file = FileManager.default.temporaryDirectory.appendingPathComponent("not-a-checkpoint.pth")
        try Data("not a zip at all, not even close to one".utf8).write(to: file)
        #expect(throws: TorchCheckpoint.ReadError.self) { try TorchCheckpoint.tensors(at: file) }
    }
}

private let metaDiscriminator = ProcessInfo.processInfo.environment["VITS_META_DISCRIMINATOR"]

/// A training checkpoint made here from the voice (VITS_PARITY_MODEL) and
/// Meta's discriminator for it (VITS_META_DISCRIMINATOR, `D_100000.pth`),
/// against the one finetune-hf-vits's converter made
/// (VITS_PARITY_TRAIN_MODEL); and a run saved part way, read back.
@Suite(
    "VITS training checkpoints",
    .serialized,
    .enabled(
        if: metaDiscriminator != nil && parityModel != nil && trainModel != nil,
        "set VITS_META_DISCRIMINATOR, VITS_PARITY_MODEL and VITS_PARITY_TRAIN_MODEL")
)
struct VitsTrainingCheckpointTests {
    @Test func madeAsTheConverterMakesIt() throws {
        let output = FileManager.default.temporaryDirectory.appendingPathComponent("vits-train-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: output) }
        try VitsModel.makeTrainingCheckpoint(
            voice: URL(fileURLWithPath: parityModel!), discriminator: URL(fileURLWithPath: metaDiscriminator!),
            to: output)

        let made = try loadArrays(url: output.appendingPathComponent("model.safetensors"))
        // The converter's transformers writes weight norm by PyTorch's newer
        // names; the voice, and so this, by the older ones.
        let converted = Dictionary(
            uniqueKeysWithValues: try loadArrays(
                url: URL(fileURLWithPath: trainModel!).appendingPathComponent("model.safetensors")
            ).map { key, value in
                (
                    key.replacingOccurrences(of: ".parametrizations.weight.original0", with: ".weight_g")
                        .replacingOccurrences(of: ".parametrizations.weight.original1", with: ".weight_v"),
                    value
                )
            })
        #expect(Set(made.keys) == Set(converted.keys))
        var worst: Float = 0
        for (key, value) in converted {
            guard let mine = made[key], mine.shape == value.shape else {
                Issue.record("\(key): \(made[key]?.shape ?? []) vs \(value.shape)")
                continue
            }
            worst = max(worst, abs(mine.asType(.float32) - value.asType(.float32)).max().item(Float.self))
        }
        print("VITS training checkpoint  largest difference from the converter's: \(worst)")
        #expect(worst <= 1e-6)

        // It loads to train.
        let (_, discriminator) = try VitsModel.loadForTraining(output)
        #expect(discriminator.discriminators.count == 6)
    }

    @Test func aRunSavedPartWayReadsBack() throws {
        let source = URL(fileURLWithPath: trainModel!)
        let output = FileManager.default.temporaryDirectory.appendingPathComponent("vits-run-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: output) }
        let (model, discriminator) = try VitsModel.loadForTraining(source)
        var config = VitsTrainingConfig()
        config.batchSize = 2
        let trainer = VitsTrainer(model: model, discriminator: discriminator, config: config)
        MLXRandom.seed(1)
        // Two clips the voice makes itself, and a step on them, so the
        // weights are no longer the source's.
        model.train(false)
        var clips: [VitsTrainingClip] = []
        for text in ["té pang 'chạu ê sa", "'tan chảu chắng bók sau 'va 'má 'toi"] {
            let ids = MLXArray(try model.tokenizer!.encode(text).map(Int32.init)).reshaped(1, -1)
            let (waveform, lengths) = model(ids)
            let audio = waveform[0, 0 ..< lengths[0].item(Int.self)]
            clips.append(try #require(try trainer.clip(name: text, audio: audio, text: text)))
        }
        model.train(true)
        trainer.train(clips, maxSteps: 1)
        try trainer.saveCheckpoint(from: source, to: output, step: 1)
        // Saved again, replacing the first.
        try trainer.saveCheckpoint(from: source, to: output, step: 1)
        #expect(vitsCheckpointStep(output) == 1)
        #expect(vitsCheckpointStep(source) == nil)

        let (again, againDiscriminator) = try VitsModel.loadForTraining(output)
        func same(_ a: Module, _ b: Module) -> Bool {
            let theirs = Dictionary(uniqueKeysWithValues: b.parameters().flattened())
            return a.parameters().flattened().allSatisfy { key, value in
                theirs[key].map { $0.shape == value.shape && allClose($0, value, rtol: 0, atol: 0).item(Bool.self) } ?? false
            }
        }
        #expect(same(model, again))
        #expect(same(discriminator, againDiscriminator))
        let (fresh, _) = try VitsModel.loadForTraining(source)
        #expect(!same(model, fresh))
    }
}

@Suite("VITS config JSON")
struct VitsJSONTests {
    @Test func floatsStayFloats() throws {
        let object = try JSONSerialization.jsonObject(
            with: Data(#"{"b": 5.0, "a": [1, 2.5, true], "c": "x/é", "d": null, "e": 1e-05}"#.utf8))
        let text = vitsJSON(object)
        #expect(text.contains("\"b\": 5.0"))
        #expect(text.contains("1,") && text.contains("2.5") && text.contains("true"))
        #expect(text.contains("\"c\": \"x/é\""))
        #expect(text.contains("\"d\": null"))
        #expect(text.contains("1e-05"))
        #expect(text.firstRange(of: "\"a\"")!.lowerBound < text.firstRange(of: "\"b\"")!.lowerBound)
    }
}
