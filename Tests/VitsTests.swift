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
