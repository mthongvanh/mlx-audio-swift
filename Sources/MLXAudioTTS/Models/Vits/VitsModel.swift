import Foundation
import HuggingFace
@preconcurrency import MLX
import MLXAudioCore
@preconcurrency import MLXLMCommon
import MLXNN

public enum VitsError: Error, LocalizedError {
    case unsupported(String)
    case nothingToRead
    case missingTokenizer

    public var errorDescription: String? {
        switch self {
        case .unsupported(let what): "VITS: unsupported \(what)"
        case .nothingToRead: "VITS: nothing in the text this voice can read."
        case .missingTokenizer: "VITS: no tokenizer; load the model from its folder."
        }
    }
}

/// VITS: Meta's MMS-TTS voices (about 1,100 languages, `facebook/mms-tts-*`)
/// and other VITS checkpoints in transformers' format, loaded as they are.
///
/// Ported from mlx-audio's Python `vits` model, which matches transformers'
/// `VitsModel` within 6e-5 on the waveform with noise off.
public final class VitsModel: Module, SpeechGenerationModel, @unchecked Sendable {
    public let config: VitsConfig
    public var tokenizer: VitsTokenizer?
    /// Speech's speed: above 1 is faster.
    public var speakingRate: Float

    @ModuleInfo(key: "text_encoder") var textEncoder: VitsTextEncoder
    @ModuleInfo var flow: VitsResidualCouplingBlock
    @ModuleInfo var decoder: VitsHifiGan
    /// A `VitsStochasticDurationPredictor` or, if the config says so, a
    /// `VitsDurationPredictor`.
    @ModuleInfo(key: "duration_predictor") var durationPredictor: Module
    @ModuleInfo(key: "embed_speaker") var embedSpeaker: Embedding?
    @ModuleInfo(key: "posterior_encoder") var posteriorEncoder: VitsPosteriorEncoder

    public var sampleRate: Int { config.samplingRate }
    public var defaultGenerationParameters: GenerateParameters { GenerateParameters() }

    public init(_ config: VitsConfig, tokenizer: VitsTokenizer? = nil) throws {
        self.config = config
        self.tokenizer = tokenizer
        speakingRate = config.speakingRate
        _textEncoder.wrappedValue = try VitsTextEncoder(config)
        _flow.wrappedValue = VitsResidualCouplingBlock(config)
        _decoder.wrappedValue = VitsHifiGan(config)
        _durationPredictor.wrappedValue = config.useStochasticDurationPrediction
            ? VitsStochasticDurationPredictor(config) : VitsDurationPredictor(config)
        if config.numSpeakers > 1 {
            _embedSpeaker.wrappedValue = Embedding(
                embeddingCount: config.numSpeakers, dimensions: config.speakerEmbeddingSize)
        } else {
            _embedSpeaker.wrappedValue = nil
        }
        _posteriorEncoder.wrappedValue = VitsPosteriorEncoder(config)
    }

    func speakerEmbeddings(_ speakerId: Int?) -> MLXArray? {
        guard config.numSpeakers > 1, let speakerId, let embedSpeaker else { return nil }
        return embedSpeaker(MLXArray([Int32(speakerId)]))[0..., .newAxis, 0...]
    }

    /// The duration predictor's log-durations, (batch, time, 1), for the
    /// text encoder's output.
    func logDurations(
        _ hidden: MLXArray, mask: MLXArray, speaker: MLXArray?, noiseScale: Float, noise: MLXArray?
    ) -> MLXArray {
        if let stochastic = durationPredictor as? VitsStochasticDurationPredictor {
            return stochastic.sample(hidden, mask: mask, conditioning: speaker, noiseScale: noiseScale, noise: noise)
        }
        return (durationPredictor as! VitsDurationPredictor)(hidden, mask: mask, conditioning: speaker)
    }

    /// Waveforms, (batch, samples), for token ids, (batch, length), and
    /// each one's length in samples. The noise arguments, if given, replace
    /// the random draws: tests pass the same to both ports.
    public func callAsFunction(
        _ inputIds: MLXArray,
        attentionMask: MLXArray? = nil,
        speakerId: Int? = nil,
        noiseScale: Float? = nil,
        noiseScaleDuration: Float? = nil,
        speakingRate: Float? = nil,
        durationNoise: MLXArray? = nil,
        priorNoise: MLXArray? = nil
    ) -> (waveform: MLXArray, lengths: MLXArray) {
        let noiseScale = noiseScale ?? config.noiseScale
        let noiseScaleDuration = noiseScaleDuration ?? config.noiseScaleDuration
        let speakingRate = speakingRate ?? self.speakingRate

        let attentionMask = attentionMask ?? MLXArray.ones(inputIds.shape)
        let inputMask = attentionMask.asType(.float32)[0..., 0..., .newAxis]
        let speaker = speakerEmbeddings(speakerId)

        let (hidden, priorMeans, priorLogVariances) = textEncoder(
            inputIds, mask: inputMask, attentionMask: attentionMask.asType(.float32))
        let logDuration = logDurations(
            hidden, mask: inputMask, speaker: speaker, noiseScale: noiseScaleDuration, noise: durationNoise)

        let duration = ceil(exp(logDuration) * inputMask / speakingRate)
        let predictedLengths = maximum(duration.sum(axes: [1, 2]), MLXArray(Float(1))).asType(.int32)
        let outputLength = predictedLengths.max().item(Int.self)
        let frames = MLXArray(0 ..< outputLength).asType(.float32)
        let outputMask = (frames[.newAxis, 0...] .< predictedLengths.asType(.float32)[0..., .newAxis])
            .asType(.float32)[0..., 0..., .newAxis]

        // Each output frame to the input token it expands.
        let cumDuration = cumsum(duration[0..., 0..., 0], axis: 1)
        let valid = (frames[.newAxis, .newAxis, 0...] .< cumDuration[0..., 0..., .newAxis]).asType(.float32)
        let path = valid - padded(valid, widths: [0, IntOrPair((1, 0)), 0])[0..., ..<valid.dim(1)]
        let attention = path.transposed(0, 2, 1) * (outputMask * inputMask.transposed(0, 2, 1))

        let means = matmul(attention, priorMeans)
        let logVariances = matmul(attention, priorLogVariances)
        let noise = priorNoise ?? MLXRandom.normal(means.shape)
        let priorLatents = means + noise * exp(logVariances) * noiseScale

        let latents = flow(priorLatents, mask: outputMask, conditioning: speaker, reverse: true)
        let waveform = decoder(latents * outputMask, conditioning: speaker)[0..., 0..., 0]
        return (waveform, predictedLengths * config.hopLength)
    }

    // MARK: - SpeechGenerationModel

    /// Speech for `text`. `voice`, if it is a number, picks the speaker of a
    /// voice that has several.
    public func generate(
        text: String,
        voice: String?,
        refAudio: sending MLXArray?,
        refText: String?,
        language: String?,
        generationParameters: GenerateParameters
    ) async throws -> sending MLXArray {
        _ = refAudio; _ = refText; _ = language; _ = generationParameters
        guard let tokenizer else { throw VitsError.missingTokenizer }
        let ids = try tokenizer.encode(text)
        guard ids.count > 1 else { throw VitsError.nothingToRead }
        try Task.checkCancellation()
        let (waveform, lengths) = self(
            MLXArray(ids.map(Int32.init)).reshaped(1, -1), speakerId: voice.flatMap { Int($0) })
        let audio = waveform[0, ..<lengths[0].item(Int.self)]
        eval(audio)
        return audio
    }

    public func generateStream(
        text: String,
        voice: String?,
        refAudio: sending MLXArray?,
        refText: String?,
        language: String?,
        generationParameters: GenerateParameters
    ) -> sending AsyncThrowingStream<AudioGeneration, Error> {
        _ = refAudio
        let (stream, continuation) = AsyncThrowingStream<AudioGeneration, Error>.makeStream()
        let task = Task { @Sendable [weak self] in
            guard let self else {
                continuation.finish(throwing: AudioGenerationError.modelNotInitialized("Model deallocated"))
                return
            }
            do {
                let audio = try await self.generate(
                    text: text, voice: voice, refAudio: nil, refText: refText, language: language,
                    generationParameters: generationParameters)
                continuation.yield(.audio(audio))
                continuation.finish()
            } catch {
                continuation.finish(throwing: error)
            }
        }
        continuation.onTermination = { @Sendable _ in task.cancel() }
        return stream
    }

    // MARK: - Weights

    /// transformers' weights, in PyTorch's layout, to this model's. Weights
    /// already in this model's layout, as mlx-audio saves them, pass as
    /// they are.
    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        let expected = Dictionary(
            parameters().flattened().map { ($0.0, $0.1.shape) }, uniquingKeysWith: { a, _ in a })
        var sanitized: [String: MLXArray] = [:]
        for (key, value) in weights {
            if key.hasPrefix("discriminator.") { continue }  // training checkpoints carry one
            // PyTorch's newer name for weight norm's two halves.
            let key = key
                .replacingOccurrences(of: ".parametrizations.weight.original0", with: ".weight_g")
                .replacingOccurrences(of: ".parametrizations.weight.original1", with: ".weight_v")
            var value = value
            if key.hasSuffix(".translate") || key.hasSuffix(".log_scale") {
                value = value.reshaped(-1)
            } else if value.ndim == 3,
                      !key.hasSuffix("emb_rel_k"), !key.hasSuffix("emb_rel_v"), !key.hasSuffix("weight_g")
            {
                // (in, out, k) for a transposed convolution, else (out, in, k),
                // to (out, k, in).
                let order = key.hasPrefix("decoder.upsampler.") ? [1, 2, 0] : [0, 2, 1]
                let transposedShape = order.map { value.dim($0) }
                // Already this model's layout: the shape fits as it is and
                // wouldn't once transposed. When both fit (in == k),
                // PyTorch's layout is assumed.
                let mlxLayout = expected[key] == value.shape && transposedShape != value.shape
                if !mlxLayout {
                    value = value.transposed(axes: order)
                }
            } else if key.hasSuffix("weight_g"), key.hasPrefix("decoder.upsampler.") {
                value = value.reshaped(1, 1, -1)  // normed over the input channel
            }
            sanitized[key] = value
        }
        return sanitized
    }

    // MARK: - Loading

    public static func fromPretrained(_ modelRepo: String, cache: HubCache = .default) async throws -> VitsModel {
        let hfToken: String? = ProcessInfo.processInfo.environment["HF_TOKEN"]
            ?? Bundle.main.object(forInfoDictionaryKey: "HF_TOKEN") as? String
        guard let repoID = Repo.ID(rawValue: modelRepo) else {
            throw AudioGenerationError.invalidInput("Invalid repository ID: \(modelRepo)")
        }
        let modelDir = try await ModelUtils.resolveOrDownloadModel(
            repoID: repoID, requiredExtension: ".safetensors", hfToken: hfToken, cache: cache)
        return try fromModelDirectory(modelDir)
    }

    public static func fromModelDirectory(_ modelDir: URL) throws -> VitsModel {
        let configData = try Data(contentsOf: modelDir.appendingPathComponent("config.json"))
        let config = try JSONDecoder().decode(VitsConfig.self, from: configData)
        let model = try VitsModel(config, tokenizer: try VitsTokenizer.fromModelDirectory(modelDir))

        var weights: [String: MLXArray] = [:]
        let files = try FileManager.default.contentsOfDirectory(at: modelDir, includingPropertiesForKeys: nil)
            .filter { $0.pathExtension == "safetensors" }
        for file in files {
            for (key, value) in try loadArrays(url: file) {
                weights[key] = value
            }
        }
        let sanitized = model.sanitize(weights: weights)
        try model.update(parameters: ModuleParameters.unflattened(sanitized), verify: .all)
        model.train(false)
        eval(model.parameters())
        return model
    }
}
