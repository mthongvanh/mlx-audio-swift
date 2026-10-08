import Foundation
@preconcurrency import MLX
import MLXAudioCore
import MLXFFT
import MLXNN
import MLXOptimizers

// Fine-tuning VITS: a port of mlx-audio's Python `vits/train.py`, itself a
// port of finetune-hf-vits (github.com/ylacombe/finetune-hf-vits, MIT), step
// for step: the same losses and weights, the same order of discriminator
// and generator updates, AdamW with PyTorch's defaults, and the learning
// rate decayed once an epoch. It starts from a checkpoint that carries the
// discriminator (`discriminator.*` weights), as that repo's
// `convert_original_discriminator_checkpoint.py` makes.

public struct VitsTrainingConfig: Sendable {
    public var learningRate: Float = 2e-5
    public var adamBeta1: Float = 0.8
    public var adamBeta2: Float = 0.99
    public var adamEpsilon: Float = 1e-8
    public var weightDecay: Float = 0.01  // torch.optim.AdamW's default
    public var lrDecay: Float = 0.999875
    public var maxGradNorm: Float = 1.0
    public var weightDisc: Float = 3.0
    public var weightFmaps: Float = 1.0
    public var weightGen: Float = 1.0
    public var weightKL: Float = 1.5
    public var weightDuration: Float = 1.0
    public var weightMel: Float = 35.0
    public var batchSize: Int = 16
    public var epochs: Int = 200
    public var segmentSize: Int = 8192  // samples the decoder makes per clip per step
    public var nFft: Int = 1024
    public var hopLength: Int = 256
    public var nMels: Int = 80
    public var minDuration: Double = 1.0
    public var maxDuration: Double = 20.0
    public var maxTokensLength: Int = 500
    public var seed: UInt64 = 456

    public init() {}
}

// MARK: - Spectrograms

/// The linear and log-mel spectrograms finetune-hf-vits trains on:
/// reflect-padded, Hann window, no centring, magnitudes with 1e-6 inside the
/// root, Slaney mel filters. Differentiable, for the mel loss.
public struct VitsSpectrogram {
    let nFft: Int
    let hopLength: Int
    let window: MLXArray
    let melFilters: MLXArray  // (bins, mels)

    public init(sampleRate: Int, nFft: Int = 1024, hopLength: Int = 256, nMels: Int = 80) {
        self.nFft = nFft
        self.hopLength = hopLength
        let n = MLXArray(0 ..< nFft).asType(.float32)
        window = 0.5 - 0.5 * cos(2 * Float.pi * n / Float(nFft))  // periodic
        melFilters = Self.slaneyMelFilters(sampleRate: sampleRate, nFft: nFft, nMels: nMels)
    }

    /// Slaney-scale, Slaney-normalised triangular filters, (bins, mels),
    /// built in Double and rounded once, as mlx-audio's `mel_filters(...,
    /// precise=True)` does: built in Float they drift by about 5e-6, enough
    /// to tip where the mel loss's `abs` turns and so its gradient.
    static func slaneyMelFilters(sampleRate: Int, nFft: Int, nMels: Int) -> MLXArray {
        let fSp = 200.0 / 3
        let minLogHz = 1000.0
        let minLogMel = minLogHz / fSp
        let logStep = Foundation.log(6.4) / 27.0
        func hzToMel(_ f: Double) -> Double { f >= minLogHz ? minLogMel + Foundation.log(f / minLogHz) / logStep : f / fSp }
        func melToHz(_ m: Double) -> Double { m >= minLogMel ? minLogHz * Foundation.exp(logStep * (m - minLogMel)) : fSp * m }

        let nFreqs = nFft / 2 + 1
        let top = Double(sampleRate / 2)
        let allFreqs = (0 ..< nFreqs).map { top * Double($0) / Double(nFreqs - 1) }
        let (mMin, mMax) = (hzToMel(0), hzToMel(Double(sampleRate) / 2))
        let fPts = (0 ..< nMels + 2).map { melToHz(mMin + (mMax - mMin) * Double($0) / Double(nMels + 1)) }
        var filters = [Float](repeating: 0, count: nFreqs * nMels)
        for i in 0 ..< nFreqs {
            for j in 0 ..< nMels {
                let down = -(fPts[j] - allFreqs[i]) / (fPts[j + 1] - fPts[j])
                let up = (fPts[j + 2] - allFreqs[i]) / (fPts[j + 2] - fPts[j + 1])
                let enorm = 2.0 / (fPts[j + 2] - fPts[j])
                filters[i * nMels + j] = Float(max(0, min(down, up)) * enorm)
            }
        }
        return MLXArray(filters).reshaped(nFreqs, nMels)
    }

    /// (batch, samples) to magnitudes (batch, frames, bins) and log-mel
    /// (batch, frames, mels).
    public func callAsFunction(_ waveform: MLXArray) -> (magnitudes: MLXArray, mel: MLXArray) {
        let pad = (nFft - hopLength) / 2
        let n = waveform.dim(1)
        let left = waveform[0..., 1 ..< (pad + 1)][0..., .stride(by: -1)]
        let right = waveform[0..., (n - pad - 1) ..< (n - 1)][0..., .stride(by: -1)]
        let x = concatenated([left, waveform, right], axis: 1)
        let numFrames = (x.dim(1) - nFft) / hopLength + 1
        let index = MLXArray(0 ..< numFrames)[0..., .newAxis] * hopLength + MLXArray(0 ..< nFft)[.newAxis, 0...]
        let frames = x[0..., index] * window
        let spectrum = MLXFFT.rfft(frames, axis: -1)
        let magnitudes = sqrt(spectrum.realPart().square() + spectrum.imaginaryPart().square() + 1e-6)
        let mel = matmul(magnitudes, melFilters)
        return (magnitudes, log(maximum(mel, MLXArray(Float(1e-5)))))
    }
}

// MARK: - The discriminator

/// A 2-D convolution, plain or weight-normed, channels-last.
final class VitsConv2d: Module {
    let stride: (Int, Int)
    let padding: (Int, Int)
    var weight: MLXArray?
    @ParameterInfo(key: "weight_g") var weightG: MLXArray?
    @ParameterInfo(key: "weight_v") var weightV: MLXArray?
    var bias: MLXArray

    init(
        _ inChannels: Int, _ outChannels: Int, kernel: (Int, Int), stride: (Int, Int), padding: (Int, Int),
        weightNorm: Bool
    ) {
        self.stride = stride
        self.padding = padding
        let shape = [outChannels, kernel.0, kernel.1, inChannels]
        if weightNorm {
            _weightV.wrappedValue = MLXArray.zeros(shape)
            _weightG.wrappedValue = MLXArray.ones([outChannels, 1, 1, 1])
        } else {
            weight = MLXArray.zeros(shape)
            _weightG.wrappedValue = nil
            _weightV.wrappedValue = nil
        }
        bias = MLXArray.zeros([outChannels])
    }

    private static func norm(_ w: MLXArray) -> MLXArray {
        sqrt(sum(w * w, axes: [1, 2, 3], keepDims: true))
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let w = weightV.map { weightG! * $0 / Self.norm($0) } ?? weight!
        return conv2d(x, w, stride: IntOrPair(stride), padding: IntOrPair(padding)) + bias
    }
}

/// One of the discriminator's parts: outputs, flattened, and the feature
/// maps along the way.
class VitsSubDiscriminator: Module {
    func callAsFunction(_ x: MLXArray) -> (MLXArray, [MLXArray]) {
        fatalError("a VitsSubDiscriminator subclass must override callAsFunction")
    }
}

final class VitsScaleDiscriminator: VitsSubDiscriminator {
    let leakyReluSlope: Float
    @ModuleInfo var convs: [VitsConv1d]
    @ModuleInfo(key: "final_conv") var finalConv: VitsConv1d

    init(channels: [Int], leakyReluSlope: Float) {
        self.leakyReluSlope = leakyReluSlope
        var convs = [VitsConv1d(channels[0], channels[1], kernelSize: 15, padding: 7, weightNorm: true)]
        var groups = 4
        for (cIn, cOut) in zip(channels[1 ..< (channels.count - 1)], channels[2...]) {
            convs.append(VitsConv1d(
                cIn, cOut, kernelSize: 41, stride: 4, padding: 20, groups: groups, weightNorm: true))
            groups *= 4
        }
        let last = channels[channels.count - 1]
        convs.append(VitsConv1d(last, last, kernelSize: 41, stride: 4, padding: 20, groups: groups, weightNorm: true))
        convs.append(VitsConv1d(last, last, kernelSize: 5, padding: 2, weightNorm: true))
        _convs.wrappedValue = convs
        _finalConv.wrappedValue = VitsConv1d(last, 1, kernelSize: 3, padding: 1, weightNorm: true)
    }

    /// x: (batch, samples, 1).
    override func callAsFunction(_ x: MLXArray) -> (MLXArray, [MLXArray]) {
        var x = x
        var fmap: [MLXArray] = []
        for conv in convs {
            x = leakyRelu(conv(x), negativeSlope: leakyReluSlope)
            fmap.append(x)
        }
        x = finalConv(x)
        fmap.append(x)
        return (x.reshaped(x.dim(0), -1), fmap)
    }
}

final class VitsPeriodDiscriminator: VitsSubDiscriminator {
    let period: Int
    let leakyReluSlope: Float
    @ModuleInfo var convs: [VitsConv2d]
    @ModuleInfo(key: "final_conv") var finalConv: VitsConv2d

    init(channels: [Int], period: Int, kernelSize: Int, stride: Int, leakyReluSlope: Float) {
        self.period = period
        self.leakyReluSlope = leakyReluSlope
        let pad = (kernelSize - 1) / 2
        var convs = zip(channels.dropLast(), channels.dropFirst()).map { cIn, cOut in
            VitsConv2d(cIn, cOut, kernel: (kernelSize, 1), stride: (stride, 1), padding: (pad, 0), weightNorm: true)
        }
        let last = channels[channels.count - 1]
        convs.append(VitsConv2d(
            last, last, kernel: (kernelSize, 1), stride: (1, 1), padding: (pad, 0), weightNorm: true))
        _convs.wrappedValue = convs
        _finalConv.wrappedValue = VitsConv2d(
            last, 1, kernel: (3, 1), stride: (1, 1), padding: (1, 0), weightNorm: true)
    }

    /// x: (batch, samples, 1).
    override func callAsFunction(_ x: MLXArray) -> (MLXArray, [MLXArray]) {
        var x = x
        var fmap: [MLXArray] = []
        let (batch, channels) = (x.dim(0), x.dim(2))
        var length = x.dim(1)
        if length % period != 0 {
            let nPad = period - length % period
            x = concatenated([x, x[0..., (length - nPad - 1) ..< (length - 1)][0..., .stride(by: -1)]], axis: 1)
            length += nPad
        }
        x = x.reshaped(batch, length / period, period, channels)
        for conv in convs {
            x = leakyRelu(conv(x), negativeSlope: leakyReluSlope)
            fmap.append(x)
        }
        x = finalConv(x)
        fmap.append(x)
        return (x.reshaped(batch, -1), fmap)
    }
}

/// The discriminator's settings in a training checkpoint's `config.json`.
public struct VitsDiscriminatorConfig: Decodable, Sendable {
    public var scaleChannels: [Int]?
    public var periods: [Int] = [2, 3, 5, 7, 11]
    public var periodChannels: [Int] = [1, 32, 128, 512, 1024]
    public var kernelSize: Int = 5
    public var stride: Int = 3
    public var leakyReluSlope: Float = 0.1

    enum CodingKeys: String, CodingKey {
        case scaleChannels = "discriminator_scale_channels"
        case periods = "discriminator_periods"
        case periodChannels = "discriminator_period_channels"
        case kernelSize = "discriminator_kernel_size"
        case stride = "discriminator_stride"
        case leakyReluSlope = "leaky_relu_slope"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        scaleChannels = try c.decodeIfPresent([Int].self, forKey: .scaleChannels)
        periods = try c.decodeIfPresent([Int].self, forKey: .periods) ?? periods
        periodChannels = try c.decodeIfPresent([Int].self, forKey: .periodChannels) ?? periodChannels
        kernelSize = try c.decodeIfPresent(Int.self, forKey: .kernelSize) ?? kernelSize
        stride = try c.decodeIfPresent(Int.self, forKey: .stride) ?? stride
        leakyReluSlope = try c.decodeIfPresent(Float.self, forKey: .leakyReluSlope) ?? leakyReluSlope
    }
}

/// The discriminator, held weight-normed throughout, as finetune-hf-vits
/// trains it.
public final class VitsDiscriminator: Module {
    @ModuleInfo var discriminators: [VitsSubDiscriminator]

    public init(_ config: VitsDiscriminatorConfig) {
        var parts: [VitsSubDiscriminator] = []
        if let scale = config.scaleChannels {
            parts.append(VitsScaleDiscriminator(channels: scale, leakyReluSlope: config.leakyReluSlope))
        }
        for period in config.periods {
            parts.append(VitsPeriodDiscriminator(
                channels: config.periodChannels, period: period, kernelSize: config.kernelSize,
                stride: config.stride, leakyReluSlope: config.leakyReluSlope))
        }
        _discriminators.wrappedValue = parts
    }

    func callAsFunction(_ waveform: MLXArray) -> (outputs: [MLXArray], fmaps: [[MLXArray]]) {
        var outputs: [MLXArray] = []
        var fmaps: [[MLXArray]] = []
        for d in discriminators {
            let (out, fmap) = d(waveform)
            outputs.append(out)
            fmaps.append(fmap)
        }
        return (outputs, fmaps)
    }

    /// The `discriminator.*` weights of a training checkpoint, PyTorch's
    /// layout to MLX's, plain weights split for weight norm.
    func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        let expected = Dictionary(
            parameters().flattened().map { ($0.0, $0.1.shape) }, uniquingKeysWith: { a, _ in a })
        var out: [String: MLXArray] = [:]
        for (key, value) in weights where key.hasPrefix("discriminator.") {
            let key = String(key.dropFirst("discriminator.".count))
            switch value.ndim {
            case 3: out[key] = value.transposed(0, 2, 1)
            case 4: out[key] = value.transposed(0, 2, 3, 1)
            default: out[key] = value
            }
        }
        return vitsSplitWeightNorm(out, expected: expected)
    }
}

// MARK: - Monotonic alignment search

/// The most likely monotonic alignment of frames to tokens, as
/// finetune-hf-vits's Cython `maximum_path` finds it. `negCent` is (batch,
/// frames, tokens), row-major; the path comes back the same shape.
func vitsMaximumPath(
    _ negCent: [Float], shape: (batch: Int, frames: Int, tokens: Int), frameLengths: [Int], textLengths: [Int]
) -> [Float] {
    let maxNegVal: Float = -1e9
    let (_, frames, tokens) = shape
    var paths = [Float](repeating: 0, count: negCent.count)
    for b in 0 ..< shape.batch {
        let (tY, tX) = (frameLengths[b], textLengths[b])
        guard tY > 0, tX > 0 else { continue }
        let base = b * frames * tokens
        var value = [Float](repeating: 0, count: tY * tX)
        for y in 0 ..< tY {
            for x in 0 ..< tX {
                value[y * tX + x] = negCent[base + y * tokens + x]
            }
        }
        for y in 0 ..< tY {
            let lo = max(0, tX + y - tY)
            let hi = min(tX, y + 1)
            guard lo < hi else { continue }
            for x in lo ..< hi {
                let vCur: Float
                let vPrev: Float
                if y == 0 {
                    vCur = maxNegVal
                    vPrev = x == 0 ? 0 : maxNegVal
                } else {
                    vCur = x == y ? maxNegVal : value[(y - 1) * tX + x]
                    vPrev = x == 0 ? maxNegVal : value[(y - 1) * tX + x - 1]
                }
                value[y * tX + x] += max(vPrev, vCur)
            }
        }
        var index = tX - 1
        for y in stride(from: tY - 1, through: 0, by: -1) {
            paths[base + y * tokens + index] = 1
            if index != 0, y != 0,
               index == y || value[(y - 1) * tX + index] < value[(y - 1) * tX + index - 1]
            {
                index -= 1
            }
        }
    }
    return paths
}

// MARK: - The training forward pass and losses

/// A padded batch: tokens with 0, audio and spectrograms with 0, and masks
/// saying what is real.
public struct VitsTrainingBatch {
    public var inputIds: MLXArray  // (batch, tokens), int32
    public var attentionMask: MLXArray  // (batch, tokens)
    public var labels: MLXArray  // (batch, frames, bins)
    public var labelsMask: MLXArray  // (batch, frames)
    public var mel: MLXArray  // (batch, frames, mels)
    public var waveform: MLXArray  // (batch, samples)

    public init(
        inputIds: MLXArray, attentionMask: MLXArray, labels: MLXArray, labelsMask: MLXArray,
        mel: MLXArray, waveform: MLXArray
    ) {
        self.inputIds = inputIds
        self.attentionMask = attentionMask
        self.labels = labels
        self.labelsMask = labelsMask
        self.mel = mel
        self.waveform = waveform
    }
}

/// A step's random draws, made once and used by both passes through the
/// generator. Tests give both ports the same.
public struct VitsTrainingNoise {
    public var posterior: MLXArray  // (batch, frames, flow size)
    public var duration: MLXArray  // (batch, tokens, 2)
    public var sliceStarts: MLXArray  // (batch,), int32

    public init(posterior: MLXArray, duration: MLXArray, sliceStarts: MLXArray) {
        self.posterior = posterior
        self.duration = duration
        self.sliceStarts = sliceStarts
    }
}

struct VitsTrainingOutputs {
    let waveform: MLXArray  // (batch, segment samples, 1)
    let logDuration: MLXArray
    let attention: MLXArray  // (batch, frames, tokens)
    let sliceStarts: MLXArray
    let labelsMask: MLXArray  // (batch, frames, 1)
    let priorLatents: MLXArray
    let priorMeans: MLXArray
    let priorLogVariances: MLXArray
    let posteriorMeans: MLXArray
    let posteriorLogVariances: MLXArray
}

/// A window of `size` from each start: x is (batch, time, channels).
func vitsSliceSegments(_ x: MLXArray, starts: MLXArray, size: Int) -> MLXArray {
    let index = stopGradient(starts[0..., .newAxis] + MLXArray(Int32(0) ..< Int32(size))[.newAxis, 0...])
    return takeAlong(x, index[0..., 0..., .newAxis], axis: 1)
}

/// finetune-hf-vits's `VitsModelForPreTraining.forward` with labels.
func vitsTrainingForward(
    _ model: VitsModel, batch: VitsTrainingBatch, segmentFrames: Int, noise: VitsTrainingNoise
) -> VitsTrainingOutputs {
    let inputMask = batch.attentionMask.asType(.float32)[0..., 0..., .newAxis]
    let labelsMask = batch.labelsMask.asType(.float32)[0..., 0..., .newAxis]

    let (hidden, priorMeans, priorLogVariances) = model.textEncoder(
        batch.inputIds, mask: inputMask, attentionMask: batch.attentionMask.asType(.float32))
    let (latents, posteriorMeans, posteriorLogVariances) = model.posteriorEncoder(
        batch.labels, mask: labelsMask, noise: noise.posterior)
    let priorLatents = model.flow(latents, mask: labelsMask, reverse: false)

    // The alignment, found without gradients: (batch, frames, tokens).
    let pl = stopGradient(priorLatents)
    let pm = stopGradient(priorMeans)
    let plv = stopGradient(priorLogVariances)
    let priorVariances = exp(-2 * plv)  // (batch, tokens, channels)
    let negCent1 = sum(Float(-0.5 * log(2 * Double.pi)) - plv, axis: -1)[0..., .newAxis, 0...]
    let negCent2 = matmul(-0.5 * pl.square(), priorVariances.transposed(0, 2, 1))
    let negCent3 = matmul(pl, (pm * priorVariances).transposed(0, 2, 1))
    let negCent4 = sum(-0.5 * pm.square() * priorVariances, axis: -1)[0..., .newAxis, 0...]
    let negCent = (negCent1 + negCent2 + negCent3 + negCent4).asType(.float32)
    let shape = (negCent.dim(0), negCent.dim(1), negCent.dim(2))
    let path = vitsMaximumPath(
        negCent.asArray(Float.self), shape: shape,
        frameLengths: batch.labelsMask.sum(axis: 1).asArray(Float.self).map { Int($0) },
        textLengths: batch.attentionMask.sum(axis: 1).asArray(Float.self).map { Int($0) })
    let attention = MLXArray(path).reshaped(shape.0, shape.1, shape.2)
    let durations = attention.sum(axis: 1)[0..., 0..., .newAxis]  // (batch, tokens, 1)

    let logDuration: MLXArray
    if let stochastic = model.durationPredictor as? VitsStochasticDurationPredictor {
        logDuration = stochastic.negativeLogLikelihood(
            hidden, mask: inputMask, durations: durations, noise: noise.duration) / inputMask.sum()
    } else {
        let padded = log(durations + 1e-6) * inputMask
        let predicted = (model.durationPredictor as! VitsDurationPredictor)(hidden, mask: inputMask)
        logDuration = sum((predicted - padded).square(), axes: [1, 2]) / inputMask.sum()
    }

    let expandedMeans = matmul(attention, priorMeans)  // (batch, frames, channels)
    let expandedLogVariances = matmul(attention, priorLogVariances)
    let latentsSlice = vitsSliceSegments(latents, starts: noise.sliceStarts, size: segmentFrames)
    let waveform = model.decoder(latentsSlice)

    return VitsTrainingOutputs(
        waveform: waveform, logDuration: logDuration, attention: attention, sliceStarts: noise.sliceStarts,
        labelsMask: labelsMask, priorLatents: priorLatents, priorMeans: expandedMeans,
        priorLogVariances: expandedLogVariances, posteriorMeans: posteriorMeans,
        posteriorLogVariances: posteriorLogVariances)
}

func vitsDiscriminatorLoss(real: [MLXArray], generated: [MLXArray]) -> (MLXArray, MLXArray, MLXArray) {
    let realLoss = real.map { mean((1 - $0).square()) }.reduce(MLXArray(Float(0)), +)
    let generatedLoss = generated.map { mean($0.square()) }.reduce(MLXArray(Float(0)), +)
    return (realLoss + generatedLoss, realLoss, generatedLoss)
}

func vitsFeatureLoss(real: [[MLXArray]], generated: [[MLXArray]]) -> MLXArray {
    var loss = MLXArray(Float(0))
    for (mapsReal, mapsGenerated) in zip(real, generated) {
        for (r, g) in zip(mapsReal, mapsGenerated) {
            loss = loss + mean(abs(stopGradient(r) - g))
        }
    }
    return loss * 2
}

func vitsGeneratorLoss(_ outputs: [MLXArray]) -> MLXArray {
    outputs.map { mean((1 - $0).square()) }.reduce(MLXArray(Float(0)), +)
}

func vitsKLLoss(
    priorLatents: MLXArray, posteriorLogVariance: MLXArray, priorMeans: MLXArray,
    priorLogVariance: MLXArray, labelsMask: MLXArray
) -> MLXArray {
    var kl = priorLogVariance - posteriorLogVariance - 0.5
    kl = kl + 0.5 * (priorLatents - priorMeans).square() * exp(-2.0 * priorLogVariance)
    return sum(kl * labelsMask) / sum(labelsMask)
}

/// Gradients scaled down together so their global norm is at most
/// `maxNorm`, as mlx's `clip_grad_norm` does.
func vitsClipGradNorm(_ grads: ModuleParameters, maxNorm: Float) -> ModuleParameters {
    let flat = grads.flattened()
    let total = sqrt(flat.map { $0.1.square().sum() }.reduce(MLXArray(Float(0)), +))
    let normalizer = minimum(maxNorm / (total + 1e-6), MLXArray(Float(1)))
    return ModuleParameters.unflattened(flat.map { ($0.0, $0.1 * normalizer) })
}

// MARK: - Training

/// The losses of one step, the generator's and the discriminator's.
public struct VitsTrainingLosses: Sendable {
    public var duration, mel, kl, fmaps, gen: Float
    public var disc, realDisc, fakeDisc: Float
    public var total: Float { duration + mel + kl + fmaps + gen }
}

public final class VitsTrainer {
    public let model: VitsModel
    public let discriminator: VitsDiscriminator
    public let config: VitsTrainingConfig
    public let spectrogram: VitsSpectrogram
    let segmentFrames: Int
    let generatorOptimizer: AdamW
    let discriminatorOptimizer: AdamW

    public init(model: VitsModel, discriminator: VitsDiscriminator, config: VitsTrainingConfig) {
        self.model = model
        self.discriminator = discriminator
        self.config = config
        segmentFrames = config.segmentSize / config.hopLength
        spectrogram = VitsSpectrogram(
            sampleRate: model.sampleRate, nFft: config.nFft, hopLength: config.hopLength, nMels: config.nMels)
        func adamW() -> AdamW {
            AdamW(
                learningRate: config.learningRate, betas: (config.adamBeta1, config.adamBeta2),
                eps: config.adamEpsilon, weightDecay: config.weightDecay, biasCorrection: true)
        }
        generatorOptimizer = adamW()
        discriminatorOptimizer = adamW()
    }

    /// ExponentialLR, stepped at the start of each epoch as finetune-hf-vits
    /// does, so epoch 0 already runs at lr × decay.
    public func setEpoch(_ epoch: Int) {
        let lr = config.learningRate * pow(config.lrDecay, Float(epoch + 1))
        generatorOptimizer.learningRate = lr
        discriminatorOptimizer.learningRate = lr
    }

    func targets(_ batch: VitsTrainingBatch, _ outputs: VitsTrainingOutputs) -> (mel: MLXArray, wave: MLXArray) {
        let mel = vitsSliceSegments(batch.mel, starts: outputs.sliceStarts, size: segmentFrames)
        let wave = vitsSliceSegments(
            batch.waveform[0..., 0..., .newAxis], starts: outputs.sliceStarts * Int32(config.hopLength),
            size: config.segmentSize)
        return (mel, wave)
    }

    /// The discriminator's weighted loss, then its loss, real and fake parts.
    func discriminatorLoss(_ discriminator: VitsDiscriminator, fake: MLXArray, real: MLXArray) -> [MLXArray] {
        let (realOut, _) = discriminator(real)
        let (fakeOut, _) = discriminator(fake)
        let (loss, realLoss, fakeLoss) = vitsDiscriminatorLoss(real: realOut, generated: fakeOut)
        return [loss * config.weightDisc, loss, realLoss, fakeLoss]
    }

    /// The generator's weighted loss, then duration, mel, KL, feature maps
    /// and adversarial parts.
    func generatorLoss(_ model: VitsModel, batch: VitsTrainingBatch, noise: VitsTrainingNoise) -> [MLXArray] {
        let c = config
        let outputs = vitsTrainingForward(model, batch: batch, segmentFrames: segmentFrames, noise: noise)
        let (melTarget, waveTarget) = targets(batch, outputs)
        let melGenerated = spectrogram(outputs.waveform[0..., 0..., 0]).mel
        let (_, fmapsTarget) = discriminator(waveTarget)
        let (discGenerated, fmapsGenerated) = discriminator(outputs.waveform)

        let duration = outputs.logDuration.sum()
        let mel = mean(abs(melTarget - melGenerated))
        let kl = vitsKLLoss(
            priorLatents: outputs.priorLatents, posteriorLogVariance: outputs.posteriorLogVariances,
            priorMeans: outputs.priorMeans, priorLogVariance: outputs.priorLogVariances,
            labelsMask: outputs.labelsMask)
        let fmaps = vitsFeatureLoss(real: fmapsTarget, generated: fmapsGenerated)
        let gen = vitsGeneratorLoss(discGenerated)
        let total = duration * c.weightDuration + mel * c.weightMel + kl * c.weightKL
            + fmaps * c.weightFmaps + gen * c.weightGen
        return [total, duration, mel, kl, fmaps, gen]
    }

    func generatorGradients(_ batch: VitsTrainingBatch, _ noise: VitsTrainingNoise) -> ([MLXArray], ModuleParameters) {
        let vg = valueAndGrad(model: model) { (model: VitsModel, _: Int) in
            self.generatorLoss(model, batch: batch, noise: noise)
        }
        return vg(model, 0)
    }

    func discriminatorGradients(fake: MLXArray, real: MLXArray) -> ([MLXArray], ModuleParameters) {
        let vg = valueAndGrad(model: discriminator) { (discriminator: VitsDiscriminator, _: Int) in
            self.discriminatorLoss(discriminator, fake: fake, real: real)
        }
        return vg(discriminator, 0)
    }

    /// A step's random draws.
    public func drawNoise(_ batch: VitsTrainingBatch) -> VitsTrainingNoise {
        let (b, frames) = (batch.labels.dim(0), batch.labels.dim(1))
        let lengths = batch.labelsMask.sum(axis: 1)
        return VitsTrainingNoise(
            posterior: MLXRandom.normal([b, frames, model.config.flowSize]),
            duration: MLXRandom.normal([b, batch.inputIds.dim(1), 2]),
            sliceStarts: (MLXRandom.uniform(0 ..< 1, [b]) * (lengths - Float(segmentFrames) + 1)).asType(.int32))
    }

    /// One update of each network. `noise`, if given, replaces the step's
    /// random draws: tests pass the same to both ports.
    @discardableResult
    public func step(_ batch: VitsTrainingBatch, noise: VitsTrainingNoise? = nil) -> VitsTrainingLosses {
        let noise = noise ?? drawNoise(batch)
        // Dropout and layer drop draw too; the same seed for both passes
        // makes the second see what the first did.
        let seed = UInt64.random(in: 0 ..< (1 << 31))

        // 1. The discriminator, on this step's generated audio.
        MLXRandom.seed(seed)
        let outputs = vitsTrainingForward(model, batch: batch, segmentFrames: segmentFrames, noise: noise)
        let fake = stopGradient(outputs.waveform)
        let (_, waveTarget) = targets(batch, outputs)
        let (discValues, discGrads) = discriminatorGradients(fake: fake, real: waveTarget)
        discriminatorOptimizer.update(
            model: discriminator, gradients: vitsClipGradNorm(discGrads, maxNorm: config.maxGradNorm))
        eval(discriminator, discriminatorOptimizer)

        // 2. The generator, against the updated discriminator.
        MLXRandom.seed(seed)
        let (genValues, genGrads) = generatorGradients(batch, noise)
        generatorOptimizer.update(model: model, gradients: vitsClipGradNorm(genGrads, maxNorm: config.maxGradNorm))
        eval(model, generatorOptimizer)
        // Every batch has its own shape, so freed buffers rarely fit the
        // next step; kept, the cache grows by gigabytes a step.
        Memory.clearCache()

        let g = genValues.map { $0.item(Float.self) }
        let d = discValues.map { $0.item(Float.self) }
        return VitsTrainingLosses(
            duration: g[1], mel: g[2], kl: g[3], fmaps: g[4], gen: g[5],
            disc: d[1], realDisc: d[2], fakeDisc: d[3])
    }
}

// MARK: - Data

/// A clip ready to train on: its tokens, audio and spectrograms.
public struct VitsTrainingClip {
    public let name: String
    public let ids: [Int32]
    public let waveform: MLXArray  // (samples,)
    public let labels: MLXArray  // (frames, bins)
    public let mel: MLXArray  // (frames, mels)
}

public extension VitsTrainer {
    /// A clip from its audio and transcript, or nil if it is too short,
    /// too long, or has nothing the voice can read.
    func clip(name: String, audio: MLXArray, text: String) throws -> VitsTrainingClip? {
        let seconds = Double(audio.dim(0)) / Double(model.sampleRate)
        guard (config.minDuration ... config.maxDuration).contains(seconds),
              let tokenizer = model.tokenizer
        else { return nil }
        let ids = try tokenizer.encode(text).prefix(config.maxTokensLength + 1).map { Int32($0) }
        guard ids.count > 1 else { return nil }
        let (magnitudes, mel) = spectrogram(audio[.newAxis, 0...])
        return VitsTrainingClip(name: name, ids: ids, waveform: audio, labels: magnitudes[0], mel: mel[0])
    }

    /// Clips from a folder's `metadata.jsonl` of `{"file_name", "text"}`, as
    /// a Hugging Face audiofolder holds them, read at the voice's rate.
    /// `textMap`'s replacements are made in each transcript first.
    func loadClips(folder: URL, textMap: [String: String] = [:]) throws -> [VitsTrainingClip] {
        let lines = try String(contentsOf: folder.appendingPathComponent("metadata.jsonl"), encoding: .utf8)
            .split(whereSeparator: \.isNewline)
        var clips: [VitsTrainingClip] = []
        for line in lines where !line.trimmingCharacters(in: .whitespaces).isEmpty {
            guard let row = try JSONSerialization.jsonObject(with: Data(line.utf8)) as? [String: Any],
                  let file = row["file_name"] as? String, var text = row["text"] as? String
            else { continue }
            for (old, new) in textMap {
                text = text.replacingOccurrences(of: old, with: new)
            }
            let (_, audio) = try loadAudioArray(from: folder.appendingPathComponent(file), sampleRate: model.sampleRate)
            if let clip = try clip(name: file, audio: audio.asType(.float32).reshaped(-1), text: text) {
                clips.append(clip)
            }
        }
        return clips
    }

    /// Clips padded into one batch.
    static func collate(_ clips: [VitsTrainingClip]) -> VitsTrainingBatch {
        func pad(_ arrays: [MLXArray]) -> (MLXArray, MLXArray) {
            let n = arrays.map { $0.dim(0) }.max() ?? 0
            let padded = arrays.map { a -> MLXArray in
                var widths = [IntOrPair](repeating: 0, count: a.ndim)
                widths[0] = IntOrPair((0, n - a.dim(0)))
                return MLX.padded(a, widths: widths)
            }
            let masks = arrays.map { a in
                concatenated([MLXArray.ones([a.dim(0)]), MLXArray.zeros([n - a.dim(0)])])
            }
            return (stacked(padded), stacked(masks))
        }
        let (ids, attentionMask) = pad(clips.map { MLXArray($0.ids) })
        let (labels, labelsMask) = pad(clips.map(\.labels))
        let (mel, _) = pad(clips.map(\.mel))
        let (waveform, _) = pad(clips.map(\.waveform))
        return VitsTrainingBatch(
            inputIds: ids.asType(.int32), attentionMask: attentionMask, labels: labels,
            labelsMask: labelsMask, mel: mel, waveform: waveform)
    }

    /// Trains on `clips` for the config's epochs, shuffled each epoch, and
    /// reports each step. Stops after `maxSteps` if given, or when
    /// `onStep` returns false.
    func train(
        _ clips: [VitsTrainingClip], maxSteps: Int? = nil,
        onStep: (_ epoch: Int, _ step: Int, _ losses: VitsTrainingLosses) -> Bool = { _, _, _ in true }
    ) {
        var step = 0
        for epoch in 0 ..< config.epochs {
            setEpoch(epoch)
            let order = clips.indices.shuffled()
            for start in stride(from: 0, to: order.count, by: config.batchSize) {
                let batch = Self.collate(order[start ..< min(start + config.batchSize, order.count)].map { clips[$0] })
                let losses = self.step(batch)
                step += 1
                if !onStep(epoch, step, losses) { return }
                if let maxSteps, step >= maxSteps { return }
            }
        }
    }
}

// MARK: - Loading and saving

public extension VitsModel {
    /// The generator and discriminator from a training checkpoint, ready
    /// to train.
    static func loadForTraining(_ modelDir: URL) throws -> (VitsModel, VitsDiscriminator) {
        let model = try VitsModel.fromModelDirectory(modelDir, forTraining: true)
        let configData = try Data(contentsOf: modelDir.appendingPathComponent("config.json"))
        let discriminator = VitsDiscriminator(try JSONDecoder().decode(VitsDiscriminatorConfig.self, from: configData))
        var weights: [String: MLXArray] = [:]
        for file in try FileManager.default.contentsOfDirectory(at: modelDir, includingPropertiesForKeys: nil)
        where file.pathExtension == "safetensors" {
            for (key, value) in try loadArrays(url: file) where key.hasPrefix("discriminator.") {
                weights[key] = value
            }
        }
        let discriminatorWeights = discriminator.sanitize(weights: weights)
        guard !discriminatorWeights.isEmpty else {
            throw VitsError.unsupported(
                "checkpoint without a discriminator: make a training checkpoint with finetune-hf-vits's "
                    + "convert_original_discriminator_checkpoint.py")
        }
        try discriminator.update(parameters: ModuleParameters.unflattened(discriminatorWeights), verify: .all)
        model.train(true)
        discriminator.train(true)
        eval(model, discriminator)
        return (model, discriminator)
    }

    /// This model's weights in transformers' names and layout, the inverse
    /// of `sanitize(weights:)`.
    func toTransformers() -> [String: MLXArray] {
        var out: [String: MLXArray] = [:]
        for (key, value) in parameters().flattened() {
            var value = value
            if key.hasSuffix(".translate") || key.hasSuffix(".log_scale") {
                value = value.reshaped(-1, 1)
            } else if value.ndim == 3, !key.hasSuffix("emb_rel_k"), !key.hasSuffix("emb_rel_v"),
                      !key.hasSuffix("weight_g")
            {
                value = key.hasPrefix("decoder.upsampler.") ? value.transposed(2, 0, 1) : value.transposed(0, 2, 1)
            } else if key.hasSuffix("weight_g"), key.hasPrefix("decoder.upsampler.") {
                value = value.reshaped(-1, 1, 1)
            }
            out[key] = value
        }
        return out
    }

    /// The generator in transformers' format, weight norm folded where an
    /// inference checkpoint has none, with the source's config and
    /// tokenizer: a voice that loads here and in transformers.
    func saveTrained(from source: URL, to output: URL) throws {
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
        let weights = toTransformers()
        var folded: [String: MLXArray] = [:]
        for (key, value) in weights {
            let foldable = key.hasPrefix("decoder.")
                || (key.hasPrefix("flow.flows.") && (key.contains(".conv_pre.") || key.contains(".conv_post.")))
            if foldable, key.hasSuffix(".weight_v") {
                let base = String(key.dropLast(".weight_v".count))
                // In PyTorch's layout the norm is over all but the first axis,
                // the input channel of a transposed convolution included.
                let g = weights[base + ".weight_g"]!
                let norm = sqrt(sum(value * value, axes: [1, 2], keepDims: true))
                folded[base + ".weight"] = g * value / norm
            } else if foldable, key.hasSuffix(".weight_g") {
                continue
            } else {
                folded[key] = value
            }
        }
        eval(Array(folded.values))
        try save(arrays: folded, metadata: ["format": "pt"], url: output.appendingPathComponent("model.safetensors"))

        var config = (try JSONSerialization.jsonObject(
            with: Data(contentsOf: source.appendingPathComponent("config.json"))) as? [String: Any]) ?? [:]
        config = config.filter { !$0.key.hasPrefix("discriminator_") }
        config["architectures"] = ["VitsModel"]
        try Data(vitsJSON(config).utf8).write(to: output.appendingPathComponent("config.json"))
        for name in ["vocab.json", "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json"] {
            let from = source.appendingPathComponent(name)
            let to = output.appendingPathComponent(name)
            guard FileManager.default.fileExists(atPath: from.path) else { continue }
            try? FileManager.default.removeItem(at: to)
            try FileManager.default.copyItem(at: from, to: to)
        }
    }
}

/// JSON as Python's `json` writes it, as far as transformers cares: a
/// float stays a float (`5.0`, not `5`), which JSONSerialization doesn't
/// keep and transformers 5 checks. Keys sorted, two-space indents.
func vitsJSON(_ value: Any, indent: String = "") -> String {
    let inner = indent + "  "
    switch value {
    case let dict as [String: Any]:
        guard !dict.isEmpty else { return "{}" }
        let items = dict.keys.sorted().map { "\(inner)\(vitsJSON($0)): \(vitsJSON(dict[$0]!, indent: inner))" }
        return "{\n" + items.joined(separator: ",\n") + "\n\(indent)}"
    case let array as [Any]:
        guard !array.isEmpty else { return "[]" }
        return "[\n" + array.map { inner + vitsJSON($0, indent: inner) }.joined(separator: ",\n") + "\n\(indent)]"
    case let string as String:
        let data = try! JSONSerialization.data(withJSONObject: [string], options: [.withoutEscapingSlashes])
        return String(decoding: data.dropFirst().dropLast(), as: UTF8.self)
    case let number as NSNumber:
        if CFGetTypeID(number) == CFBooleanGetTypeID() { return number.boolValue ? "true" : "false" }
        if CFNumberIsFloatType(number) {
            let double = number.doubleValue
            if double.isFinite, double == double.rounded(), abs(double) < 1e16 { return String(format: "%.1f", double) }
            return "\(double)"
        }
        return number.stringValue
    case is NSNull:
        return "null"
    default:
        return "null"
    }
}
