import Foundation
@preconcurrency import MLX
import MLXNN

// VITS's layers, ported from mlx-audio's `vits/modules.py`, itself a port
// of transformers' `modeling_vits.py`. Names follow transformers', so its
// checkpoints (MMS-TTS among them) load without renaming. Tensors are
// channels-last, (batch, time, channels); masks are (batch, time, 1).

private func norm(_ w: MLXArray, axes: [Int]) -> MLXArray {
    sqrt(sum(w * w, axes: axes, keepDims: true))
}

private func lastAxis(_ x: MLXArray, _ range: Range<Int>) -> MLXArray {
    x[.ellipsis, range]
}

/// The last axis reversed: VITS's flows swap their two halves this way.
private func flipped(_ x: MLXArray) -> MLXArray {
    x[.ellipsis, .stride(by: -1)]
}

// MARK: - Convolutions

/// A 1-D convolution that holds its weight plainly or weight-normed.
///
/// Weight-normed, it holds `weight_g` and `weight_v` as transformers'
/// checkpoints do, and the weight is `weight_g * weight_v / |weight_v|`,
/// the norm taken over each output channel.
final class VitsConv1d: Module {
    let stride: Int
    let padding: Int
    let dilation: Int
    let groups: Int

    var weight: MLXArray?
    @ParameterInfo(key: "weight_g") var weightG: MLXArray?
    @ParameterInfo(key: "weight_v") var weightV: MLXArray?
    var bias: MLXArray?

    init(
        _ inChannels: Int,
        _ outChannels: Int,
        kernelSize: Int,
        stride: Int = 1,
        padding: Int = 0,
        dilation: Int = 1,
        groups: Int = 1,
        bias: Bool = true,
        weightNorm: Bool = false
    ) {
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        let shape = [outChannels, kernelSize, inChannels / groups]
        if weightNorm {
            _weightV.wrappedValue = MLXArray.zeros(shape)
            _weightG.wrappedValue = MLXArray.ones([outChannels, 1, 1])
        } else {
            weight = MLXArray.zeros(shape)
            _weightV.wrappedValue = nil
            _weightG.wrappedValue = nil
        }
        self.bias = bias ? MLXArray.zeros([outChannels]) : nil
    }

    var effectiveWeight: MLXArray {
        if let weightV, let weightG {
            return weightG * weightV / norm(weightV, axes: [1, 2])
        }
        return weight!
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let y = conv1d(
            x, effectiveWeight, stride: stride, padding: padding, dilation: dilation, groups: groups)
        if let bias { return y + bias }
        return y
    }
}

/// A transposed 1-D convolution, plain or weight-normed.
///
/// PyTorch normalises a transposed convolution's weight over its *input*
/// channel. In MLX's layout, (out, kernel, in), that is the last axis, so
/// the norm is over the other two.
final class VitsConvTranspose1d: Module {
    let stride: Int
    let padding: Int

    var weight: MLXArray?
    @ParameterInfo(key: "weight_g") var weightG: MLXArray?
    @ParameterInfo(key: "weight_v") var weightV: MLXArray?
    var bias: MLXArray?

    init(
        _ inChannels: Int, _ outChannels: Int, kernelSize: Int, stride: Int, padding: Int, bias: Bool = true,
        weightNorm: Bool = false
    ) {
        self.stride = stride
        self.padding = padding
        let shape = [outChannels, kernelSize, inChannels]
        if weightNorm {
            _weightV.wrappedValue = MLXArray.zeros(shape)
            _weightG.wrappedValue = MLXArray.ones([1, 1, inChannels])
        } else {
            weight = MLXArray.zeros(shape)
            _weightV.wrappedValue = nil
            _weightG.wrappedValue = nil
        }
        self.bias = bias ? MLXArray.zeros([outChannels]) : nil
    }

    var effectiveWeight: MLXArray {
        if let weightV, let weightG {
            return weightG * weightV / norm(weightV, axes: [0, 1])
        }
        return weight!
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let y = convTransposed1d(x, effectiveWeight, stride: stride, padding: padding)
        if let bias { return y + bias }
        return y
    }
}

// MARK: - WaveNet and the posterior encoder

final class VitsWaveNet: Module {
    let hiddenSize: Int
    let numLayers: Int

    @ModuleInfo(key: "cond_layer") var condLayer: VitsConv1d?
    @ModuleInfo(key: "in_layers") var inLayers: [VitsConv1d]
    @ModuleInfo(key: "res_skip_layers") var resSkipLayers: [VitsConv1d]
    let dropout: Dropout

    init(_ config: VitsConfig, numLayers: Int) {
        hiddenSize = config.hiddenSize
        self.numLayers = numLayers
        dropout = Dropout(p: config.wavenetDropout)
        if config.speakerEmbeddingSize != 0 {
            _condLayer.wrappedValue = VitsConv1d(
                config.speakerEmbeddingSize, 2 * config.hiddenSize * numLayers, kernelSize: 1, weightNorm: true)
        } else {
            _condLayer.wrappedValue = nil
        }
        var inLayers: [VitsConv1d] = []
        var resSkipLayers: [VitsConv1d] = []
        for i in 0 ..< numLayers {
            let dilation = Int(pow(Double(config.wavenetDilationRate), Double(i)))
            let padding = (config.wavenetKernelSize * dilation - dilation) / 2
            inLayers.append(VitsConv1d(
                config.hiddenSize, 2 * config.hiddenSize, kernelSize: config.wavenetKernelSize,
                padding: padding, dilation: dilation, weightNorm: true))
            let resSkipChannels = i < numLayers - 1 ? 2 * config.hiddenSize : config.hiddenSize
            resSkipLayers.append(VitsConv1d(config.hiddenSize, resSkipChannels, kernelSize: 1, weightNorm: true))
        }
        _inLayers.wrappedValue = inLayers
        _resSkipLayers.wrappedValue = resSkipLayers
    }

    func callAsFunction(_ inputs: MLXArray, mask: MLXArray, conditioning: MLXArray? = nil) -> MLXArray {
        var inputs = inputs
        var outputs = MLXArray.zeros(like: inputs)
        let conditioning = conditioning.map { condLayer!($0) }
        for i in 0 ..< numLayers {
            let hidden = inLayers[i](inputs)
            let global: MLXArray
            if let conditioning {
                let offset = i * 2 * hiddenSize
                global = lastAxis(conditioning, offset ..< offset + 2 * hiddenSize)
            } else {
                global = MLXArray.zeros(like: hidden)
            }
            let summed = hidden + global
            var acts = tanh(lastAxis(summed, 0 ..< hiddenSize))
                * sigmoid(summed[.ellipsis, hiddenSize...])
            acts = dropout(acts)
            let resSkip = resSkipLayers[i](acts)
            if i < numLayers - 1 {
                inputs = (inputs + lastAxis(resSkip, 0 ..< hiddenSize)) * mask
                outputs = outputs + resSkip[.ellipsis, hiddenSize...]
            } else {
                outputs = outputs + resSkip
            }
        }
        return outputs * mask
    }
}

final class VitsPosteriorEncoder: Module {
    @ModuleInfo(key: "conv_pre") var convPre: VitsConv1d
    @ModuleInfo var wavenet: VitsWaveNet
    @ModuleInfo(key: "conv_proj") var convProj: VitsConv1d

    init(_ config: VitsConfig) {
        _convPre.wrappedValue = VitsConv1d(config.spectrogramBins, config.hiddenSize, kernelSize: 1)
        _wavenet.wrappedValue = VitsWaveNet(config, numLayers: config.posteriorEncoderNumWavenetLayers)
        _convProj.wrappedValue = VitsConv1d(config.hiddenSize, config.flowSize * 2, kernelSize: 1)
    }

    func callAsFunction(
        _ inputs: MLXArray, mask: MLXArray, conditioning: MLXArray? = nil, noise: MLXArray? = nil
    ) -> (sampled: MLXArray, mean: MLXArray, logStddev: MLXArray) {
        var hidden = convPre(inputs) * mask
        hidden = wavenet(hidden, mask: mask, conditioning: conditioning)
        let stats = convProj(hidden) * mask
        let halves = split(stats, parts: 2, axis: -1)
        let noise = noise ?? MLXRandom.normal(halves[0].shape)
        let sampled = (halves[0] + noise * exp(halves[1])) * mask
        return (sampled, halves[0], halves[1])
    }
}

// MARK: - HiFi-GAN

final class VitsHifiGanResidualBlock: Module {
    let leakyReluSlope: Float
    @ModuleInfo var convs1: [VitsConv1d]
    @ModuleInfo var convs2: [VitsConv1d]

    init(channels: Int, kernelSize: Int, dilation: [Int], leakyReluSlope: Float, weightNorm: Bool) {
        self.leakyReluSlope = leakyReluSlope
        _convs1.wrappedValue = dilation.map {
            VitsConv1d(
                channels, channels, kernelSize: kernelSize, padding: (kernelSize * $0 - $0) / 2, dilation: $0,
                weightNorm: weightNorm)
        }
        _convs2.wrappedValue = dilation.map { _ in
            VitsConv1d(
                channels, channels, kernelSize: kernelSize, padding: (kernelSize - 1) / 2, weightNorm: weightNorm)
        }
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var hidden = x
        for (conv1, conv2) in zip(convs1, convs2) {
            let residual = hidden
            hidden = leakyRelu(hidden, negativeSlope: leakyReluSlope)
            hidden = conv1(hidden)
            hidden = leakyRelu(hidden, negativeSlope: leakyReluSlope)
            hidden = conv2(hidden)
            hidden = hidden + residual
        }
        return hidden
    }
}

final class VitsHifiGan: Module {
    let leakyReluSlope: Float
    let numKernels: Int
    let numUpsamples: Int

    @ModuleInfo(key: "conv_pre") var convPre: VitsConv1d
    @ModuleInfo var upsampler: [VitsConvTranspose1d]
    @ModuleInfo var resblocks: [VitsHifiGanResidualBlock]
    @ModuleInfo(key: "conv_post") var convPost: VitsConv1d
    @ModuleInfo var cond: VitsConv1d?

    /// `weightNorm` holds the upsampling and residual layers weight-normed,
    /// as finetune-hf-vits trains them.
    init(_ config: VitsConfig, weightNorm: Bool = false) {
        leakyReluSlope = config.leakyReluSlope
        numKernels = config.resblockKernelSizes.count
        numUpsamples = config.upsampleRates.count
        let initial = config.upsampleInitialChannel
        _convPre.wrappedValue = VitsConv1d(config.flowSize, initial, kernelSize: 7, padding: 3)
        _upsampler.wrappedValue = zip(config.upsampleRates, config.upsampleKernelSizes).enumerated().map {
            i, rateAndKernel in
            let (rate, kernel) = rateAndKernel
            return VitsConvTranspose1d(
                initial / (1 << i), initial / (1 << (i + 1)),
                kernelSize: kernel, stride: rate, padding: (kernel - rate) / 2, weightNorm: weightNorm)
        }
        var resblocks: [VitsHifiGanResidualBlock] = []
        var channels = initial
        for i in 0 ..< numUpsamples {
            channels = initial / (1 << (i + 1))
            for (kernel, dilation) in zip(config.resblockKernelSizes, config.resblockDilationSizes) {
                resblocks.append(VitsHifiGanResidualBlock(
                    channels: channels, kernelSize: kernel, dilation: dilation,
                    leakyReluSlope: config.leakyReluSlope, weightNorm: weightNorm))
            }
        }
        _resblocks.wrappedValue = resblocks
        _convPost.wrappedValue = VitsConv1d(channels, 1, kernelSize: 7, padding: 3, bias: false)
        if config.speakerEmbeddingSize != 0 {
            _cond.wrappedValue = VitsConv1d(config.speakerEmbeddingSize, initial, kernelSize: 1)
        } else {
            _cond.wrappedValue = nil
        }
    }

    func callAsFunction(_ spectrogram: MLXArray, conditioning: MLXArray? = nil) -> MLXArray {
        var hidden = convPre(spectrogram)
        if let conditioning, let cond {
            hidden = hidden + cond(conditioning)
        }
        for i in 0 ..< numUpsamples {
            hidden = leakyRelu(hidden, negativeSlope: leakyReluSlope)
            hidden = upsampler[i](hidden)
            var resState = resblocks[i * numKernels](hidden)
            for j in 1 ..< numKernels {
                resState = resState + resblocks[i * numKernels + j](hidden)
            }
            hidden = resState / Float(numKernels)
        }
        // The last activation uses PyTorch's default slope, not the config's.
        hidden = leakyRelu(hidden, negativeSlope: 0.01)
        return tanh(convPost(hidden))
    }
}

// MARK: - The flow

final class VitsResidualCouplingLayer: Module {
    let halfChannels: Int
    @ModuleInfo(key: "conv_pre") var convPre: VitsConv1d
    @ModuleInfo var wavenet: VitsWaveNet
    @ModuleInfo(key: "conv_post") var convPost: VitsConv1d

    /// `weightNorm` holds the input and output layers weight-normed, as
    /// finetune-hf-vits trains them.
    init(_ config: VitsConfig, weightNorm: Bool = false) {
        halfChannels = config.flowSize / 2
        _convPre.wrappedValue = VitsConv1d(halfChannels, config.hiddenSize, kernelSize: 1, weightNorm: weightNorm)
        _wavenet.wrappedValue = VitsWaveNet(config, numLayers: config.priorEncoderNumWavenetLayers)
        _convPost.wrappedValue = VitsConv1d(config.hiddenSize, halfChannels, kernelSize: 1, weightNorm: weightNorm)
    }

    func callAsFunction(
        _ inputs: MLXArray, mask: MLXArray, conditioning: MLXArray? = nil, reverse: Bool = false
    ) -> MLXArray {
        let halves = split(inputs, parts: 2, axis: -1)
        var hidden = convPre(halves[0]) * mask
        hidden = wavenet(hidden, mask: mask, conditioning: conditioning)
        let mean = convPost(hidden) * mask
        // Mean only: the scale is 1, so the log-determinant is 0.
        let second = reverse ? (halves[1] - mean) * mask : mean + halves[1] * mask
        return concatenated([halves[0], second], axis: -1)
    }
}

final class VitsResidualCouplingBlock: Module {
    @ModuleInfo var flows: [VitsResidualCouplingLayer]

    init(_ config: VitsConfig, weightNorm: Bool = false) {
        _flows.wrappedValue = (0 ..< config.priorEncoderNumFlows).map { _ in
            VitsResidualCouplingLayer(config, weightNorm: weightNorm)
        }
    }

    func callAsFunction(
        _ inputs: MLXArray, mask: MLXArray, conditioning: MLXArray? = nil, reverse: Bool = false
    ) -> MLXArray {
        var x = inputs
        if !reverse {
            for flow in flows {
                x = flipped(flow(x, mask: mask, conditioning: conditioning))
            }
        } else {
            for flow in flows.reversed() {
                x = flow(flipped(x), mask: mask, conditioning: conditioning, reverse: true)
            }
        }
        return x
    }
}

// MARK: - Duration prediction

final class VitsDilatedDepthSeparableConv: Module {
    let numLayers: Int
    let dropout: Dropout
    @ModuleInfo(key: "convs_dilated") var convsDilated: [VitsConv1d]
    @ModuleInfo(key: "convs_pointwise") var convsPointwise: [VitsConv1d]
    @ModuleInfo(key: "norms_1") var norms1: [LayerNorm]
    @ModuleInfo(key: "norms_2") var norms2: [LayerNorm]

    init(_ config: VitsConfig, dropoutRate: Float = 0) {
        let kernel = config.durationPredictorKernelSize
        let channels = config.hiddenSize
        numLayers = config.depthSeparableNumLayers
        dropout = Dropout(p: dropoutRate)
        _convsDilated.wrappedValue = (0 ..< numLayers).map { i in
            let dilation = Int(pow(Double(kernel), Double(i)))
            return VitsConv1d(
                channels, channels, kernelSize: kernel, padding: (kernel * dilation - dilation) / 2,
                dilation: dilation, groups: channels)
        }
        _convsPointwise.wrappedValue = (0 ..< numLayers).map { _ in VitsConv1d(channels, channels, kernelSize: 1) }
        _norms1.wrappedValue = (0 ..< numLayers).map { _ in LayerNorm(dimensions: channels) }
        _norms2.wrappedValue = (0 ..< numLayers).map { _ in LayerNorm(dimensions: channels) }
    }

    func callAsFunction(_ inputs: MLXArray, mask: MLXArray, conditioning: MLXArray? = nil) -> MLXArray {
        var x = conditioning.map { inputs + $0 } ?? inputs
        for i in 0 ..< numLayers {
            var hidden = convsDilated[i](x * mask)
            hidden = gelu(norms1[i](hidden))
            hidden = convsPointwise[i](hidden)
            hidden = gelu(norms2[i](hidden))
            hidden = dropout(hidden)
            x = x + hidden
        }
        return x * mask
    }
}

/// A flow of the stochastic duration predictor: an affine one or a spline.
class VitsDurationFlow: Module {
    func callAsFunction(
        _ inputs: MLXArray, mask: MLXArray, conditioning: MLXArray?, reverse: Bool
    ) -> (MLXArray, MLXArray?) {
        fatalError("a VitsDurationFlow subclass must override callAsFunction")
    }
}

final class VitsConvFlow: VitsDurationFlow {
    let filterChannels: Int
    let halfChannels: Int
    let numBins: Int
    let tailBound: Float
    @ModuleInfo(key: "conv_pre") var convPre: VitsConv1d
    @ModuleInfo(key: "conv_dds") var convDDS: VitsDilatedDepthSeparableConv
    @ModuleInfo(key: "conv_proj") var convProj: VitsConv1d

    init(_ config: VitsConfig) {
        filterChannels = config.hiddenSize
        halfChannels = config.depthSeparableChannels / 2
        numBins = config.durationPredictorFlowBins
        tailBound = config.durationPredictorTailBound
        _convPre.wrappedValue = VitsConv1d(halfChannels, filterChannels, kernelSize: 1)
        _convDDS.wrappedValue = VitsDilatedDepthSeparableConv(config)
        _convProj.wrappedValue = VitsConv1d(filterChannels, halfChannels * (numBins * 3 - 1), kernelSize: 1)
    }

    override func callAsFunction(
        _ inputs: MLXArray, mask: MLXArray, conditioning: MLXArray?, reverse: Bool
    ) -> (MLXArray, MLXArray?) {
        let halves = split(inputs, parts: 2, axis: -1)
        var hidden = convPre(halves[0])
        hidden = convDDS(hidden, mask: mask, conditioning: conditioning)
        hidden = convProj(hidden) * mask

        let (batch, length, channels) = (halves[0].dim(0), halves[0].dim(1), halves[0].dim(2))
        // (batch, time, channels, bins), channel-major as transformers splits it.
        hidden = hidden.reshaped(batch, length, channels, -1)
        let scale = Float(filterChannels).squareRoot()
        let widths = hidden[.ellipsis, ..<numBins] / scale
        let heights = hidden[.ellipsis, numBins ..< 2 * numBins] / scale
        let derivatives = hidden[.ellipsis, (2 * numBins)...]

        let (second, logAbsDet) = VitsSpline.unconstrained(
            halves[1], widths: widths, heights: heights, derivatives: derivatives,
            reverse: reverse, tailBound: tailBound)
        let outputs = concatenated([halves[0], second], axis: -1) * mask
        if !reverse {
            return (outputs, sum(logAbsDet * mask, axes: [1, 2]))
        }
        return (outputs, nil)
    }
}

final class VitsElementwiseAffine: VitsDurationFlow {
    // Per channel, held as (channels,); transformers holds (channels, 1).
    var translate: MLXArray
    @ParameterInfo(key: "log_scale") var logScale: MLXArray

    init(_ config: VitsConfig) {
        translate = MLXArray.zeros([config.depthSeparableChannels])
        _logScale.wrappedValue = MLXArray.zeros([config.depthSeparableChannels])
    }

    override func callAsFunction(
        _ inputs: MLXArray, mask: MLXArray, conditioning: MLXArray?, reverse: Bool
    ) -> (MLXArray, MLXArray?) {
        if !reverse {
            let outputs = (translate + exp(logScale) * inputs) * mask
            return (outputs, sum(logScale * mask, axes: [1, 2]))
        }
        return ((inputs - translate) * exp(-logScale) * mask, nil)
    }
}

final class VitsStochasticDurationPredictor: Module {
    @ModuleInfo(key: "conv_pre") var convPre: VitsConv1d
    @ModuleInfo(key: "conv_proj") var convProj: VitsConv1d
    @ModuleInfo(key: "conv_dds") var convDDS: VitsDilatedDepthSeparableConv
    @ModuleInfo var cond: VitsConv1d?
    @ModuleInfo var flows: [VitsDurationFlow]
    @ModuleInfo(key: "post_conv_pre") var postConvPre: VitsConv1d
    @ModuleInfo(key: "post_conv_proj") var postConvProj: VitsConv1d
    @ModuleInfo(key: "post_conv_dds") var postConvDDS: VitsDilatedDepthSeparableConv
    @ModuleInfo(key: "post_flows") var postFlows: [VitsDurationFlow]

    init(_ config: VitsConfig) {
        let filterChannels = config.hiddenSize
        _convPre.wrappedValue = VitsConv1d(filterChannels, filterChannels, kernelSize: 1)
        _convProj.wrappedValue = VitsConv1d(filterChannels, filterChannels, kernelSize: 1)
        _convDDS.wrappedValue = VitsDilatedDepthSeparableConv(config, dropoutRate: config.durationPredictorDropout)
        if config.speakerEmbeddingSize != 0 {
            _cond.wrappedValue = VitsConv1d(config.speakerEmbeddingSize, filterChannels, kernelSize: 1)
        } else {
            _cond.wrappedValue = nil
        }
        let flows: () -> [VitsDurationFlow] = {
            [VitsElementwiseAffine(config)]
                + (0 ..< config.durationPredictorNumFlows).map { _ in VitsConvFlow(config) }
        }
        _flows.wrappedValue = flows()
        _postConvPre.wrappedValue = VitsConv1d(1, filterChannels, kernelSize: 1)
        _postConvProj.wrappedValue = VitsConv1d(filterChannels, filterChannels, kernelSize: 1)
        _postConvDDS.wrappedValue = VitsDilatedDepthSeparableConv(
            config, dropoutRate: config.durationPredictorDropout)
        _postFlows.wrappedValue = flows()
    }

    private func encode(_ inputs: MLXArray, mask: MLXArray, conditioning: MLXArray?) -> MLXArray {
        var x = convPre(stopGradient(inputs))
        if let conditioning, let cond {
            x = x + cond(stopGradient(conditioning))
        }
        x = convDDS(x, mask: mask)
        return convProj(x) * mask
    }

    /// The negative log-likelihood of `durations`, (batch, time, 1), per
    /// item: what training minimises. `noise`, if given, replaces the
    /// random draw.
    func negativeLogLikelihood(
        _ inputs: MLXArray, mask: MLXArray, conditioning: MLXArray? = nil,
        durations: MLXArray, noise: MLXArray? = nil
    ) -> MLXArray {
        let x = encode(inputs, mask: mask, conditioning: conditioning)
        let log2pi = Float(log(2 * Double.pi))

        var hidden = postConvPre(durations)
        hidden = postConvDDS(hidden, mask: mask)
        hidden = postConvProj(hidden) * mask
        let randomPosterior = (noise ?? MLXRandom.normal([durations.dim(0), durations.dim(1), 2])) * mask

        var logDeterminantPosterior = MLXArray.zeros([durations.dim(0)])
        var latentsPosterior = randomPosterior
        // No flip after the first flow, the affine one, as in the original
        // VITS and finetune-hf-vits; transformers' copy flips there too,
        // which its inference never reaches.
        for (i, flow) in postFlows.enumerated() {
            let (latents, logDeterminant) = flow(
                latentsPosterior, mask: mask, conditioning: x + hidden, reverse: false)
            latentsPosterior = i > 0 ? flipped(latents) : latents
            logDeterminantPosterior = logDeterminantPosterior + logDeterminant!
        }
        let posteriorHalves = split(latentsPosterior, parts: 2, axis: -1)
        logDeterminantPosterior = logDeterminantPosterior + sum(
            (logSigmoid(posteriorHalves[0]) + logSigmoid(-posteriorHalves[0])) * mask, axes: [1, 2])
        let logq = sum(-0.5 * (log2pi + randomPosterior.square()) * mask, axes: [1, 2])
            - logDeterminantPosterior

        var first = (durations - sigmoid(posteriorHalves[0])) * mask
        first = log(maximum(first, MLXArray(Float(1e-5)))) * mask
        var logDeterminantSum = sum(-first, axes: [1, 2])

        var latents = concatenated([first, posteriorHalves[1]], axis: -1)
        for (i, flow) in flows.enumerated() {
            let (next, logDeterminant) = flow(latents, mask: mask, conditioning: x, reverse: false)
            latents = i > 0 ? flipped(next) : next
            logDeterminantSum = logDeterminantSum + logDeterminant!
        }
        let nll = sum(0.5 * (log2pi + latents.square()) * mask, axes: [1, 2]) - logDeterminantSum
        return nll + logq
    }

    /// Sampled log-durations, (batch, time, 1). `noise`, if given, replaces
    /// the random draw: tests pass the same to both ports.
    func sample(
        _ inputs: MLXArray, mask: MLXArray, conditioning: MLXArray? = nil,
        noiseScale: Float = 1, noise: MLXArray? = nil
    ) -> MLXArray {
        let x = encode(inputs, mask: mask, conditioning: conditioning)

        // transformers drops one flow here.
        var reversed = Array(flows.reversed())
        reversed = Array(reversed.dropLast(2)) + [reversed.last!]
        var latents = (noise ?? MLXRandom.normal([x.dim(0), x.dim(1), 2])) * noiseScale
        for flow in reversed {
            latents = flipped(latents)
            latents = flow(latents, mask: mask, conditioning: x, reverse: true).0
        }
        return latents[.ellipsis, ..<1]
    }
}

final class VitsDurationPredictor: Module {
    let dropout: Dropout
    @ModuleInfo(key: "conv_1") var conv1: VitsConv1d
    @ModuleInfo(key: "norm_1") var norm1: LayerNorm
    @ModuleInfo(key: "conv_2") var conv2: VitsConv1d
    @ModuleInfo(key: "norm_2") var norm2: LayerNorm
    @ModuleInfo var proj: VitsConv1d
    @ModuleInfo var cond: VitsConv1d?

    init(_ config: VitsConfig) {
        let kernel = config.durationPredictorKernelSize
        let filters = config.durationPredictorFilterChannels
        dropout = Dropout(p: config.durationPredictorDropout)
        _conv1.wrappedValue = VitsConv1d(config.hiddenSize, filters, kernelSize: kernel, padding: kernel / 2)
        _norm1.wrappedValue = LayerNorm(dimensions: filters, eps: config.layerNormEps)
        _conv2.wrappedValue = VitsConv1d(filters, filters, kernelSize: kernel, padding: kernel / 2)
        _norm2.wrappedValue = LayerNorm(dimensions: filters, eps: config.layerNormEps)
        _proj.wrappedValue = VitsConv1d(filters, 1, kernelSize: 1)
        if config.speakerEmbeddingSize != 0 {
            _cond.wrappedValue = VitsConv1d(config.speakerEmbeddingSize, config.hiddenSize, kernelSize: 1)
        } else {
            _cond.wrappedValue = nil
        }
    }

    func callAsFunction(_ inputs: MLXArray, mask: MLXArray, conditioning: MLXArray? = nil) -> MLXArray {
        var x = stopGradient(inputs)
        if let conditioning, let cond {
            x = x + cond(stopGradient(conditioning))
        }
        x = dropout(norm1(relu(conv1(x * mask))))
        x = dropout(norm2(relu(conv2(x * mask))))
        return proj(x * mask) * mask
    }
}

// MARK: - The text encoder

/// Multi-head attention with relative position embeddings in a window.
final class VitsAttention: Module {
    let embedDim: Int
    let numHeads: Int
    let windowSize: Int
    let headDim: Int
    let scaling: Float
    let dropout: Dropout

    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "out_proj") var outProj: Linear
    @ParameterInfo(key: "emb_rel_k") var embRelK: MLXArray?
    @ParameterInfo(key: "emb_rel_v") var embRelV: MLXArray?

    init(_ config: VitsConfig) {
        embedDim = config.hiddenSize
        numHeads = config.numAttentionHeads
        windowSize = config.windowSize
        headDim = embedDim / numHeads
        scaling = pow(Float(headDim), -0.5)
        dropout = Dropout(p: config.attentionDropout)
        _kProj.wrappedValue = Linear(embedDim, embedDim, bias: config.useBias)
        _vProj.wrappedValue = Linear(embedDim, embedDim, bias: config.useBias)
        _qProj.wrappedValue = Linear(embedDim, embedDim, bias: config.useBias)
        _outProj.wrappedValue = Linear(embedDim, embedDim, bias: config.useBias)
        if windowSize > 0 {
            _embRelK.wrappedValue = MLXArray.zeros([1, windowSize * 2 + 1, headDim])
            _embRelV.wrappedValue = MLXArray.zeros([1, windowSize * 2 + 1, headDim])
        } else {
            _embRelK.wrappedValue = nil
            _embRelV.wrappedValue = nil
        }
    }

    private func heads(_ x: MLXArray, _ batch: Int, _ length: Int) -> MLXArray {
        x.reshaped(batch, length, numHeads, headDim).transposed(0, 2, 1, 3)
            .reshaped(batch * numHeads, length, headDim)
    }

    func callAsFunction(_ hidden: MLXArray, attentionMask: MLXArray? = nil) -> MLXArray {
        let (batch, length) = (hidden.dim(0), hidden.dim(1))
        let query = heads(qProj(hidden) * scaling, batch, length)
        let key = heads(kProj(hidden), batch, length)
        let value = heads(vProj(hidden), batch, length)

        var weights = matmul(query, key.transposed(0, 2, 1))
        if let embRelK {
            let keyRel = relativeEmbeddings(embRelK, length)
            weights = weights + Self.relativeToAbsolute(matmul(query, keyRel.transposed(0, 2, 1)))
        }
        if let attentionMask {
            weights = (weights.reshaped(batch, numHeads, length, length) + attentionMask)
                .reshaped(batch * numHeads, length, length)
        }
        weights = softmax(weights, axis: -1, precise: true)
        let probs = dropout(weights)

        var output = matmul(probs, value)
        if let embRelV {
            let valueRel = relativeEmbeddings(embRelV, length)
            output = output + matmul(Self.absoluteToRelative(probs), valueRel)
        }
        output = output.reshaped(batch, numHeads, length, headDim).transposed(0, 2, 1, 3)
            .reshaped(batch, length, embedDim)
        return outProj(output)
    }

    private func relativeEmbeddings(_ embeddings: MLXArray, _ length: Int) -> MLXArray {
        var embeddings = embeddings
        let padLength = max(length - (windowSize + 1), 0)
        if padLength > 0 {
            embeddings = padded(embeddings, widths: [0, IntOrPair((padLength, padLength)), 0])
        }
        let start = max((windowSize + 1) - length, 0)
        return embeddings[0..., start ..< start + 2 * length - 1]
    }

    static func relativeToAbsolute(_ x: MLXArray) -> MLXArray {
        let (batchHeads, length) = (x.dim(0), x.dim(1))
        var flat = padded(x, widths: [0, 0, IntOrPair((0, 1))]).reshaped(batchHeads, length * 2 * length)
        flat = padded(flat, widths: [0, IntOrPair((0, length - 1))])
        return flat.reshaped(batchHeads, length + 1, 2 * length - 1)[0..., ..<length, (length - 1)...]
    }

    static func absoluteToRelative(_ x: MLXArray) -> MLXArray {
        let (batchHeads, length) = (x.dim(0), x.dim(1))
        var flat = padded(x, widths: [0, 0, IntOrPair((0, length - 1))])
            .reshaped(batchHeads, length * (2 * length - 1))
        flat = padded(flat, widths: [0, IntOrPair((length, 0))])
        return flat.reshaped(batchHeads, length, 2 * length)[0..., 0..., 1...]
    }
}

final class VitsFeedForward: Module {
    @ModuleInfo(key: "conv_1") var conv1: VitsConv1d
    @ModuleInfo(key: "conv_2") var conv2: VitsConv1d
    let dropout: Dropout
    let gelu: Bool
    let padding: (Int, Int)?

    init(_ config: VitsConfig) throws {
        let k = config.ffnKernelSize
        _conv1.wrappedValue = VitsConv1d(config.hiddenSize, config.ffnDim, kernelSize: k)
        _conv2.wrappedValue = VitsConv1d(config.ffnDim, config.hiddenSize, kernelSize: k)
        dropout = Dropout(p: config.activationDropout)
        guard config.hiddenAct == "relu" || config.hiddenAct == "gelu" else {
            throw VitsError.unsupported("hidden_act \(config.hiddenAct)")
        }
        gelu = config.hiddenAct == "gelu"
        padding = k > 1 ? ((k - 1) / 2, k / 2) : nil
    }

    private func pad(_ x: MLXArray) -> MLXArray {
        guard let padding else { return x }
        return padded(x, widths: [0, IntOrPair(padding), 0])
    }

    func callAsFunction(_ hidden: MLXArray, mask: MLXArray) -> MLXArray {
        var x = conv1(pad(hidden * mask))
        x = dropout(gelu ? MLXNN.gelu(x) : relu(x))
        x = conv2(pad(x * mask))
        return x * mask
    }
}

final class VitsEncoderLayer: Module {
    @ModuleInfo var attention: VitsAttention
    let dropout: Dropout
    @ModuleInfo(key: "layer_norm") var layerNorm: LayerNorm
    @ModuleInfo(key: "feed_forward") var feedForward: VitsFeedForward
    @ModuleInfo(key: "final_layer_norm") var finalLayerNorm: LayerNorm

    init(_ config: VitsConfig) throws {
        _attention.wrappedValue = VitsAttention(config)
        dropout = Dropout(p: config.hiddenDropout)
        _layerNorm.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: config.layerNormEps)
        _feedForward.wrappedValue = try VitsFeedForward(config)
        _finalLayerNorm.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: config.layerNormEps)
    }

    func callAsFunction(_ hidden: MLXArray, mask: MLXArray, attentionMask: MLXArray?) -> MLXArray {
        var x = layerNorm(hidden + dropout(attention(hidden, attentionMask: attentionMask)))
        x = finalLayerNorm(x + dropout(feedForward(x, mask: mask)))
        return x
    }
}

final class VitsEncoder: Module {
    @ModuleInfo var layers: [VitsEncoderLayer]
    /// In training, the chance of skipping each layer in a pass.
    var layerdrop: Float

    init(_ config: VitsConfig) throws {
        _layers.wrappedValue = try (0 ..< config.numHiddenLayers).map { _ in try VitsEncoderLayer(config) }
        layerdrop = config.layerdrop
    }

    func callAsFunction(_ hidden: MLXArray, mask: MLXArray, attentionMask: MLXArray?) -> MLXArray {
        // (batch, time) of 1 and 0 to an additive (batch, 1, 1, time).
        let additive = attentionMask.map { (1 - $0[0..., .newAxis, .newAxis, 0...]) * Float(-1e9) }
        var x = hidden * mask
        for layer in layers {
            // Drawn from MLX's generator, so seeding it repeats a pass exactly.
            if training, layerdrop > 0, MLXRandom.uniform(0 ..< 1).item(Float.self) < layerdrop {
                continue
            }
            x = layer(x, mask: mask, attentionMask: additive)
        }
        return x * mask
    }
}

final class VitsTextEncoder: Module {
    let hiddenSize: Int
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo var encoder: VitsEncoder
    @ModuleInfo var project: VitsConv1d

    init(_ config: VitsConfig) throws {
        hiddenSize = config.hiddenSize
        _embedTokens.wrappedValue = Embedding(embeddingCount: config.vocabSize, dimensions: config.hiddenSize)
        _encoder.wrappedValue = try VitsEncoder(config)
        _project.wrappedValue = VitsConv1d(config.hiddenSize, config.flowSize * 2, kernelSize: 1)
    }

    /// The encoded text, and the prior's means and log-variances.
    func callAsFunction(
        _ inputIds: MLXArray, mask: MLXArray, attentionMask: MLXArray?
    ) -> (hidden: MLXArray, priorMeans: MLXArray, priorLogVariances: MLXArray) {
        var hidden = embedTokens(inputIds) * Float(hiddenSize).squareRoot()
        hidden = encoder(hidden, mask: mask, attentionMask: attentionMask)
        let stats = project(hidden) * mask
        let halves = split(stats, parts: 2, axis: -1)
        return (hidden, halves[0], halves[1])
    }
}
