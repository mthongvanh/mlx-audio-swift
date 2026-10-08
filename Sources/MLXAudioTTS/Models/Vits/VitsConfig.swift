import Foundation

/// A VITS checkpoint's `config.json`, as transformers' `VitsConfig` writes it.
/// Every field has transformers' default, so a config that leaves one out
/// still loads.
public struct VitsConfig: Codable, Sendable {
    public var modelType: String = "vits"
    public var vocabSize: Int = 38
    public var hiddenSize: Int = 192
    public var numHiddenLayers: Int = 6
    public var numAttentionHeads: Int = 2
    public var windowSize: Int = 4
    public var useBias: Bool = true
    public var ffnDim: Int = 768
    public var layerdrop: Float = 0.1
    public var ffnKernelSize: Int = 3
    public var flowSize: Int = 192
    public var spectrogramBins: Int = 513
    public var hiddenAct: String = "relu"
    public var hiddenDropout: Float = 0.1
    public var attentionDropout: Float = 0.1
    public var activationDropout: Float = 0.1
    public var layerNormEps: Float = 1e-5
    public var useStochasticDurationPrediction: Bool = true
    public var numSpeakers: Int = 1
    public var speakerEmbeddingSize: Int = 0
    public var upsampleInitialChannel: Int = 512
    public var upsampleRates: [Int] = [8, 8, 2, 2]
    public var upsampleKernelSizes: [Int] = [16, 16, 4, 4]
    public var resblockKernelSizes: [Int] = [3, 7, 11]
    public var resblockDilationSizes: [[Int]] = [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
    public var leakyReluSlope: Float = 0.1
    public var depthSeparableChannels: Int = 2
    public var depthSeparableNumLayers: Int = 3
    public var durationPredictorFlowBins: Int = 10
    public var durationPredictorTailBound: Float = 5.0
    public var durationPredictorKernelSize: Int = 3
    public var durationPredictorDropout: Float = 0.5
    public var durationPredictorNumFlows: Int = 4
    public var durationPredictorFilterChannels: Int = 256
    public var priorEncoderNumFlows: Int = 4
    public var priorEncoderNumWavenetLayers: Int = 4
    public var posteriorEncoderNumWavenetLayers: Int = 16
    public var wavenetKernelSize: Int = 5
    public var wavenetDilationRate: Int = 1
    public var wavenetDropout: Float = 0.0
    public var speakingRate: Float = 1.0
    public var noiseScale: Float = 0.667
    public var noiseScaleDuration: Float = 0.8
    public var samplingRate: Int = 16000

    /// Samples per spectrogram frame: the decoder's upsampling, all told.
    public var hopLength: Int { upsampleRates.reduce(1, *) }

    public init() {}

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case vocabSize = "vocab_size"
        case hiddenSize = "hidden_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case windowSize = "window_size"
        case useBias = "use_bias"
        case ffnDim = "ffn_dim"
        case layerdrop
        case ffnKernelSize = "ffn_kernel_size"
        case flowSize = "flow_size"
        case spectrogramBins = "spectrogram_bins"
        case hiddenAct = "hidden_act"
        case hiddenDropout = "hidden_dropout"
        case attentionDropout = "attention_dropout"
        case activationDropout = "activation_dropout"
        case layerNormEps = "layer_norm_eps"
        case useStochasticDurationPrediction = "use_stochastic_duration_prediction"
        case numSpeakers = "num_speakers"
        case speakerEmbeddingSize = "speaker_embedding_size"
        case upsampleInitialChannel = "upsample_initial_channel"
        case upsampleRates = "upsample_rates"
        case upsampleKernelSizes = "upsample_kernel_sizes"
        case resblockKernelSizes = "resblock_kernel_sizes"
        case resblockDilationSizes = "resblock_dilation_sizes"
        case leakyReluSlope = "leaky_relu_slope"
        case depthSeparableChannels = "depth_separable_channels"
        case depthSeparableNumLayers = "depth_separable_num_layers"
        case durationPredictorFlowBins = "duration_predictor_flow_bins"
        case durationPredictorTailBound = "duration_predictor_tail_bound"
        case durationPredictorKernelSize = "duration_predictor_kernel_size"
        case durationPredictorDropout = "duration_predictor_dropout"
        case durationPredictorNumFlows = "duration_predictor_num_flows"
        case durationPredictorFilterChannels = "duration_predictor_filter_channels"
        case priorEncoderNumFlows = "prior_encoder_num_flows"
        case priorEncoderNumWavenetLayers = "prior_encoder_num_wavenet_layers"
        case posteriorEncoderNumWavenetLayers = "posterior_encoder_num_wavenet_layers"
        case wavenetKernelSize = "wavenet_kernel_size"
        case wavenetDilationRate = "wavenet_dilation_rate"
        case wavenetDropout = "wavenet_dropout"
        case speakingRate = "speaking_rate"
        case noiseScale = "noise_scale"
        case noiseScaleDuration = "noise_scale_duration"
        case samplingRate = "sampling_rate"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        func value<T: Decodable>(_ key: CodingKeys, _ fallback: T) throws -> T {
            try c.decodeIfPresent(T.self, forKey: key) ?? fallback
        }
        let d = VitsConfig()
        modelType = try value(.modelType, d.modelType)
        vocabSize = try value(.vocabSize, d.vocabSize)
        hiddenSize = try value(.hiddenSize, d.hiddenSize)
        numHiddenLayers = try value(.numHiddenLayers, d.numHiddenLayers)
        numAttentionHeads = try value(.numAttentionHeads, d.numAttentionHeads)
        windowSize = try value(.windowSize, d.windowSize)
        useBias = try value(.useBias, d.useBias)
        ffnDim = try value(.ffnDim, d.ffnDim)
        layerdrop = try value(.layerdrop, d.layerdrop)
        ffnKernelSize = try value(.ffnKernelSize, d.ffnKernelSize)
        flowSize = try value(.flowSize, d.flowSize)
        spectrogramBins = try value(.spectrogramBins, d.spectrogramBins)
        hiddenAct = try value(.hiddenAct, d.hiddenAct)
        hiddenDropout = try value(.hiddenDropout, d.hiddenDropout)
        attentionDropout = try value(.attentionDropout, d.attentionDropout)
        activationDropout = try value(.activationDropout, d.activationDropout)
        layerNormEps = try value(.layerNormEps, d.layerNormEps)
        useStochasticDurationPrediction = try value(
            .useStochasticDurationPrediction, d.useStochasticDurationPrediction)
        numSpeakers = try value(.numSpeakers, d.numSpeakers)
        speakerEmbeddingSize = try value(.speakerEmbeddingSize, d.speakerEmbeddingSize)
        upsampleInitialChannel = try value(.upsampleInitialChannel, d.upsampleInitialChannel)
        upsampleRates = try value(.upsampleRates, d.upsampleRates)
        upsampleKernelSizes = try value(.upsampleKernelSizes, d.upsampleKernelSizes)
        resblockKernelSizes = try value(.resblockKernelSizes, d.resblockKernelSizes)
        resblockDilationSizes = try value(.resblockDilationSizes, d.resblockDilationSizes)
        leakyReluSlope = try value(.leakyReluSlope, d.leakyReluSlope)
        depthSeparableChannels = try value(.depthSeparableChannels, d.depthSeparableChannels)
        depthSeparableNumLayers = try value(.depthSeparableNumLayers, d.depthSeparableNumLayers)
        durationPredictorFlowBins = try value(.durationPredictorFlowBins, d.durationPredictorFlowBins)
        durationPredictorTailBound = try value(.durationPredictorTailBound, d.durationPredictorTailBound)
        durationPredictorKernelSize = try value(
            .durationPredictorKernelSize, d.durationPredictorKernelSize)
        durationPredictorDropout = try value(.durationPredictorDropout, d.durationPredictorDropout)
        durationPredictorNumFlows = try value(.durationPredictorNumFlows, d.durationPredictorNumFlows)
        durationPredictorFilterChannels = try value(
            .durationPredictorFilterChannels, d.durationPredictorFilterChannels)
        priorEncoderNumFlows = try value(.priorEncoderNumFlows, d.priorEncoderNumFlows)
        priorEncoderNumWavenetLayers = try value(
            .priorEncoderNumWavenetLayers, d.priorEncoderNumWavenetLayers)
        posteriorEncoderNumWavenetLayers = try value(
            .posteriorEncoderNumWavenetLayers, d.posteriorEncoderNumWavenetLayers)
        wavenetKernelSize = try value(.wavenetKernelSize, d.wavenetKernelSize)
        wavenetDilationRate = try value(.wavenetDilationRate, d.wavenetDilationRate)
        wavenetDropout = try value(.wavenetDropout, d.wavenetDropout)
        speakingRate = try value(.speakingRate, d.speakingRate)
        noiseScale = try value(.noiseScale, d.noiseScale)
        noiseScaleDuration = try value(.noiseScaleDuration, d.noiseScaleDuration)
        samplingRate = try value(.samplingRate, d.samplingRate)
    }
}
