import Foundation
@preconcurrency import MLX
import MLXNN

// Rational-quadratic spline flows for VITS's stochastic duration predictor,
// ported from transformers' `_unconstrained_rational_quadratic_spline`.
// Where transformers indexes only the elements inside the interval, this
// computes every element and picks with `which`. Inputs are clipped into
// the interval first, so the unpicked branch stays finite and its gradient
// can't turn into NaN.

enum VitsSpline {
    static func unconstrained(
        _ inputs: MLXArray,
        widths unnormalizedWidths: MLXArray,
        heights unnormalizedHeights: MLXArray,
        derivatives unnormalizedDerivatives: MLXArray,
        reverse: Bool,
        tailBound: Float,
        minBinWidth: Float = 1e-3,
        minBinHeight: Float = 1e-3,
        minDerivative: Float = 1e-3
    ) -> (outputs: MLXArray, logAbsDet: MLXArray) {
        let inside = logicalAnd(inputs .>= -tailBound, inputs .<= tailBound)

        // The derivative at each end is fixed so the spline meets the
        // identity outside the interval with slope 1.
        let constant = Float(log(exp(1 - Double(minDerivative)) - 1))
        var edgeShape = unnormalizedDerivatives.shape
        edgeShape[edgeShape.count - 1] = 1
        let edge = MLXArray.ones(edgeShape) * constant
        let derivatives = concatenated([edge, unnormalizedDerivatives, edge], axis: -1)

        let (outputs, logAbsDet) = rationalQuadratic(
            clip(inputs, min: -tailBound, max: tailBound),
            widths: unnormalizedWidths,
            heights: unnormalizedHeights,
            derivatives: derivatives,
            reverse: reverse,
            tailBound: tailBound,
            minBinWidth: minBinWidth,
            minBinHeight: minBinHeight,
            minDerivative: minDerivative
        )
        return (
            which(inside, outputs, inputs),
            which(inside, logAbsDet, MLXArray(Float(0)))
        )
    }

    /// Knot positions from unnormalised sizes, both ends pinned exactly as
    /// transformers pins them, and the bins' sizes.
    private static func knots(
        _ unnormalized: MLXArray, lower: Float, upper: Float, minSize: Float
    ) -> (cumulative: MLXArray, sizes: MLXArray) {
        let numBins = unnormalized.dim(-1)
        var sizes = softmax(unnormalized, axis: -1)
        sizes = minSize + (1 - minSize * Float(numBins)) * sizes
        var cumulative = cumsum(sizes, axis: -1)
        cumulative = (upper - lower) * cumulative + lower
        var endShape = cumulative.shape
        endShape[endShape.count - 1] = 1
        cumulative = concatenated(
            [
                MLXArray.ones(endShape) * lower,
                cumulative[.ellipsis, ..<(numBins - 1)],
                MLXArray.ones(endShape) * upper,
            ],
            axis: -1
        )
        let widths = cumulative[.ellipsis, 1...] - cumulative[.ellipsis, ..<numBins]
        return (cumulative, widths)
    }

    private static func take(_ params: MLXArray, _ index: MLXArray) -> MLXArray {
        takeAlong(params, index.expandedDimensions(axis: -1), axis: -1).squeezed(axis: -1)
    }

    private static func rationalQuadratic(
        _ inputs: MLXArray,
        widths unnormalizedWidths: MLXArray,
        heights unnormalizedHeights: MLXArray,
        derivatives unnormalizedDerivatives: MLXArray,
        reverse: Bool,
        tailBound: Float,
        minBinWidth: Float,
        minBinHeight: Float,
        minDerivative: Float
    ) -> (MLXArray, MLXArray) {
        let lower = -tailBound
        let upper = tailBound

        let (cumwidths, widths) = knots(unnormalizedWidths, lower: lower, upper: upper, minSize: minBinWidth)
        let (cumheights, heights) = knots(unnormalizedHeights, lower: lower, upper: upper, minSize: minBinHeight)
        let derivatives = minDerivative + softplus(unnormalizedDerivatives)

        // Nudge the last knot so an input exactly at the upper bound falls
        // in the last bin.
        var binLocations = reverse ? cumheights : cumwidths
        let count = binLocations.dim(-1)
        var nudgeShape = binLocations.shape
        nudgeShape[nudgeShape.count - 1] = count - 1
        var lastShape = binLocations.shape
        lastShape[lastShape.count - 1] = 1
        let nudge = concatenated(
            [MLXArray.zeros(nudgeShape), MLXArray.ones(lastShape) * Float(1e-6)], axis: -1)
        binLocations = stopGradient(binLocations + nudge)
        var binIndex = (inputs.expandedDimensions(axis: -1) .>= binLocations)
            .asType(.int32).sum(axis: -1) - 1
        binIndex = stopGradient(clip(binIndex, min: 0, max: widths.dim(-1) - 1))

        let inputCumwidths = take(cumwidths, binIndex)
        let inputBinWidths = take(widths, binIndex)
        let inputCumheights = take(cumheights, binIndex)
        let delta = heights / widths
        let inputDelta = take(delta, binIndex)
        let inputDerivatives = take(derivatives, binIndex)
        let inputDerivativesPlusOne = take(derivatives[.ellipsis, 1...], binIndex)
        let inputHeights = take(heights, binIndex)

        let intermediate1 = inputDerivatives + inputDerivativesPlusOne - 2 * inputDelta
        if !reverse {
            let theta = (inputs - inputCumwidths) / inputBinWidths
            let thetaOneMinusTheta = theta * (1 - theta)
            let numerator = inputHeights * (inputDelta * theta.square() + inputDerivatives * thetaOneMinusTheta)
            let denominator = inputDelta + intermediate1 * thetaOneMinusTheta
            let outputs = inputCumheights + numerator / denominator
            let derivativeNumerator = inputDelta.square() * (
                inputDerivativesPlusOne * theta.square()
                    + 2 * inputDelta * thetaOneMinusTheta
                    + inputDerivatives * (1 - theta).square()
            )
            return (outputs, log(derivativeNumerator) - 2 * log(denominator))
        }

        let intermediate2 = inputs - inputCumheights
        let intermediate3 = intermediate2 * intermediate1
        let a = inputHeights * (inputDelta - inputDerivatives) + intermediate3
        let b = inputHeights * inputDerivatives - intermediate3
        let c = -inputDelta * intermediate2
        let discriminant = maximum(b.square() - 4 * a * c, MLXArray(Float(0)))
        let root = (2 * c) / (-b - sqrt(discriminant))
        let outputs = root * inputBinWidths + inputCumwidths
        let thetaOneMinusTheta = root * (1 - root)
        let denominator = inputDelta + intermediate1 * thetaOneMinusTheta
        let derivativeNumerator = inputDelta.square() * (
            inputDerivativesPlusOne * root.square()
                + 2 * inputDelta * thetaOneMinusTheta
                + inputDerivatives * (1 - root).square()
        )
        return (outputs, -(log(derivativeNumerator) - 2 * log(denominator)))
    }
}
