import Foundation

/// VITS's character tokenizer, reading text as transformers' `VitsTokenizer`
/// does: one id per Unicode scalar, with a blank (id 0) between every
/// character and at both ends when the checkpoint asks for it.
///
/// transformers drops characters outside the vocabulary without a word,
/// which can quietly turn a sentence into another. This one drops them the
/// same way and says which in `prepare(_:)`.
public struct VitsTokenizer: Sendable {
    public enum TokenizerError: Error, LocalizedError {
        case romanisationNeeded
        case phonemesNeeded

        public var errorDescription: String? {
            switch self {
            case .romanisationNeeded:
                "This voice reads romanised text (uroman). Romanise the text first."
            case .phonemesNeeded:
                "This voice reads phonemes (espeak). Phonemise the text first."
            }
        }
    }

    /// The text as the voice reads it, and the characters it dropped.
    public struct Prepared: Sendable {
        public let text: String
        public let dropped: [Character]
    }

    public let vocabulary: [String: Int]
    public let addBlank: Bool
    public let normalize: Bool
    public let phonemize: Bool
    public let isUroman: Bool
    public let language: String?

    /// Every token, as scalars, in the order normalising tries them: the
    /// vocabulary file's own order, then the added tokens'.
    private let matchOrder: [[Unicode.Scalar]]
    private let addedTokens: [String: Int]

    public init(
        vocabulary: [String: Int],
        order: [String]? = nil,
        addBlank: Bool = true,
        normalize: Bool = true,
        phonemize: Bool = false,
        isUroman: Bool = false,
        language: String? = nil,
        addedTokens: [String: Int] = [:],
        addedOrder: [String]? = nil
    ) {
        self.vocabulary = vocabulary
        self.addBlank = addBlank
        self.normalize = normalize
        self.phonemize = phonemize
        self.isUroman = isUroman
        self.language = language
        self.addedTokens = addedTokens
        let words = (order ?? vocabulary.keys.sorted()) + (addedOrder ?? addedTokens.keys.sorted())
        matchOrder = words.map { Array($0.unicodeScalars) }
    }

    /// Reads `vocab.json`, `tokenizer_config.json` and `added_tokens.json`
    /// from a model's folder.
    public static func fromModelDirectory(_ directory: URL) throws -> VitsTokenizer {
        let (vocabulary, order) = try readOrderedObject(directory.appendingPathComponent("vocab.json"))
        var config: [String: Any] = [:]
        let configURL = directory.appendingPathComponent("tokenizer_config.json")
        if FileManager.default.fileExists(atPath: configURL.path) {
            config = (try JSONSerialization.jsonObject(with: Data(contentsOf: configURL)) as? [String: Any]) ?? [:]
        }
        var added: [String: Int] = [:]
        var addedOrder: [String] = []
        let addedURL = directory.appendingPathComponent("added_tokens.json")
        if FileManager.default.fileExists(atPath: addedURL.path) {
            (added, addedOrder) = try readOrderedObject(addedURL)
        }
        return VitsTokenizer(
            vocabulary: vocabulary,
            order: order,
            addBlank: config["add_blank"] as? Bool ?? true,
            normalize: config["normalize"] as? Bool ?? true,
            phonemize: config["phonemize"] as? Bool ?? false,
            isUroman: config["is_uroman"] as? Bool ?? false,
            language: config["language"] as? String,
            addedTokens: added,
            addedOrder: addedOrder
        )
    }

    /// Each token's id, matched in the vocabulary's order.
    public func encode(_ text: String) throws -> [Int] {
        let prepared = try prepare(text)
        let unknown = vocabulary["<unk>"] ?? 0
        let ids = prepared.text.unicodeScalars.map { scalar -> Int in
            let token = String(scalar)
            return vocabulary[token] ?? addedTokens[token] ?? unknown
        }
        guard addBlank else { return ids }
        var spaced = [Int](repeating: 0, count: ids.count * 2 + 1)
        for (i, id) in ids.enumerated() {
            spaced[2 * i + 1] = id
        }
        return spaced
    }

    /// The text as the voice reads it: normalised, and only its own
    /// characters if it normalises.
    public func prepare(_ text: String) throws -> Prepared {
        var text = normalize ? normalized(text) : text
        if language == "ron" {
            text = text.replacingOccurrences(of: "ț", with: "ţ")
        }
        if isUroman, text.unicodeScalars.contains(where: { !$0.isASCII }) {
            throw TokenizerError.romanisationNeeded
        }
        if phonemize {
            throw TokenizerError.phonemesNeeded
        }
        guard normalize else { return Prepared(text: text, dropped: []) }

        var kept = String.UnicodeScalarView()
        var dropped = Set<Unicode.Scalar>()
        for scalar in text.unicodeScalars {
            if vocabulary[String(scalar)] != nil {
                kept.append(scalar)
            } else {
                dropped.insert(scalar)
            }
        }
        let trimmed = String(kept).trimmingCharacters(in: .whitespacesAndNewlines)
        return Prepared(
            text: trimmed,
            dropped: dropped.sorted { $0.value < $1.value }.map { Character($0) }
        )
    }

    /// Tokens kept as they are where they match, in order; any other
    /// character lower-cased.
    func normalized(_ text: String) -> String {
        let scalars = Array(text.unicodeScalars)
        var out = String.UnicodeScalarView()
        var i = 0
        outer: while i < scalars.count {
            for word in matchOrder where !word.isEmpty && i + word.count <= scalars.count {
                if scalars[i ..< i + word.count].elementsEqual(word) {
                    out.append(contentsOf: word)
                    i += word.count
                    continue outer
                }
            }
            out.append(contentsOf: String(scalars[i]).lowercased().unicodeScalars)
            i += 1
        }
        return String(out)
    }

    /// A flat JSON object of string keys and integer values, with its keys
    /// in the file's order, which JSONSerialization does not keep.
    static func readOrderedObject(_ url: URL) throws -> ([String: Int], [String]) {
        let data = try Data(contentsOf: url)
        guard let object = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return ([:], [])
        }
        var values: [String: Int] = [:]
        for (key, value) in object {
            if let number = value as? NSNumber { values[key] = number.intValue }
        }
        let source = String(decoding: data, as: UTF8.self)
        var order: [String] = []
        var seen = Set<String>()
        let pattern = try NSRegularExpression(pattern: #""((?:[^"\\]|\\.)*)"\s*:"#)
        for match in pattern.matches(in: source, range: NSRange(source.startIndex..., in: source)) {
            guard let range = Range(match.range(at: 1), in: source),
                  let key = try? JSONDecoder().decode(String.self, from: Data("\"\(source[range])\"".utf8)),
                  values[key] != nil, seen.insert(key).inserted
            else { continue }
            order.append(key)
        }
        // Anything the scan missed keeps a stable place at the end.
        order += values.keys.filter { !seen.contains($0) }.sorted()
        return (values, order)
    }
}
