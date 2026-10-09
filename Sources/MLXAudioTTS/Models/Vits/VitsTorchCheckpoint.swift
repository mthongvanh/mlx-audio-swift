import Foundation
@preconcurrency import MLX

/// A PyTorch checkpoint, as `torch.save` writes one (a zip of a pickle and
/// the tensors' storages), read as far as a state dict of tensors goes,
/// without Python. Meta publishes each MMS voice's discriminator this way
/// (`facebook/mms-tts`, `full_models/<code>/D_100000.pth`).
///
/// The zip's entries must be stored, not compressed, as `torch.save`
/// stores them. The pickle runs on a small machine that knows what state
/// dicts are made of: containers, numbers, strings, `OrderedDict` and
/// tensors rebuilt from their storages. Any other object it calls for
/// becomes an opaque value, and nothing in the file is run.
struct TorchCheckpoint {
    enum ReadError: LocalizedError {
        case notZip
        case compressed(String)
        case missing(String)
        case pickle(String)

        var errorDescription: String? {
            switch self {
            case .notZip: "Not a PyTorch checkpoint: it isn't a zip."
            case .compressed(let name): "\(name) is compressed; torch.save stores its entries."
            case .missing(let name): "The checkpoint has no \(name)."
            case .pickle(let why): "The checkpoint's pickle can't be read: \(why)."
            }
        }
    }

    /// A tensor as the pickle describes it: where its values are, and how
    /// they are laid out.
    struct TensorRef {
        let storage: String
        let dtype: DType
        let offset: Int
        let shape: [Int]
        let strides: [Int]
    }

    final class DictBox {
        var items: [(key: Value, value: Value)] = []
    }

    final class ListBox {
        var items: [Value] = []
    }

    indirect enum Value {
        case none
        case bool(Bool)
        case int(Int)
        case float(Double)
        case string(String)
        case bytes(Data)
        case tuple([Value])
        case list(ListBox)
        case dict(DictBox)
        case global(String)
        case storage(key: String, dtype: DType)
        case tensor(TensorRef)
        case object(String)
        case mark

        subscript(key: String) -> Value? {
            guard case .dict(let box) = self else { return nil }
            return box.items.first { if case .string(key) = $0.key { true } else { false } }?.value
        }
    }

    let root: Value
    private let file: Data
    /// Each entry's name, and where its bytes are in [file].
    private let entries: [String: Range<Int>]
    /// The folder the entries share (`archive/`).
    private let prefix: String

    init(url: URL) throws {
        file = try Data(contentsOf: url, options: .alwaysMapped)
        entries = try Self.readZip(file)
        guard let pickle = entries.keys.first(where: { $0.hasSuffix("data.pkl") }) else {
            throw ReadError.missing("data.pkl")
        }
        prefix = String(pickle.dropLast("data.pkl".count))
        var machine = Unpickler(file[entries[pickle]!])
        root = try machine.run()
    }

    /// The tensors in the dict at [key] of the checkpoint's top-level dict
    /// (or the top-level dict itself), by name. Anything there that isn't
    /// a tensor is left out.
    static func tensors(at url: URL, under key: String? = nil) throws -> [String: MLXArray] {
        let checkpoint = try TorchCheckpoint(url: url)
        let dict = key.map { checkpoint.root[$0] } ?? checkpoint.root
        guard case .dict(let box)? = dict else {
            throw ReadError.missing(key.map { "dict \"\($0)\"" } ?? "dict at its top")
        }
        var out: [String: MLXArray] = [:]
        for (name, value) in box.items {
            if case .string(let name) = name, case .tensor(let ref) = value {
                out[name] = try checkpoint.array(ref)
            }
        }
        return out
    }

    /// The tensor's values, read from its storage. Only contiguous tensors
    /// are read: a state dict's are.
    func array(_ ref: TensorRef) throws -> MLXArray {
        guard let range = entries[prefix + "data/" + ref.storage] else {
            throw ReadError.missing("storage \(ref.storage)")
        }
        var expected = 1
        for (size, stride) in zip(ref.shape, ref.strides).reversed() {
            guard size == 1 || stride == expected else {
                throw ReadError.pickle("a tensor that isn't contiguous")
            }
            expected *= size
        }
        let size = ref.dtype.size
        let start = range.lowerBound + ref.offset * size
        let end = start + ref.shape.reduce(1, *) * size
        guard end <= range.upperBound else { throw ReadError.pickle("a tensor past its storage") }
        return MLXArray(file[start ..< end], ref.shape, dtype: ref.dtype)
    }

    // MARK: - The zip

    private static func readZip(_ data: Data) throws -> [String: Range<Int>] {
        func u16(_ at: Int) -> Int { Int(data[at]) | Int(data[at + 1]) << 8 }
        func u32(_ at: Int) -> Int { u16(at) | u16(at + 2) << 16 }
        func u64(_ at: Int) -> Int { u32(at) | u32(at + 4) << 32 }

        // The end of the central directory, searched for from the end.
        let base = data.startIndex
        guard data.count >= 22 else { throw ReadError.notZip }
        var eocd = data.count - 22
        let lowest = max(0, data.count - 22 - 65535)
        while eocd >= lowest, u32(base + eocd) != 0x0605_4b50 { eocd -= 1 }
        guard eocd >= lowest else { throw ReadError.notZip }
        var count = u16(base + eocd + 10)
        var directory = u32(base + eocd + 16)
        // Zip64, where the counts or offsets don't fit.
        if count == 0xFFFF || directory == 0xFFFF_FFFF, eocd >= 20,
           u32(base + eocd - 20) == 0x0706_4b50
        {
            let record = u64(base + eocd - 12)
            guard u32(base + record) == 0x0606_4b50 else { throw ReadError.notZip }
            count = u64(base + record + 32)
            directory = u64(base + record + 48)
        }

        var entries: [String: Range<Int>] = [:]
        var at = directory
        for _ in 0 ..< count {
            guard u32(base + at) == 0x0201_4b50 else { throw ReadError.notZip }
            let method = u16(base + at + 10)
            var size = u32(base + at + 20)
            let nameLength = u16(base + at + 28)
            let extraLength = u16(base + at + 30)
            let commentLength = u16(base + at + 32)
            var local = u32(base + at + 42)
            let name = String(decoding: data[(base + at + 46) ..< (base + at + 46 + nameLength)], as: UTF8.self)
            // Zip64's sizes and offset, in the order the plain ones overflowed.
            var extra = at + 46 + nameLength
            let extraEnd = extra + extraLength
            while extra + 4 <= extraEnd {
                let id = u16(base + extra)
                let length = u16(base + extra + 2)
                if id == 0x0001 {
                    var field = extra + 4
                    if u32(base + at + 24) == 0xFFFF_FFFF { field += 8 }  // uncompressed size
                    if size == 0xFFFF_FFFF {
                        size = u64(base + field)
                        field += 8
                    }
                    if local == 0xFFFF_FFFF { local = u64(base + field) }
                }
                extra += 4 + length
            }
            guard method == 0 else { throw ReadError.compressed(name) }
            guard u32(base + local) == 0x0403_4b50 else { throw ReadError.notZip }
            let start = local + 30 + u16(base + local + 26) + u16(base + local + 28)
            entries[name] = (base + start) ..< (base + start + size)
            at += 46 + nameLength + extraLength + commentLength
        }
        return entries
    }

    // MARK: - The pickle

    private struct Unpickler {
        let data: Data
        var at: Int
        var stack: [Value] = []
        var memo: [Int: Value] = [:]

        init(_ data: Data) {
            self.data = data
            at = data.startIndex
        }

        mutating func byte() throws -> UInt8 {
            guard at < data.endIndex else { throw ReadError.pickle("it ends early") }
            defer { at += 1 }
            return data[at]
        }

        mutating func bytes(_ n: Int) throws -> Data {
            guard n >= 0, at + n <= data.endIndex else { throw ReadError.pickle("it ends early") }
            defer { at += n }
            return data[at ..< at + n]
        }

        /// An unsigned little-endian integer of [n] bytes.
        mutating func uint(_ n: Int) throws -> Int {
            try bytes(n).reversed().reduce(0) { $0 << 8 | Int($1) }
        }

        mutating func line() throws -> String {
            guard let end = data[at...].firstIndex(of: 0x0A) else { throw ReadError.pickle("it ends early") }
            defer { at = end + 1 }
            return String(decoding: data[at ..< end], as: UTF8.self)
        }

        mutating func pop() throws -> Value {
            guard let value = stack.popLast() else { throw ReadError.pickle("its stack runs out") }
            return value
        }

        /// What was pushed since the last mark, which goes too.
        mutating func popToMark() throws -> [Value] {
            guard let mark = stack.lastIndex(where: { if case .mark = $0 { true } else { false } }) else {
                throw ReadError.pickle("no mark")
            }
            let items = Array(stack[(mark + 1)...])
            stack.removeSubrange(mark...)
            return items
        }

        mutating func setItems(_ items: [Value]) throws {
            guard case .dict(let box)? = stack.last, items.count % 2 == 0 else {
                throw ReadError.pickle("items for something that isn't a dict")
            }
            for i in stride(from: 0, to: items.count, by: 2) {
                box.items.append((items[i], items[i + 1]))
            }
        }

        mutating func append(_ items: [Value]) throws {
            guard case .list(let box)? = stack.last else {
                throw ReadError.pickle("items for something that isn't a list")
            }
            box.items += items
        }

        mutating func run() throws -> Value {
            while true {
                let op = try byte()
                switch op {
                case 0x80: _ = try byte()  // PROTO
                case 0x95: _ = try bytes(8)  // FRAME
                case 0x2E: return try pop()  // STOP
                case 0x28: stack.append(.mark)  // MARK
                case 0x4E: stack.append(.none)
                case 0x88: stack.append(.bool(true))
                case 0x89: stack.append(.bool(false))
                case 0x4B: stack.append(.int(try uint(1)))  // BININT1
                case 0x4D: stack.append(.int(try uint(2)))  // BININT2
                case 0x4A: stack.append(.int(Int(Int32(truncatingIfNeeded: try uint(4)))))  // BININT
                case 0x8A:  // LONG1
                    let n = Int(try byte())
                    let raw = try bytes(n)
                    guard n <= 8 else { throw ReadError.pickle("an integer too large") }
                    var value = raw.reversed().reduce(0) { $0 << 8 | Int($1) }
                    if n > 0, n < 8, raw.last! & 0x80 != 0 { value -= 1 << (8 * n) }
                    stack.append(.int(value))
                case 0x47:  // BINFLOAT, big-endian
                    let bits = try bytes(8).reduce(UInt64(0)) { $0 << 8 | UInt64($1) }
                    stack.append(.float(Double(bitPattern: bits)))
                case 0x58: stack.append(.string(String(decoding: try bytes(try uint(4)), as: UTF8.self)))
                case 0x8C: stack.append(.string(String(decoding: try bytes(try uint(1)), as: UTF8.self)))
                case 0x8D: stack.append(.string(String(decoding: try bytes(try uint(8)), as: UTF8.self)))
                case 0x54: stack.append(.string(String(decoding: try bytes(try uint(4)), as: UTF8.self)))
                case 0x55: stack.append(.string(String(decoding: try bytes(try uint(1)), as: UTF8.self)))
                case 0x42: stack.append(.bytes(try bytes(try uint(4))))  // BINBYTES
                case 0x43: stack.append(.bytes(try bytes(try uint(1))))  // SHORT_BINBYTES
                case 0x7D: stack.append(.dict(DictBox()))  // EMPTY_DICT
                case 0x5D: stack.append(.list(ListBox()))  // EMPTY_LIST
                case 0x29: stack.append(.tuple([]))  // EMPTY_TUPLE
                case 0x64:  // DICT
                    let items = try popToMark()
                    stack.append(.dict(DictBox()))
                    try setItems(items)
                case 0x6C:  // LIST
                    let box = ListBox()
                    box.items = try popToMark()
                    stack.append(.list(box))
                case 0x74: stack.append(.tuple(try popToMark()))  // TUPLE
                case 0x85: stack.append(.tuple([try pop()]))  // TUPLE1
                case 0x86:  // TUPLE2
                    let b = try pop()
                    stack.append(.tuple([try pop(), b]))
                case 0x87:  // TUPLE3
                    let c = try pop()
                    let b = try pop()
                    stack.append(.tuple([try pop(), b, c]))
                case 0x71: memo[try uint(1)] = stack.last  // BINPUT
                case 0x72: memo[try uint(4)] = stack.last  // LONG_BINPUT
                case 0x94: memo[memo.count] = stack.last  // MEMOIZE
                case 0x68, 0x6A:  // BINGET, LONG_BINGET
                    let key = try uint(op == 0x68 ? 1 : 4)
                    guard let value = memo[key] else { throw ReadError.pickle("memo \(key) unset") }
                    stack.append(value)
                case 0x73:  // SETITEM
                    let value = try pop()
                    let key = try pop()
                    try setItems([key, value])
                case 0x75: try setItems(try popToMark())  // SETITEMS
                case 0x61: try append([try pop()])  // APPEND
                case 0x65: try append(try popToMark())  // APPENDS
                case 0x63:  // GLOBAL
                    let module = try line()
                    stack.append(.global(module + "." + (try line())))
                case 0x93:  // STACK_GLOBAL
                    guard case .string(let name) = try pop(), case .string(let module) = try pop() else {
                        throw ReadError.pickle("a global without a name")
                    }
                    stack.append(.global(module + "." + name))
                case 0x51:  // BINPERSID: a storage
                    guard case .tuple(let id) = try pop(), id.count >= 3,
                          case .string("storage") = id[0], case .global(let type) = id[1],
                          case .string(let key) = id[2]
                    else { throw ReadError.pickle("a persistent id that isn't a storage") }
                    stack.append(.storage(key: key, dtype: try Self.dtype(type)))
                case 0x52:  // REDUCE
                    let args = try pop()
                    stack.append(try reduce(try pop(), args))
                case 0x81:  // NEWOBJ
                    _ = try pop()
                    guard case .global(let name) = try pop() else { throw ReadError.pickle("NEWOBJ of a non-class") }
                    stack.append(.object(name))
                case 0x62: _ = try pop()  // BUILD: an object's state, which nothing here needs
                default:
                    throw ReadError.pickle(String(format: "opcode 0x%02X", op))
                }
            }
        }

        func reduce(_ callable: Value, _ args: Value) throws -> Value {
            guard case .global(let name) = callable else { throw ReadError.pickle("a call of a non-global") }
            let args: [Value] = if case .tuple(let a) = args { a } else { [] }
            switch name {
            case "collections.OrderedDict":
                return .dict(DictBox())
            case "torch._utils._rebuild_tensor_v2", "torch._utils._rebuild_tensor":
                guard args.count >= 4, case .storage(let key, let dtype) = args[0], case .int(let offset) = args[1],
                      case .tuple(let size) = args[2], case .tuple(let stride) = args[3]
                else { throw ReadError.pickle("a tensor without its storage") }
                func ints(_ values: [Value]) throws -> [Int] {
                    try values.map { if case .int(let i) = $0 { i } else { throw ReadError.pickle("a shape") } }
                }
                return .tensor(TensorRef(
                    storage: key, dtype: dtype, offset: offset, shape: try ints(size), strides: try ints(stride)))
            case "torch._utils._rebuild_parameter":
                return args.first ?? .none
            default:
                return .object(name)
            }
        }

        static func dtype(_ storage: String) throws -> DType {
            switch storage.split(separator: ".").last ?? "" {
            case "FloatStorage": .float32
            case "HalfStorage": .float16
            case "BFloat16Storage": .bfloat16
            case "DoubleStorage": .float64
            case "LongStorage": .int64
            case "IntStorage": .int32
            case "ShortStorage": .int16
            case "CharStorage": .int8
            case "ByteStorage": .uint8
            case "BoolStorage": .bool
            default: throw ReadError.pickle("storage \(storage)")
            }
        }
    }
}
