import Accelerate
import CoreML
import Embeddings
import Foundation
import VecturaCore

@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
/// A vector database implementation that stores and searches documents using their vector embeddings.
public class VecturaKit: VecturaEmbeddingProtocol {

    /// The configuration for this vector database instance.
    private let config: VecturaConfig

    /// In-memory cache of all documents.
    private var documents: [UUID: VecturaDocument]

    /// The storage directory for documents.
    private let storageDirectory: URL

    /// The storage provider that handles document persistence.
    private let storageProvider: VecturaStorage

    /// Cached normalized embeddings for faster searches.
    private var normalizedEmbeddings: [UUID: [Float]] = [:]

    /// BM25 index for text search
    private var bm25Index: BM25Index?

    /// Swift-Embeddings model bundle that you can reuse (e.g. BERT, XLM-R, CLIP, etc.)
    private var bertModel: Bert.ModelBundle?

    // MARK: - Initialization

    public init(config: VecturaConfig) async throws {
        self.config = config
        self.documents = [:]

        if let customStorageDirectory = config.directoryURL {
            let databaseDirectory = customStorageDirectory.appending(path: config.name)
            if !FileManager.default.fileExists(atPath: databaseDirectory.path(percentEncoded: false)) {
                try FileManager.default.createDirectory(
                    at: databaseDirectory, withIntermediateDirectories: true)
            }
            self.storageDirectory = databaseDirectory
        } else {
            // Create default storage directory
            self.storageDirectory = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)
                .first!
                .appendingPathComponent("VecturaKit")
                .appendingPathComponent(config.name)
        }

        try FileManager.default.createDirectory(at: storageDirectory, withIntermediateDirectories: true)

        // Instantiate the storage provider (currently the file-based implementation).
        self.storageProvider = try FileStorageProvider(storageDirectory: storageDirectory)

        // Load existing documents using the storage provider.
        let storedDocuments = try await storageProvider.loadDocuments()
        for doc in storedDocuments {
            self.documents[doc.id] = doc
            // Compute normalized embedding and store in cache.
            let norm = l2Norm(doc.embedding)
            var divisor = norm + 1e-9
            var normalized = [Float](repeating: 0, count: doc.embedding.count)
            vDSP_vsdiv(doc.embedding, 1, &divisor, &normalized, 1, vDSP_Length(doc.embedding.count))
            self.normalizedEmbeddings[doc.id] = normalized
        }
    }

    /// Adds multiple documents to the vector store in batch.
    public func addDocuments(
        texts: [String],
        ids: [UUID]? = nil,
        model: VecturaModelSource = .default,
        metadatas: [[String: String]?]? = nil
    ) async throws -> [UUID] {
        if let ids = ids, ids.count != texts.count {
            throw VecturaError.invalidInput("Number of IDs must match number of texts")
        }
        if let metadatas = metadatas, metadatas.count != texts.count {
            throw VecturaError.invalidInput("Number of metadatas must match number of texts")
        }
        if bertModel == nil {
            bertModel = try await Bert.loadModelBundle(from: model)
        }
        guard let modelBundle = bertModel else {
            throw VecturaError.invalidInput("Failed to load BERT model: \(model)")
        }
        
        let embeddingsTensor = try modelBundle.batchEncode(texts)
        let shape = embeddingsTensor.shape
        if shape.count != 2 {
            throw VecturaError.invalidInput("Expected shape [N, D], got \(shape)")
        }
        if shape[1] != config.dimension {
            throw VecturaError.dimensionMismatch(
                expected: config.dimension,
                got: shape[1]
            )
        }
        let embeddingShapedArray = await embeddingsTensor.cast(to: Float.self).shapedArray(
            of: Float.self)
        let allScalars = embeddingShapedArray.scalars
        var documentIds = [UUID]()
        var documentsToSave = [VecturaDocument]()
        for i in 0..<texts.count {
            let startIndex = i * config.dimension
            let endIndex = startIndex + config.dimension
            let embeddingRow = Array(allScalars[startIndex..<endIndex])
            let docId = ids?[i] ?? UUID()
            let metadata = metadatas?[i]
            
            print("[DEBUG] AddDocuments - Raw embedding for '\(texts[i].prefix(30))...': \(embeddingRow.prefix(5))... (len: \(embeddingRow.count))")
            print("[DEBUG] AddDocuments - L2 norm: \(l2Norm(embeddingRow))")
            
            let doc = VecturaDocument(
                id: docId,
                text: texts[i],
                embedding: embeddingRow,
                metadata: metadata
            )
            documentsToSave.append(doc)
            documentIds.append(docId)
        }
        
        for doc in documentsToSave {
            let norm = l2Norm(doc.embedding)
            var divisor = norm + 1e-9
            var normalized = [Float](repeating: 0, count: doc.embedding.count)
            vDSP_vsdiv(doc.embedding, 1, &divisor, &normalized, 1, vDSP_Length(doc.embedding.count))
            normalizedEmbeddings[doc.id] = normalized
            documents[doc.id] = doc
            print("[DEBUG] AddDocuments - Stored Normalized embedding for '\(doc.text.prefix(30))...': \(normalized.prefix(5))...")
        }
        let allDocs = Array(documents.values)
        bm25Index = BM25Index(
            documents: allDocs,
            k1: config.searchOptions.k1,
            b: config.searchOptions.b
        )
        try await withThrowingTaskGroup(of: Void.self) { group in
            let directory = self.storageDirectory
            for doc in documentsToSave {
                group.addTask {
                    let documentURL = directory.appendingPathComponent("\(doc.id).json")
                    let encoder = JSONEncoder()
                    encoder.outputFormatting = .prettyPrinted
                    let data = try encoder.encode(doc)
                    try data.write(to: documentURL)
                }
            }
            try await group.waitForAll()
        }
        return documentIds
    }

    /// Adds multiple documents with pre-computed embeddings to the vector store in batch.
    ///
    /// - Parameters:
    ///   - texts: The text contents of the documents.
    ///   - embeddings: Pre-computed embeddings for the documents.
    ///   - ids: Optional unique identifiers for the documents.
    ///   - metadatas: Optional metadata for the documents.
    /// - Returns: The IDs of the added documents.
    public func addDocumentsWithEmbeddings(
        texts: [String],
        embeddings: [[Float]],
        ids: [UUID]? = nil,
        metadatas: [[String: String]?]? = nil
    ) async throws -> [UUID] {
        if let ids = ids, ids.count != texts.count {
            throw VecturaError.invalidInput("Number of IDs must match number of texts")
        }
        if let metadatas = metadatas, metadatas.count != texts.count {
            throw VecturaError.invalidInput("Number of metadatas must match number of texts")
        }
        if texts.count != embeddings.count {
            throw VecturaError.invalidInput("Number of texts must match number of embeddings")
        }
        for embedding in embeddings {
            if embedding.count != config.dimension {
                throw VecturaError.dimensionMismatch(
                    expected: config.dimension,
                    got: embedding.count
                )
            }
        }
        var documentIds = [UUID]()
        var documentsToSave = [VecturaDocument]()
        for i in 0..<texts.count {
            let docId = ids?[i] ?? UUID()
            let metadata = metadatas?[i]
            let doc = VecturaDocument(
                id: docId,
                text: texts[i],
                embedding: embeddings[i],
                metadata: metadata
            )
            documentsToSave.append(doc)
            documentIds.append(docId)
        }
        for doc in documentsToSave {
            let norm = l2Norm(doc.embedding)
            var divisor = norm + 1e-9
            var normalized = [Float](repeating: 0, count: doc.embedding.count)
            vDSP_vsdiv(doc.embedding, 1, &divisor, &normalized, 1, vDSP_Length(doc.embedding.count))
            normalizedEmbeddings[doc.id] = normalized
            documents[doc.id] = doc
        }
        let allDocs = Array(documents.values)
        bm25Index = BM25Index(
            documents: allDocs,
            k1: config.searchOptions.k1,
            b: config.searchOptions.b
        )
        try await withThrowingTaskGroup(of: Void.self) { group in
            let directory = self.storageDirectory
            for doc in documentsToSave {
                group.addTask {
                    let documentURL = directory.appendingPathComponent("\(doc.id).json")
                    let encoder = JSONEncoder()
                    encoder.outputFormatting = .prettyPrinted
                    let data = try encoder.encode(doc)
                    try data.write(to: documentURL)
                }
            }
            try await group.waitForAll()
        }
        return documentIds
    }

    /// Helper for metadata filter matching
    private func matchesMetadataFilter(_ doc: VecturaDocument, filter: [String: String]?) -> Bool {
        guard let filter = filter else { return true }
        guard let metadata = doc.metadata else { return false }
        for (key, value) in filter {
            if metadata[key] != value { return false }
        }
        return true
    }

    public func search(
        query queryEmbedding: [Float],
        numResults: Int? = nil,
        threshold: Float? = nil,
        filter: [String: String]? = nil
    ) async throws -> [VecturaSearchResult] {
        guard !queryEmbedding.isEmpty else {
            throw VecturaError.invalidInput("Query embedding cannot be empty")
        }
        
        if queryEmbedding.count != config.dimension {
            throw VecturaError.dimensionMismatch(
                expected: config.dimension,
                got: queryEmbedding.count
            )
        }

        // Normalize query embedding
        let queryNorm = l2Norm(queryEmbedding)
        var queryDivisor = queryNorm + 1e-9
        var normalizedQuery = [Float](repeating: 0, count: queryEmbedding.count)
        vDSP_vsdiv(queryEmbedding, 1, &queryDivisor, &normalizedQuery, 1, vDSP_Length(queryEmbedding.count))

        var results: [VecturaSearchResult] = []

        for doc in documents.values {
            // Apply metadata filter if provided
            if !matchesMetadataFilter(doc, filter: filter) {
                continue
            }
            
            guard let normDoc = normalizedEmbeddings[doc.id] else { continue }
            let vectorSimilarity = dotProduct(normalizedQuery, normDoc)

            // Apply threshold if specified
            if let minT = threshold ?? config.searchOptions.minThreshold, vectorSimilarity < minT {
                continue
            }

            // For pure vector search (weight = 1.0) or when BM25 index is not available
            let finalScore: Float
            if config.searchOptions.hybridWeight >= 0.999 || bm25Index == nil {
                finalScore = vectorSimilarity
            } else {
                // Hybrid search: combine vector similarity with BM25 score
                let bm25Results = bm25Index?.search(query: "", topK: documents.count) ?? []
                let bm25Score = bm25Results.first(where: { $0.document.id == doc.id })?.score ?? 0.0
                finalScore = doc.hybridScore(
                    vectorScore: vectorSimilarity,
                    bm25Score: bm25Score,
                    weight: config.searchOptions.hybridWeight
                )
            }

            results.append(
                VecturaSearchResult(
                    id: doc.id,
                    text: doc.text,
                    score: finalScore,
                    createdAt: doc.createdAt,
                    metadata: doc.metadata
                )
            )
        }

        results.sort { $0.score > $1.score }

        let limit = numResults ?? config.searchOptions.defaultNumResults
        return Array(results.prefix(limit))
    }

    public func search(
        query: String,
        numResults: Int? = nil,
        threshold: Float? = nil,
        model: VecturaModelSource = .default,
        filter: [String: String]? = nil
    ) async throws -> [VecturaSearchResult] {
        if bertModel == nil {
            bertModel = try await Bert.loadModelBundle(from: model)
        }

        guard let modelBundle = bertModel else {
            throw VecturaError.invalidInput("Failed to load BERT model: \(model)")
        }

        // Initialize BM25 index if needed
        if bm25Index == nil {
            let docs = documents.values.map { $0 }
            bm25Index = BM25Index(
                documents: docs,
                k1: config.searchOptions.k1,
                b: config.searchOptions.b
            )
        }

        // Get vector similarity results
        let queryEmbeddingTensor = try modelBundle.encode(query)
        let queryEmbeddingFloatArray = await tensorToArray(queryEmbeddingTensor)
        print("[DEBUG] Search - Raw query embedding for '\(query.prefix(30))...': \(queryEmbeddingFloatArray.prefix(5))...")
        let vectorResults = try await search(
            query: queryEmbeddingFloatArray,
            numResults: nil,
            threshold: nil,
            filter: filter
        )

        let bm25Results =
        bm25Index?.search(
            query: query,
            topK: documents.count
        ) ?? []

        // Create a map of document IDs to their BM25 scores
        let bm25Scores = Dictionary(
            bm25Results.map { ($0.document.id, $0.score) },
            uniquingKeysWith: { first, _ in first }
        )

        // Combine scores using hybrid scoring
        var hybridResults = vectorResults.map { result in
            let bm25Score = bm25Scores[result.id] ?? 0
            let hybridScore = VecturaDocument(
                id: result.id,
                text: result.text,
                embedding: [],
                metadata: result.metadata
            ).hybridScore(
                vectorScore: result.score,
                bm25Score: bm25Score,
                weight: config.searchOptions.hybridWeight
            )

            return VecturaSearchResult(
                id: result.id,
                text: result.text,
                score: hybridScore,
                createdAt: result.createdAt,
                metadata: result.metadata
            )
        }

        hybridResults.sort { $0.score > $1.score }

        if let threshold = threshold ?? config.searchOptions.minThreshold {
            hybridResults = hybridResults.filter { $0.score >= threshold }
        }

        let limit = numResults ?? config.searchOptions.defaultNumResults
        return Array(hybridResults.prefix(limit))
    }

    @_disfavoredOverload
    public func search(
        query: String,
        numResults: Int? = nil,
        threshold: Float? = nil,
        modelId: String = VecturaModelSource.defaultModelId,
        filter: [String: String]? = nil
    ) async throws -> [VecturaSearchResult] {
        try await search(
            query: query, numResults: numResults, threshold: threshold, model: .id(modelId), filter: filter)
    }

    /// Searches for documents using a pre-computed embedding from an external source.
    ///
    /// - Parameters:
    ///   - queryText: The original text query (used for hybrid search combining vector and BM25)
    ///   - queryEmbedding: The pre-computed embedding for the query text
    ///   - numResults: Optional limit on the number of results to return
    ///   - threshold: Optional minimum similarity threshold
    ///   - filter: Optional metadata filter
    /// - Returns: An array of search results
    public func searchWithExternalEmbedding(
        queryText: String,
        queryEmbedding: [Float],
        numResults: Int? = nil,
        threshold: Float? = nil,
        filter: [String: String]? = nil
    ) async throws -> [VecturaSearchResult] {
        if queryEmbedding.count != config.dimension {
            throw VecturaError.dimensionMismatch(
                expected: config.dimension,
                got: queryEmbedding.count
            )
        }
        
        // Get vector similarity results without re-computing the embedding
        let vectorResults = try await search(
            query: queryEmbedding,
            numResults: nil,
            threshold: nil,
            filter: filter
        )
        
        // Initialize BM25 index if needed
        if bm25Index == nil {
            let docs = documents.values.map { $0 }
            bm25Index = BM25Index(
                documents: docs,
                k1: config.searchOptions.k1,
                b: config.searchOptions.b
            )
        }
        
        let bm25Results = bm25Index?.search(
            query: queryText,
            topK: documents.count
        ) ?? []
        
        // Create a map of document IDs to their BM25 scores
        let bm25Scores = Dictionary(
            bm25Results.map { ($0.document.id, $0.score) },
            uniquingKeysWith: { first, _ in first }
        )
        
        // Combine scores using hybrid scoring
        var hybridResults = vectorResults.map { result in
            let bm25Score = bm25Scores[result.id] ?? 0
            let hybridScore = VecturaDocument(
                id: result.id,
                text: result.text,
                embedding: [],
                metadata: result.metadata
            ).hybridScore(
                vectorScore: result.score,
                bm25Score: bm25Score,
                weight: config.searchOptions.hybridWeight
            )
            
            return VecturaSearchResult(
                id: result.id,
                text: result.text,
                score: hybridScore,
                createdAt: result.createdAt,
                metadata: result.metadata
            )
        }
        
        hybridResults.sort { $0.score > $1.score }
        
        if let threshold = threshold ?? config.searchOptions.minThreshold {
            hybridResults = hybridResults.filter { $0.score >= threshold }
        }
        
        let limit = numResults ?? config.searchOptions.defaultNumResults
        return Array(hybridResults.prefix(limit))
    }

    /// Returns the number of documents in the vector database.
    /// - Returns: The count of documents currently stored in the database.
    public func documentCount() -> Int {
        return documents.count
    }

    /// Retrieves the stored raw and normalized embeddings for a specific document.
    /// - Parameter id: The UUID of the document.
    /// - Returns: A tuple containing the optional raw and normalized embedding vectors, or (nil, nil) if the document is not found.
    public func getDocumentEmbedding(id: UUID) -> (raw: [Float]?, normalized: [Float]?) {
        let rawEmbedding = documents[id]?.embedding
        let normalizedEmbedding = normalizedEmbeddings[id]
        return (raw: rawEmbedding, normalized: normalizedEmbedding)
    }

    /// Generates and returns both the raw and normalized embeddings for a given query string using the specified model.
    /// - Parameters:
    ///   - query: The text query to embed.
    ///   - model: The embedding model source to use.
    /// - Returns: A tuple containing the raw and normalized embedding vectors.
    /// - Throws: `VecturaError` if the model cannot be loaded or embedding fails.
    public func getQueryEmbedding(query: String, model: VecturaModelSource = .default) async throws -> (raw: [Float], normalized: [Float]) {
        if bertModel == nil {
            bertModel = try await Bert.loadModelBundle(from: model)
        }
        guard let modelBundle = bertModel else {
            throw VecturaError.invalidInput("Failed to load BERT model: \(model)")
        }
        let queryEmbeddingTensor = try modelBundle.encode(query)
        let rawEmbedding = await tensorToArray(queryEmbeddingTensor)
        
        // Normalize the query embedding
        let norm = l2Norm(rawEmbedding)
        var divisor = norm + 1e-9 // Add epsilon to prevent division by zero
        var normalizedEmbedding = [Float](repeating: 0, count: rawEmbedding.count)
        vDSP_vsdiv(rawEmbedding, 1, &divisor, &normalizedEmbedding, 1, vDSP_Length(rawEmbedding.count))
        
        return (raw: rawEmbedding, normalized: normalizedEmbedding)
    }

    /// Checks if a document with the specified ID exists in the database.
    /// - Parameter id: The UUID of the document to check
    /// - Returns: `true` if the document exists, `false` otherwise
    public func documentExists(id: UUID) -> Bool {
        return documents[id] != nil
    }
    
    /// Checks if documents with the specified IDs exist in the database.
    /// - Parameter ids: An array of UUIDs to check
    /// - Returns: A dictionary mapping each UUID to a boolean indicating whether it exists
    public func documentsExist(ids: [UUID]) -> [UUID: Bool] {
        return ids.reduce(into: [:]) { result, id in
            result[id] = documents[id] != nil
        }
    }

    public func reset() async throws {
        documents.removeAll()
        normalizedEmbeddings.removeAll()
        bm25Index = nil

        let files = try FileManager.default.contentsOfDirectory(
            at: storageDirectory, includingPropertiesForKeys: nil)
        for fileURL in files {
            try FileManager.default.removeItem(at: fileURL)
        }
    }

    public func deleteDocuments(filter: [String: String]) async throws {
        let documentsToDelete = documents.values.filter { doc in
            matchesMetadataFilter(doc, filter: filter)
        }
        
        for doc in documentsToDelete {
            documents[doc.id] = nil
            normalizedEmbeddings[doc.id] = nil
            
            let documentURL = storageDirectory.appendingPathComponent("\(doc.id).json")
            try FileManager.default.removeItem(at: documentURL)
        }
        
        // Rebuild BM25 index
        let allDocs = Array(documents.values)
        if !allDocs.isEmpty {
            bm25Index = BM25Index(
                documents: allDocs,
                k1: config.searchOptions.k1,
                b: config.searchOptions.b
            )
        } else {
            bm25Index = nil
        }
    }

    /// Delete documents by IDs (internal helper)
    private func performDeleteDocumentsByIDs(_ ids: [UUID]) async throws {
        for id in ids {
            guard let doc = documents[id] else { continue }
            documents[id] = nil
            normalizedEmbeddings[id] = nil
            let documentURL = storageDirectory.appendingPathComponent("\(doc.id).json")
            try FileManager.default.removeItem(at: documentURL)
        }
        // Rebuild BM25 index
        let allDocs = Array(documents.values)
        if !allDocs.isEmpty {
            bm25Index = BM25Index(
                documents: allDocs,
                k1: config.searchOptions.k1,
                b: config.searchOptions.b
            )
        } else {
            bm25Index = nil
        }
    }

    /// Public API: Delete documents by IDs
    public func deleteDocuments(ids: [UUID]) async throws {
        try await performDeleteDocumentsByIDs(ids)
    }

    public func updateDocument(
        id: UUID,
        newText: String,
        model: VecturaModelSource = .default
    ) async throws {
        let oldMetadata = documents[id]?.metadata
        try await performDeleteDocumentsByIDs([id])

        _ = try await addDocument(text: newText, id: id, model: model, metadata: oldMetadata)
    }

    @_disfavoredOverload
    public func updateDocument(
        id: UUID,
        newText: String,
        modelId: String = VecturaModelSource.defaultModelId,
        metadata: [String: String]? = nil
    ) async throws {
        try await updateDocument(id: id, newText: newText, model: .id(modelId))
    }

    /// Ingests a file by chunking its text and adding each chunk as a document with metadata.
    /// - Parameters:
    ///   - fileURL: The URL of the file to ingest.
    ///   - chunkSize: The maximum number of characters per chunk.
    ///   - overlap: The number of characters to overlap between chunks.
    ///   - model: The embedding model to use.
    ///   - fileID: The UUID to use as the originalFileID in metadata (or generate one if nil).
    /// - Returns: The UUIDs of the created chunk documents.
    public func ingestFileChunks(
        fileURL: URL,
        chunkSize: Int = 1000,
        overlap: Int = 100,
        model: VecturaModelSource = .default,
        fileID: UUID? = nil
    ) async throws -> [UUID] {
        // 1. Try to read file content as plain text (UTF-8)
        let fileText: String
        do {
            fileText = try String(contentsOf: fileURL, encoding: .utf8)
        } catch {
            throw VecturaError.invalidInput("File could not be read as plain text (UTF-8): \(fileURL.lastPathComponent)")
        }
        // 2. Chunk the text
        var chunks: [String] = []
        var start = 0
        let length = fileText.count
        let fileIDString = (fileID ?? UUID()).uuidString
        let textArray = Array(fileText)
        while start < length {
            let end = min(start + chunkSize, length)
            let chunk = String(textArray[start..<end])
            chunks.append(chunk)
            if end == length { break }
            start = end - overlap
            if start < 0 { start = 0 }
        }
        // 3. Prepare metadata for each chunk
        let metadatas: [[String: String]] = chunks.enumerated().map { (i, chunkText) in
            [
                "originalFileID": fileIDString,
                "chunkIndex": String(i),
                "text": chunkText,
                "type": "fileChunk"
            ]
        }
        // 4. Add all chunks as documents
        let chunkIDs = try await addDocuments(
            texts: chunks,
            ids: nil,
            model: model,
            metadatas: metadatas
        )
        return chunkIDs
    }

    // MARK: - Private

    private func tensorToArray(_ tensor: MLTensor) async -> [Float] {
        let shaped = await tensor.cast(to: Float.self).shapedArray(of: Float.self)
        return shaped.scalars
    }

    private func dotProduct(_ a: [Float], _ b: [Float]) -> Float {
        var result: Float = 0
        vDSP_dotpr(a, 1, b, 1, &result, vDSP_Length(a.count))
        return result
    }

    private func l2Norm(_ v: [Float]) -> Float {
        var sumSquares: Float = 0
        vDSP_svesq(v, 1, &sumSquares, vDSP_Length(v.count))
        return sqrt(sumSquares)
    }
}

@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
extension Bert {
    static func loadModelBundle(from source: VecturaModelSource) async throws -> Bert.ModelBundle {
        switch source {
        case .id(let modelId):
            try await loadModelBundle(from: modelId)
        case .folder(let url):
            try await loadModelBundle(from: url)
        }
    }
}
