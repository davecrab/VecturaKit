import Accelerate
import Foundation
import VecturaCore

@available(macOS 14.0, iOS 17.0, tvOS 17.0, visionOS 1.0, watchOS 10.0, *)
/// A vector database implementation that stores and searches documents using pre-computed vector embeddings.
/// This implementation does not include embedding generation capabilities and thus has lower platform requirements.
public class VecturaExternalKit: VecturaProtocol {

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

    // MARK: - VecturaProtocol Implementation

    public func documentCount() -> Int {
        return documents.count
    }

    public func documentExists(id: UUID) -> Bool {
        return documents[id] != nil
    }

    public func documentsExist(ids: [UUID]) -> [UUID: Bool] {
        return ids.reduce(into: [:]) { result, id in
            result[id] = documents[id] != nil
        }
    }

    /// Adds multiple documents with pre-computed embeddings to the vector store in batch.
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
        
        // Update BM25 index
        let allDocs = Array(documents.values)
        bm25Index = BM25Index(
            documents: allDocs,
            k1: config.searchOptions.k1,
            b: config.searchOptions.b
        )
        
        // Save documents to storage
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

    // MARK: - Private Helper Methods

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