import Accelerate
import Foundation
import MLXEmbedders
import VecturaCore

@available(macOS 14.0, iOS 17.0, tvOS 17.0, visionOS 1.0, watchOS 10.0, *)
public class VecturaMLXKit: VecturaEmbeddingProtocol {
    private let config: VecturaConfig
    private let embedder: MLXEmbedder
    private var documents: [UUID: VecturaDocument] = [:]
    private var normalizedEmbeddings: [UUID: [Float]] = [:]
    private let storageDirectory: URL
    private var bm25Index: BM25Index?
    
    public init(config: VecturaConfig, modelConfiguration: ModelConfiguration = .nomic_text_v1_5)
    async throws
    {
        self.config = config
        self.embedder = try await MLXEmbedder(configuration: modelConfiguration)
        
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
        
        // Attempt to load existing docs
        try loadDocuments()
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
        
        let embeddings = await embedder.embed(texts: texts)
        var documentIds = [UUID]()
        var documentsToSave = [VecturaDocument]()
        
        for (index, text) in texts.enumerated() {
            let docId = ids?[index] ?? UUID()
            let metadata = metadatas?[index]
            let doc = VecturaDocument(id: docId, text: text, embedding: embeddings[index], metadata: metadata)
            
            // Normalize embedding for cosine similarity
            let norm = l2Norm(doc.embedding)
            var divisor = norm + 1e-9
            var normalized = [Float](repeating: 0, count: doc.embedding.count)
            vDSP_vsdiv(doc.embedding, 1, &divisor, &normalized, 1, vDSP_Length(doc.embedding.count))
            
            normalizedEmbeddings[doc.id] = normalized
            documents[doc.id] = doc
            documentIds.append(docId)
            documentsToSave.append(doc)
        }
        
        // Update BM25 index
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
        
        let norm = l2Norm(queryEmbedding)
        var divisorQuery = norm + 1e-9
        var normalizedQuery = [Float](repeating: 0, count: queryEmbedding.count)
        vDSP_vsdiv(
            queryEmbedding, 1, &divisorQuery, &normalizedQuery, 1, vDSP_Length(queryEmbedding.count))
        
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
    
    // Additional convenience methods for text-based search
    public func search(query: String, numResults: Int? = nil, threshold: Float? = nil, filter: [String: String]? = nil) async throws
    -> [VecturaSearchResult]
    {
        guard !query.isEmpty else {
            throw VecturaError.invalidInput("Query cannot be empty")
        }
        
        let queryEmbedding = try await embedder.embed(text: query)
        return try await search(query: queryEmbedding, numResults: numResults, threshold: threshold, filter: filter)
    }
    
    // MARK: - Deletion & Update APIs
    
    /// Internal helper that removes documents by their IDs and rebuilds the BM25 index.
    private func performDeleteDocumentsByIDs(_ ids: [UUID]) async throws {
        for id in ids {
            guard let doc = documents[id] else { continue }
            documents[id] = nil
            normalizedEmbeddings[id] = nil
            let documentURL = storageDirectory.appendingPathComponent("\(doc.id).json")
            try FileManager.default.removeItem(at: documentURL)
        }
        // Rebuild BM25 index
        let remainingDocs = Array(documents.values)
        if !remainingDocs.isEmpty {
            bm25Index = BM25Index(
                documents: remainingDocs,
                k1: config.searchOptions.k1,
                b: config.searchOptions.b
            )
        } else {
            bm25Index = nil
        }
    }
    
    /// Public API – delete documents by IDs.
    public func deleteDocuments(ids: [UUID]) async throws {
        try await performDeleteDocumentsByIDs(ids)
    }
    
    /// Convenience: delete by metadata filter (already implemented above)
    /// updateDocument – remove existing doc and re-add with new content
    public func updateDocument(id: UUID, newText: String) async throws {
        let oldMetadata = documents[id]?.metadata
        try await performDeleteDocumentsByIDs([id])
        _ = try await addDocument(text: newText, id: id, metadata: oldMetadata)
    }
    
    // MARK: - Private
    
    private func loadDocuments() throws {
        let fileURLs = try FileManager.default.contentsOfDirectory(
            at: storageDirectory, includingPropertiesForKeys: nil)
        
        let decoder = JSONDecoder()
        var loadErrors: [String] = []
        
        for fileURL in fileURLs where fileURL.pathExtension == "json" {
            do {
                let data = try Data(contentsOf: fileURL)
                let doc = try decoder.decode(VecturaDocument.self, from: data)
                
                // Rebuild normalized embeddings
                let norm = l2Norm(doc.embedding)
                var divisor = norm + 1e-9
                var normalized = [Float](repeating: 0, count: doc.embedding.count)
                vDSP_vsdiv(doc.embedding, 1, &divisor, &normalized, 1, vDSP_Length(doc.embedding.count))
                normalizedEmbeddings[doc.id] = normalized
                documents[doc.id] = doc
            } catch {
                loadErrors.append(
                    "Failed to load \(fileURL.lastPathComponent): \(error.localizedDescription)")
            }
        }
        
        if !loadErrors.isEmpty {
            throw VecturaError.loadFailed(loadErrors.joined(separator: "\n"))
        }
        
        // Build BM25 index if we have documents
        if !documents.isEmpty {
            let allDocs = Array(documents.values)
            bm25Index = BM25Index(
                documents: allDocs,
                k1: config.searchOptions.k1,
                b: config.searchOptions.b
            )
        }
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