import Accelerate
import Foundation
import MLXEmbedders
import VecturaKit

@available(macOS 14.0, iOS 17.0, tvOS 17.0, visionOS 1.0, watchOS 10.0, *)
public class VecturaMLXKit {
    private let config: VecturaConfig
    private let embedder: MLXEmbedder
    private var documents: [UUID: VecturaDocument] = [:]
    private var normalizedEmbeddings: [UUID: [Float]] = [:]
    private let storageDirectory: URL
    
    // Configuration for memory optimization
    private let maxBatchSize: Int = 16
    
    public init(config: VecturaConfig, 
                modelConfiguration: ModelConfiguration = .nomic_text_v1_5,
                maxBatchSize: Int = 16,
                maxTokenLength: Int = 512) async throws
    {
        self.config = config
        self.embedder = try await MLXEmbedder(
            configuration: modelConfiguration,
            maxBatchSize: maxBatchSize,
            defaultMaxLength: maxTokenLength
        )
        
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
    
    public func addDocuments(texts: [String], ids: [UUID]? = nil) async throws -> [UUID] {
        if let ids = ids, ids.count != texts.count {
            throw VecturaError.invalidInput("Number of IDs must match number of texts")
        }
        
        // Pre-create the document IDs
        let documentIds = ids ?? texts.map { _ in UUID() }
        
        // For large batches, process in smaller chunks to reduce memory pressure
        if texts.count > maxBatchSize {
            var processedIds: [UUID] = []
            
            for i in stride(from: 0, to: texts.count, by: maxBatchSize) {
                let endIdx = min(i + maxBatchSize, texts.count)
                let batchTexts = Array(texts[i..<endIdx])
                let batchIds = Array(documentIds[i..<endIdx])
                
                let batchResultIds = try await addDocumentsBatch(texts: batchTexts, ids: batchIds)
                processedIds.append(contentsOf: batchResultIds)
                
                // Allow for temporary data to be cleaned up between batches
                await Task.yield()
            }
            
            return processedIds
        } else {
            return try await addDocumentsBatch(texts: texts, ids: documentIds)
        }
    }
    
    private func addDocumentsBatch(texts: [String], ids: [UUID]) async throws -> [UUID] {
        let embeddings = await embedder.embed(texts: texts)
        var documentsToSave = [VecturaDocument]()
        documentsToSave.reserveCapacity(texts.count)
        
        // Preallocate the normalized array to avoid repeated allocations
        let embeddingDimension = embeddings.first?.count ?? config.dimension
        var normalizedBuffer = [Float](repeating: 0, count: embeddingDimension)
        
        for (index, text) in texts.enumerated() {
            let docId = ids[index]
            let doc = VecturaDocument(id: docId, text: text, embedding: embeddings[index])
            
            // Normalize embedding for cosine similarity (reusing buffer)
            let norm = l2Norm(doc.embedding)
            var divisor = norm + 1e-9
            vDSP_vsdiv(doc.embedding, 1, &divisor, &normalizedBuffer, 1, vDSP_Length(doc.embedding.count))
            
            // Create a copy of the normalized buffer for storage
            normalizedEmbeddings[doc.id] = Array(normalizedBuffer)
            documents[doc.id] = doc
            documentsToSave.append(doc)
        }
        
        // Perform file operations in parallel with a reasonable chunk size
        try await withThrowingTaskGroup(of: Void.self) { group in
            let directory = self.storageDirectory
            let encoder = JSONEncoder()
            encoder.outputFormatting = .prettyPrinted
            
            for doc in documentsToSave {
                group.addTask {
                    let documentURL = directory.appendingPathComponent("\(doc.id).json")
                    let data = try encoder.encode(doc)
                    try data.write(to: documentURL)
                }
            }
            
            try await group.waitForAll()
        }
        
        return ids
    }
    
    public func search(query: String, numResults: Int? = nil, threshold: Float? = nil) async throws
    -> [VecturaSearchResult]
    {
        guard !query.isEmpty else {
            throw VecturaError.invalidInput("Query cannot be empty")
        }
        
        let queryEmbedding = try await embedder.embed(text: query)
        
        // Reuse buffer for normalized query
        let norm = l2Norm(queryEmbedding)
        var divisorQuery = norm + 1e-9
        var normalizedQuery = [Float](repeating: 0, count: queryEmbedding.count)
        vDSP_vsdiv(
            queryEmbedding, 1, &divisorQuery, &normalizedQuery, 1, vDSP_Length(queryEmbedding.count))
        
        // Optimize for large document collections
        var results: [VecturaSearchResult] = []
        results.reserveCapacity(min(documents.count, numResults ?? config.searchOptions.defaultNumResults))
        
        let minThreshold = threshold ?? config.searchOptions.minThreshold ?? 0
        
        // Use Accelerate framework for batch processing if appropriate
        // For now, process each document individually but with optimizations
        for doc in documents.values {
            guard let normDoc = normalizedEmbeddings[doc.id] else { continue }
            let similarity = dotProduct(normalizedQuery, normDoc)
            
            if similarity < minThreshold {
                continue
            }
            
            results.append(
                VecturaSearchResult(
                    id: doc.id,
                    text: doc.text,
                    score: similarity,
                    createdAt: doc.createdAt
                )
            )
        }
        
        results.sort { $0.score > $1.score }
        
        let limit = numResults ?? config.searchOptions.defaultNumResults
        return results.count <= limit ? results : Array(results.prefix(limit))
    }
    
    public func deleteDocuments(ids: [UUID]) async throws {
        for id in ids {
            documents[id] = nil
            normalizedEmbeddings[id] = nil
            
            let documentURL = storageDirectory.appendingPathComponent("\(id).json")
            try FileManager.default.removeItem(at: documentURL)
        }
    }
    
    public func updateDocument(id: UUID, newText: String) async throws {
        try await deleteDocuments(ids: [id])
        _ = try await addDocuments(texts: [newText], ids: [id])
    }
    
    public func reset() async throws {
        documents.removeAll()
        normalizedEmbeddings.removeAll()
        
        let files = try FileManager.default.contentsOfDirectory(
            at: storageDirectory, includingPropertiesForKeys: nil)
        for fileURL in files {
            try FileManager.default.removeItem(at: fileURL)
        }
    }
    
    // MARK: - Private
    
    private func loadDocuments() throws {
        let fileURLs = try FileManager.default.contentsOfDirectory(
            at: storageDirectory, includingPropertiesForKeys: nil)
        
        let decoder = JSONDecoder()
        var loadErrors: [String] = []
        
        // Pre-allocate the normalized buffer
        var normalizedBuffer: [Float]? = nil
        
        for fileURL in fileURLs where fileURL.pathExtension == "json" {
            do {
                let data = try Data(contentsOf: fileURL)
                let doc = try decoder.decode(VecturaDocument.self, from: data)
                
                // Initialize buffer lazily with the correct size
                if normalizedBuffer == nil || normalizedBuffer!.count != doc.embedding.count {
                    normalizedBuffer = [Float](repeating: 0, count: doc.embedding.count)
                }
                
                // Rebuild normalized embeddings using the pre-allocated buffer
                let norm = l2Norm(doc.embedding)
                var divisor = norm + 1e-9
                vDSP_vsdiv(doc.embedding, 1, &divisor, &normalizedBuffer!, 1, vDSP_Length(doc.embedding.count))
                
                // Store a copy of the normalized embedding
                normalizedEmbeddings[doc.id] = Array(normalizedBuffer!)
                documents[doc.id] = doc
            } catch {
                loadErrors.append(
                    "Failed to load \(fileURL.lastPathComponent): \(error.localizedDescription)")
            }
        }
        
        if !loadErrors.isEmpty {
            throw VecturaError.loadFailed(loadErrors.joined(separator: "\n"))
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
