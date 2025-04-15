import Foundation

/// A protocol defining the requirements for a vector database instance.
public protocol VecturaProtocol {
    
    /// Returns the number of documents in the vector database.
    /// - Returns: The count of documents currently stored in the database.
    func documentCount() -> Int
    
    /// Checks if a document with the specified ID exists in the database.
    /// - Parameter id: The UUID of the document to check
    /// - Returns: `true` if the document exists, `false` otherwise
    func documentExists(id: UUID) -> Bool
    
    /// Checks if documents with the specified IDs exist in the database.
    /// - Parameter ids: An array of UUIDs to check
    /// - Returns: A dictionary mapping each UUID to a boolean indicating whether it exists
    func documentsExist(ids: [UUID]) -> [UUID: Bool]

    /// Adds multiple documents to the vector store in batch.
    ///
    /// - Parameters:
    ///   - texts: The text contents of the documents.
    ///   - ids: Optional unique identifiers for the documents.
    ///   - model: A ``VecturaModelSource`` specifying how to load the model.
    ///              (e.g.,`.id("sentence-transformers/all-MiniLM-L6-v2")`).
    /// - Returns: The IDs of the added documents.
    func addDocuments(
        texts: [String],
        ids: [UUID]?,
        model: VecturaModelSource
    ) async throws -> [UUID]
    
    /// Adds multiple documents with pre-computed embeddings to the vector store in batch.
    ///
    /// - Parameters:
    ///   - texts: The text contents of the documents.
    ///   - embeddings: Pre-computed embeddings for the documents.
    ///   - ids: Optional unique identifiers for the documents.
    /// - Returns: The IDs of the added documents.
    func addDocumentsWithEmbeddings(
        texts: [String],
        embeddings: [[Float]],
        ids: [UUID]?
    ) async throws -> [UUID]

    /// Searches for similar documents using a *pre-computed query embedding*.
    ///
    /// - Parameters:
    ///   - query: The query vector to search with.
    ///   - numResults: Maximum number of results to return.
    ///   - threshold: Minimum similarity threshold.
    /// - Returns: An array of search results ordered by similarity.
    func search(
        query: [Float],
        numResults: Int?,
        threshold: Float?
    ) async throws -> [VecturaSearchResult]

    /// Removes all documents from the vector store.
    func reset() async throws
}

// MARK: - Default Implementations

public extension VecturaProtocol {

    /// Adds a document to the vector store by embedding text.
    ///
    /// - Parameters:
    ///   - text: The text content of the document.
    ///   - id: Optional unique identifier for the document.
    ///   - model: A ``VecturaModelSource`` specifying how to load the model.
    ///              (e.g.,`.id("sentence-transformers/all-MiniLM-L6-v2")`).
    /// - Returns: The ID of the added document.
    func addDocument(
        text: String,
        id: UUID? = nil,
        model: VecturaModelSource = .default
    ) async throws -> UUID {
        let ids = try await addDocuments(
            texts: [text],
            ids: id.map { [$0] },
            model: model
        )
        return ids[0]
    }
    
    /// Adds a document with a pre-computed embedding to the vector store.
    ///
    /// - Parameters:
    ///   - text: The text content of the document.
    ///   - embedding: Pre-computed embedding for the document.
    ///   - id: Optional unique identifier for the document.
    /// - Returns: The ID of the added document.
    func addDocumentWithEmbedding(
        text: String,
        embedding: [Float],
        id: UUID? = nil
    ) async throws -> UUID {
        let ids = try await addDocumentsWithEmbeddings(
            texts: [text],
            embeddings: [embedding],
            ids: id.map { [$0] }
        )
        return ids[0]
    }

    /// Adds a document to the vector store by embedding text.
    ///
    /// - Parameters:
    ///   - text: The text content of the document.
    ///   - id: Optional unique identifier for the document.
    ///   - modelId: Identifier of the model to use for generating the embedding
    ///              (e.g., "sentence-transformers/all-MiniLM-L6-v2").
    /// - Returns: The ID of the added document.
    @_disfavoredOverload
    func addDocument(
        text: String,
        id: UUID?,
        modelId: String = VecturaModelSource.defaultModelId
    ) async throws -> UUID {
        try await addDocument(text: text, id: id, model: .id(modelId))
    }

    /// Adds multiple documents to the vector store in batch.
    ///
    /// - Parameters:
    ///   - texts: The text contents of the documents.
    ///   - ids: Optional unique identifiers for the documents.
    ///   - modelId: Identifier of the model to use for generating the embedding
    ///              (e.g.,`.id("sentence-transformers/all-MiniLM-L6-v2")`).
    /// - Returns: The IDs of the added documents.
    func addDocuments(
        texts: [String],
        ids: [UUID]? = nil,
        modelId: String = VecturaModelSource.defaultModelId
    ) async throws -> [UUID] {
        try await addDocuments(texts: texts, ids: ids, model: .id(modelId))
    }
}
