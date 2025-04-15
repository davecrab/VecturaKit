import Foundation

/// A document stored in the vector database.
public struct VecturaDocument: Identifiable, Codable, Sendable {
    /// The unique identifier of the document.
    public let id: UUID
    
    /// The text content of the document.
    public let text: String
    
    /// The vector embedding of the document.
    public let embedding: [Float]
    
    /// The timestamp when the document was created.
    public let createdAt: Date

    /// Creates a new document with the given properties.
    /// - Parameters:
    ///   - id: The unique identifier for the document. If nil, a new UUID will be generated.
    ///   - text: The text content of the document.
    ///   - embedding: The vector embedding of the document.
    public init(id: UUID? = nil, text: String, embedding: [Float]) {
        self.id = id ?? UUID()
        self.text = text
        self.embedding = embedding
        self.createdAt = Date()
    }
    
    /// Calculates a hybrid score combining vector similarity and BM25 rankings.
    /// - Parameters:
    ///   - vectorScore: The vector similarity score (0.0 to 1.0).
    ///   - bm25Score: The BM25 score from text-based search.
    ///   - weight: The weight to apply to the vector score (0.0 to 1.0).
    /// - Returns: A combined hybrid score.
    public func hybridScore(vectorScore: Float, bm25Score: Float, weight: Float) -> Float {
        // Normalize BM25 score to 0-1 range
        let normalizedBM25 = bm25Score / max(bm25Score, 1.0)
        
        // Combine with weighted average
        return weight * vectorScore + (1.0 - weight) * normalizedBM25
    }

    // MARK: - Codable
    enum CodingKeys: String, CodingKey {
        case id, text, embedding, createdAt
    }
}
