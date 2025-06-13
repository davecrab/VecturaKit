import Foundation

/// A search result from a vector similarity search.
public struct VecturaSearchResult: Codable, Sendable {
    /// The unique identifier of the document.
    public let id: UUID
    
    /// The text content of the document.
    public let text: String
    
    /// The similarity score (0.0 to 1.0).
    public let score: Float
    
    /// The timestamp when the document was created.
    public let createdAt: Date

    /// Optional metadata dictionary for storing arbitrary key-value pairs.
    public let metadata: [String: String]?

    /// Creates a new search result.
    public init(id: UUID, text: String, score: Float, createdAt: Date, metadata: [String: String]? = nil) {
        self.id = id
        self.text = text
        self.score = score
        self.createdAt = createdAt
        self.metadata = metadata
    }
} 