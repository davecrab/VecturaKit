import Foundation

/// Represents a search result from the vector database.
public struct VecturaSearchResult: Identifiable, Sendable {

    /// The unique identifier of the matching document.
    public let id: UUID
    
    /// The text content of the matching document.
    public let text: String
    
    /// The similarity score between the query and the document.
    public let score: Float
    
    /// The timestamp when the document was created.
    public let createdAt: Date

    /// Optional metadata dictionary for the matching document.
    public let metadata: [String: String]?
    
    /// Creates a new search result with the given properties.
    ///
    /// - Parameters:
    ///   - id: The unique identifier of the matching document.
    ///   - text: The text content of the matching document.
    ///   - score: The similarity score between the query and the document.
    ///   - createdAt: The timestamp when the document was created.
    ///   - metadata: Optional metadata dictionary.
    public init(id: UUID, text: String, score: Float, createdAt: Date, metadata: [String: String]? = nil) {
        self.id = id
        self.text = text
        self.score = score
        self.createdAt = createdAt
        self.metadata = metadata
    }
}
