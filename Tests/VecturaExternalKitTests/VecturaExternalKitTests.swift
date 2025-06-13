import XCTest
@testable import VecturaExternalKit
@testable import VecturaCore

@available(macOS 14.0, iOS 17.0, tvOS 17.0, visionOS 1.0, watchOS 10.0, *)
final class VecturaExternalKitTests: XCTestCase {
    
    func testAddDocumentWithEmbedding() async throws {
        let config = VecturaConfig(name: "test-external", dimension: 384)
        let vectorDB = try await VecturaExternalKit(config: config)
        
        let embedding = (0..<384).map { _ in Float.random(in: -1...1) }
        let docId = try await vectorDB.addDocumentWithEmbedding(
            text: "Test document",
            embedding: embedding,
            metadata: ["category": "test"]
        )
        
        XCTAssertNotNil(docId)
        XCTAssertEqual(vectorDB.documentCount(), 1)
        XCTAssertTrue(vectorDB.documentExists(id: docId))
        
        try await vectorDB.reset()
    }
    
    func testBatchAddDocumentsWithEmbeddings() async throws {
        let config = VecturaConfig(name: "test-external-batch", dimension: 384)
        let vectorDB = try await VecturaExternalKit(config: config)
        
        let texts = ["Document 1", "Document 2", "Document 3"]
        let embeddings = (0..<3).map { _ in
            (0..<384).map { _ in Float.random(in: -1...1) }
        }
        let metadatas: [[String: String]?] = [
            ["category": "test1"],
            ["category": "test2"],
            nil
        ]
        
        let docIds = try await vectorDB.addDocumentsWithEmbeddings(
            texts: texts,
            embeddings: embeddings,
            ids: nil,
            metadatas: metadatas
        )
        
        XCTAssertEqual(docIds.count, 3)
        XCTAssertEqual(vectorDB.documentCount(), 3)
        
        for docId in docIds {
            XCTAssertTrue(vectorDB.documentExists(id: docId))
        }
        
        try await vectorDB.reset()
    }
    
    func testSearchWithEmbedding() async throws {
        let config = VecturaConfig(name: "test-external-search", dimension: 384)
        let vectorDB = try await VecturaExternalKit(config: config)
        
        // Add some documents
        let texts = ["Machine learning", "Artificial intelligence", "Data science"]
        let embeddings = [
            (0..<384).map { _ in Float.random(in: 0.1...0.3) }, // Similar embeddings
            (0..<384).map { _ in Float.random(in: 0.1...0.3) },
            (0..<384).map { _ in Float.random(in: 0.7...0.9) }  // Different embedding
        ]
        
        _ = try await vectorDB.addDocumentsWithEmbeddings(
            texts: texts,
            embeddings: embeddings,
            ids: nil,
            metadatas: nil
        )
        
        // Search with a query embedding similar to the first two
        let queryEmbedding = (0..<384).map { _ in Float.random(in: 0.1...0.3) }
        let results = try await vectorDB.search(
            query: queryEmbedding,
            numResults: 2,
            threshold: nil,
            filter: nil
        )
        
        XCTAssertEqual(results.count, 2)
        XCTAssertTrue(results[0].score >= results[1].score) // Results should be sorted by score
        
        try await vectorDB.reset()
    }
    
    func testMetadataFilter() async throws {
        let config = VecturaConfig(name: "test-external-filter", dimension: 384)
        let vectorDB = try await VecturaExternalKit(config: config)
        
        let texts = ["Doc 1", "Doc 2", "Doc 3"]
        let embeddings = (0..<3).map { _ in
            (0..<384).map { _ in Float.random(in: -1...1) }
        }
        let metadatas: [[String: String]?] = [
            ["category": "A", "source": "test"],
            ["category": "B", "source": "test"],
            ["category": "A", "source": "prod"]
        ]
        
        _ = try await vectorDB.addDocumentsWithEmbeddings(
            texts: texts,
            embeddings: embeddings,
            ids: nil,
            metadatas: metadatas
        )
        
        // Search with filter
        let queryEmbedding = (0..<384).map { _ in Float.random(in: -1...1) }
        let results = try await vectorDB.search(
            query: queryEmbedding,
            numResults: 10,
            threshold: nil,
            filter: ["category": "A"]
        )
        
        XCTAssertEqual(results.count, 2)
        for result in results {
            XCTAssertEqual(result.metadata?["category"], "A")
        }
        
        try await vectorDB.reset()
    }
    
    func testDeleteDocuments() async throws {
        let config = VecturaConfig(name: "test-external-delete", dimension: 384)
        let vectorDB = try await VecturaExternalKit(config: config)
        
        let texts = ["Doc 1", "Doc 2", "Doc 3"]
        let embeddings = (0..<3).map { _ in
            (0..<384).map { _ in Float.random(in: -1...1) }
        }
        let metadatas: [[String: String]?] = [
            ["category": "delete"],
            ["category": "keep"],
            ["category": "delete"]
        ]
        
        _ = try await vectorDB.addDocumentsWithEmbeddings(
            texts: texts,
            embeddings: embeddings,
            ids: nil,
            metadatas: metadatas
        )
        
        XCTAssertEqual(vectorDB.documentCount(), 3)
        
        // Delete documents with category "delete"
        try await vectorDB.deleteDocuments(filter: ["category": "delete"])
        
        XCTAssertEqual(vectorDB.documentCount(), 1)
        
        // Verify only the "keep" document remains
        let queryEmbedding = (0..<384).map { _ in Float.random(in: -1...1) }
        let results = try await vectorDB.search(
            query: queryEmbedding,
            numResults: 10,
            threshold: nil,
            filter: nil
        )
        
        XCTAssertEqual(results.count, 1)
        XCTAssertEqual(results[0].metadata?["category"], "keep")
        
        try await vectorDB.reset()
    }
    
    func testDimensionMismatch() async throws {
        let config = VecturaConfig(name: "test-external-dimension", dimension: 384)
        let vectorDB = try await VecturaExternalKit(config: config)
        
        // Try to add document with wrong dimension
        let wrongEmbedding = (0..<256).map { _ in Float.random(in: -1...1) } // Wrong dimension
        
        do {
            _ = try await vectorDB.addDocumentWithEmbedding(
                text: "Test document",
                embedding: wrongEmbedding
            )
            XCTFail("Should have thrown dimension mismatch error")
        } catch VecturaError.dimensionMismatch(let expected, let got) {
            XCTAssertEqual(expected, 384)
            XCTAssertEqual(got, 256)
        }
        
        try await vectorDB.reset()
    }
} 