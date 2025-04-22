import XCTest

@testable import VecturaKit
import Embeddings

@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, watchOS 11.0, *)
final class VecturaKitTests: XCTestCase {
    var vectura: VecturaKit!
    var config: VecturaConfig!
    
    // Add properties for vector-only testing
    var vectorOnlyVectura: VecturaKit!
    var vectorOnlyConfig: VecturaConfig!

    override func setUp() async throws {
        let searchOptions = VecturaConfig.SearchOptions(hybridWeight: 1.0)
        config = VecturaConfig(name: "test-db", dimension: 384, searchOptions: searchOptions)
        vectura = try await VecturaKit(config: config)
    }
    
    // Add setup function for vector-only tests
    func setupVectorOnlyTest() async throws {
        vectorOnlyConfig = VecturaConfig(name: "VectorOnlyTestDB", dimension: 384, searchOptions: .init(hybridWeight: 1.0)) // Set weight to 1.0
        vectorOnlyVectura = try await VecturaKit(config: vectorOnlyConfig)
        try await vectorOnlyVectura.reset()
    }

    override func tearDown() async throws {
        try await vectura.reset()
        vectura = nil
    }
    
    func testAddAndSearchDocument() async throws {
        let text = "This is a test document"
        let id = try await vectura.addDocument(text: text)
        
        let results = try await vectura.search(query: "test document")
        XCTAssertEqual(results.count, 1)
        XCTAssertEqual(results[0].id, id)
        XCTAssertEqual(results[0].text, text)
    }
    
    func testAddMultipleDocuments() async throws {
        let documents = [
            "The quick brown fox jumps over the lazy dog",
            "Pack my box with five dozen liquor jugs",
            "How vexingly quick daft zebras jump",
        ]
        
        let ids = try await vectura.addDocuments(texts: documents)
        XCTAssertEqual(ids.count, 3)
        
        let results = try await vectura.search(query: "quick jumping animals")
        XCTAssertGreaterThanOrEqual(results.count, 2)
        XCTAssertTrue(results[0].score > results[1].score)
    }
    
    func testPersistence() async throws {
        // Add documents
        let texts = ["Document 1", "Document 2"]
        let ids = try await vectura.addDocuments(texts: texts)
        
        // Create new instance with same config
        let config = VecturaConfig(name: "test-db", dimension: 384)
        let newVectura = try await VecturaKit(config: config)
        
        // Search should work with new instance
        let results = try await newVectura.search(query: "Document")
        XCTAssertEqual(results.count, 2)
        XCTAssertTrue(ids.contains(results[0].id))
        XCTAssertTrue(ids.contains(results[1].id))
    }
    
    func testSearchThreshold() async throws {
        let documents = [
            "Very relevant document about cats",
            "Somewhat relevant about pets",
            "Completely irrelevant about weather",
        ]
        _ = try await vectura.addDocuments(texts: documents)
        
        // With high threshold, should get fewer results
        let results = try await vectura.search(query: "cats and pets", threshold: 0.8)
        XCTAssertLessThan(results.count, 3)
    }
    
    func testCustomIds() async throws {
        let customId = UUID()
        let text = "Document with custom ID"
        
        let resultId = try await vectura.addDocument(text: text, id: customId)
        XCTAssertEqual(customId, resultId)
        
        let results = try await vectura.search(query: text)
        XCTAssertEqual(results[0].id, customId)
    }
    
    func testModelReuse() async throws {
        // Multiple operations should reuse the same model
        let start = Date()
        for i in 1...5 {
            _ = try await vectura.addDocument(text: "Test document \(i)")
        }
        let duration = Date().timeIntervalSince(start)
        
        // If model is being reused, this should be relatively quick
        XCTAssertLessThan(duration, 5.0)  // Adjust threshold as needed
    }
    
    func testEmptySearch() async throws {
        let results = try await vectura.search(query: "test query")
        XCTAssertEqual(results.count, 0, "Search on empty database should return no results")
    }
    
    func testDimensionMismatch() async throws {
        // Test with wrong dimension config
        let wrongConfig = VecturaConfig(name: "wrong-dim-db", dimension: 128)
        let wrongVectura = try await VecturaKit(config: wrongConfig)
        
        let text = "Test document"
        
        do {
            _ = try await wrongVectura.addDocument(text: text)
            XCTFail("Expected dimension mismatch error")
        } catch let error as VecturaError {
            // Should throw dimension mismatch since BERT model outputs 384 dimensions
            switch error {
            case .dimensionMismatch(let expected, let got):
                XCTAssertEqual(expected, 128)
                XCTAssertEqual(got, 384)
            default:
                XCTFail("Wrong error type: \(error)")
            }
        }
    }
    
    func testDuplicateIds() async throws {
        let id = UUID()
        let text1 = "First document"
        let text2 = "Second document"
        
        // Add first document
        _ = try await vectura.addDocument(text: text1, id: id)
        
        // Adding second document with same ID should overwrite
        _ = try await vectura.addDocument(text: text2, id: id)
        
        let results = try await vectura.search(query: text2)
        XCTAssertEqual(results.count, 1)
        XCTAssertEqual(results[0].text, text2)
    }
    
    func testSearchThresholdEdgeCases() async throws {
        let documents = ["Test document"]
        _ = try await vectura.addDocuments(texts: documents)
        
        // Test with threshold = 1.0 (exact match only)
        let perfectResults = try await vectura.search(query: "Test document", threshold: 1.0)
        XCTAssertEqual(perfectResults.count, 0)  // Should find no perfect matches due to encoding differences
        
        // Test with threshold = 0.0 (all matches)
        let allResults = try await vectura.search(query: "completely different", threshold: 0.0)
        XCTAssertEqual(allResults.count, 1)  // Should return all documents
    }
    
    func testLargeNumberOfDocuments() async throws {
        let documentCount = 100
        var documents: [String] = []
        
        for i in 0..<documentCount {
            documents.append("Test document number \(i)")
        }
        
        let ids = try await vectura.addDocuments(texts: documents)
        XCTAssertEqual(ids.count, documentCount)
        
        let results = try await vectura.search(query: "document", numResults: 10)
        XCTAssertEqual(results.count, 10)
    }
    
    func testPersistenceAfterReset() async throws {
        // Add a document
        let text = "Test document"
        _ = try await vectura.addDocument(text: text)
        
        // Reset the database
        try await vectura.reset()
        
        // Verify search returns no results
        let results = try await vectura.search(query: text)
        XCTAssertEqual(results.count, 0)
        
        // Create new instance and verify it's empty
        let newVectura = try await VecturaKit(config: config)
        let newResults = try await newVectura.search(query: text)
        XCTAssertEqual(newResults.count, 0)
    }
    
    func testFolderURLModelSource() async throws {
        /// First load the model from a remote source in order to make it available in the local filesystem.
        _ = try await Bert.loadModelBundle(from: .default)
        
        /// Local model will be downloaded to a predictable location (this may break if `swift-transformers` updates where it downloads models).
        let url = try FileManager.default.url(for: .documentDirectory, in: .userDomainMask, appropriateFor: nil, create: false)
            .appending(path: "huggingface/models/\(VecturaModelSource.defaultModelId)")
        
        XCTAssertTrue(FileManager.default.fileExists(atPath: url.path(percentEncoded: false)), "Expected downloaded model to be available locally at \(url.path())")
        
        let documents = [
            "The quick brown fox jumps over the lazy dog",
            "Pack my box with five dozen liquor jugs",
            "How vexingly quick daft zebras jump",
        ]
        
        /// Proceed as usual now, but loading the model directly from the local directory instead of downloading it.
        let ids = try await vectura.addDocuments(texts: documents, model: .folder(url))
        XCTAssertEqual(ids.count, 3)
        
        let results = try await vectura.search(query: "quick jumping animals")
        XCTAssertGreaterThanOrEqual(results.count, 2)
        XCTAssertTrue(results[0].score > results[1].score)
    }
    
    func testCustomStorageDirectory() async throws {
        let customDirectoryURL = URL(filePath: NSTemporaryDirectory()).appending(path: "VecturaKitTest")
        defer { try? FileManager.default.removeItem(at: customDirectoryURL) }
        
        let instance = try await VecturaKit(config: .init(name: "test", directoryURL: customDirectoryURL, dimension: 384))
        let text = "Test document"
        let id = UUID()
        _ = try await instance.addDocument(text: text, id: id)
        
        let documentPath = customDirectoryURL.appending(path: "test/\(id).json").path(percentEncoded: false)
        XCTAssertTrue(FileManager.default.fileExists(atPath: documentPath), "Custom storage directory inserted document doesn't exist at \(documentPath)")
    }
    
    func testAddDocumentWithMetadataAndSearchByMetadata() async throws {
        let text = "Metadata test document"
        let meta: [String: String] = ["foo": "bar", "baz": "qux"]
        let id = try await vectura.addDocument(text: text, metadata: meta)
        // Search by text, check metadata
        let results = try await vectura.search(query: "Metadata test document")
        XCTAssertEqual(results.count, 1)
        XCTAssertEqual(results[0].metadata?["foo"], "bar")
        // Search by metadata filter
        let metaResults = try await vectura.search(query: "Metadata test document", filter: ["foo": "bar"])
        XCTAssertEqual(metaResults.count, 1)
        let noResults = try await vectura.search(query: "Metadata test document", filter: ["foo": "notfound"])
        XCTAssertEqual(noResults.count, 0)
    }

    func testDeleteDocumentsByMetadata() async throws {
        let id1 = try await vectura.addDocument(text: "Doc1", metadata: ["group": "A"])
        let id2 = try await vectura.addDocument(text: "Doc2", metadata: ["group": "B"])
        let id3 = try await vectura.addDocument(text: "Doc3", metadata: ["group": "A"])
        // Delete all group A
        try await vectura.deleteDocuments(filter: ["group": "A"])
        let results = try await vectura.search(query: "Doc", numResults: 10)
        let ids = results.map { $0.id }
        XCTAssertTrue(ids.contains(id2))
        XCTAssertFalse(ids.contains(id1))
        XCTAssertFalse(ids.contains(id3))
    }

    func testIngestFileChunks() async throws {
        // Create a temporary file with multiple paragraphs to ensure it gets chunked
        let fileContent = """
        This is the first chunk of text that will be used to test the chunking functionality.
        It needs to be long enough to span multiple chunks when using a small chunk size.
        
        This is the second paragraph that should end up in another chunk.
        We want to make sure the chunking works correctly across paragraphs.
        
        Finally, this is the third section of text that should be in yet another chunk.
        This way we can verify that all the metadata is properly attached to each chunk.
        """
        
        // Create a temporary file
        let tempDir = FileManager.default.temporaryDirectory
        let fileURL = tempDir.appendingPathComponent(UUID().uuidString).appendingPathExtension("txt")
        try fileContent.write(to: fileURL, atomically: true, encoding: .utf8)
        
        // Ensure cleanup
        addTeardownBlock {
            try? FileManager.default.removeItem(at: fileURL)
        }
        
        // Use a small chunk size to ensure multiple chunks
        let chunkIDs = try await vectura.ingestFileChunks(fileURL: fileURL, chunkSize: 100, overlap: 10)
        XCTAssertGreaterThan(chunkIDs.count, 1, "Text should be split into multiple chunks")
        
        // All chunks should have metadata with originalFileID and chunkIndex
        let firstChunk = try await vectura.search(query: "first chunk", filter: ["chunkIndex": "0"])
        XCTAssertEqual(firstChunk.count, 1)
        let meta = firstChunk[0].metadata
        XCTAssertNotNil(meta?["originalFileID"], "Chunk should have originalFileID metadata")
        XCTAssertEqual(meta?["chunkIndex"], "0", "First chunk should have index 0")
        XCTAssertEqual(meta?["type"], "fileChunk", "Chunk should have type fileChunk")
        XCTAssertNotNil(meta?["text"], "Chunk should have text metadata")
        
        // Deletion by fileID
        let fileID = meta?["originalFileID"]
        try await vectura.deleteDocuments(filter: ["originalFileID": fileID!])
        let afterDelete = try await vectura.search(query: "chunk", filter: ["originalFileID": fileID!])
        XCTAssertEqual(afterDelete.count, 0, "All chunks should be deleted")
    }

    func testExactMatchScoreWithChunking() async throws {
        let queryText = "how many belts do i have?"

        // 1. Create a temporary file
        let tempDir = FileManager.default.temporaryDirectory
        let fileURL = tempDir.appendingPathComponent(UUID().uuidString).appendingPathExtension("txt")
        try queryText.write(to: fileURL, atomically: true, encoding: .utf8)

        // Ensure cleanup
        addTeardownBlock {
            try? FileManager.default.removeItem(at: fileURL)
        }

        // 2. Ingest the file (will create one chunk)
        let chunkIDs = try await vectura.ingestFileChunks(fileURL: fileURL, chunkSize: 50, overlap: 10) // Small chunk size to ensure it's chunked
        XCTAssertEqual(chunkIDs.count, 1, "Ingesting the short text should result in one chunk.")
        let chunkID = chunkIDs[0]

        // 3. Search for the original text
        let results = try await vectura.search(query: queryText, numResults: 1)

        XCTAssertEqual(results.count, 1, "Should find exactly one result for the exact match.")
        guard let result = results.first else {
            XCTFail("Failed to get the first result.")
            return
        }

        // 4. Assert score is close to 1.0
        XCTAssertEqual(result.id, chunkID, "The found document ID should match the created chunk ID.")
        XCTAssertEqual(result.text, queryText, "The found document text should match the query text.")
        XCTAssertGreaterThan(result.score, 0.99, "Score for exact match via chunking should be very close to 1.0")
        XCTAssertLessThanOrEqual(result.score, 1.0, "Score should not exceed 1.0")

        print("[TEST] Exact match score (via chunking) for '\(queryText)': \(result.score)")
    }

    func testExactMatchScore() async throws {
        let queryText = "how many belts do i have?"
        let id = try await vectura.addDocument(text: queryText)

        let results = try await vectura.search(query: queryText, numResults: 1)

        XCTAssertEqual(results.count, 1, "Should find exactly one result for the exact match.")
        guard let result = results.first else {
            XCTFail("Failed to get the first result.")
            return
        }

        XCTAssertEqual(result.id, id, "The found document ID should match the added document ID.")
        XCTAssertEqual(result.text, queryText, "The found document text should match the query text.")

        // Check if the score is very close to 1.0 (e.g., > 0.99)
        // Cosine similarity of a vector with itself should be 1.0 after normalization.
        // Allow for minor floating-point inaccuracies.
        XCTAssertGreaterThan(result.score, 0.99, "Score for exact match should be very close to 1.0")
        XCTAssertLessThanOrEqual(result.score, 1.0, "Score should not exceed 1.0")

        print("[TEST] Exact match score for '\(queryText)': \(result.score)")
    }
    
    func testExactMatchScoreWithMetadata() async throws {
        let queryText = "how many belts do i have?"
        let metadata: [String: String] = ["category": "questions", "priority": "high", "type": "inventory"]
        let id = try await vectura.addDocument(text: queryText, metadata: metadata)

        let results = try await vectura.search(query: queryText, numResults: 1)

        XCTAssertEqual(results.count, 1, "Should find exactly one result for the exact match.")
        guard let result = results.first else {
            XCTFail("Failed to get the first result.")
            return
        }

        XCTAssertEqual(result.id, id, "The found document ID should match the added document ID.")
        XCTAssertEqual(result.text, queryText, "The found document text should match the query text.")
        XCTAssertNotNil(result.metadata, "The metadata should be present")
        XCTAssertEqual(result.metadata?["category"], "questions", "The metadata category should match")
        XCTAssertEqual(result.metadata?["priority"], "high", "The metadata priority should match")
        XCTAssertEqual(result.metadata?["type"], "inventory", "The metadata type should match")
        
        // Check if the score is very close to 1.0 (e.g., > 0.99)
        // Cosine similarity of a vector with itself should be 1.0 after normalization.
        // Allow for minor floating-point inaccuracies.
        XCTAssertGreaterThan(result.score, 0.99, "Score for exact match should be very close to 1.0")
        XCTAssertLessThanOrEqual(result.score, 1.0, "Score should not exceed 1.0")

        print("[TEST] Exact match score with metadata for '\(queryText)': \(result.score)")
        
        // Also test with filter to ensure metadata filtering doesn't affect scores
        let filteredResults = try await vectura.search(query: queryText, filter: ["category": "questions"])
        XCTAssertEqual(filteredResults.count, 1, "Should find exactly one result with filter")
        XCTAssertGreaterThan(filteredResults[0].score, 0.99, "Score for filtered exact match should be very close to 1.0")
    }
    
    func testMultipleEmbeddingsBatchSearch() async throws {
        try await setupVectorOnlyTest() // Use the vector-only instance

        // A set of documents with varying degrees of similarity
        let documents = [
            "how many belts do i have?",                   // Exact match
            "how many black belts do i have?",             // Very similar
            "do i have any belts in my closet?",           // Similar
            "where are all my belts stored?",              // Related 
            "what is the total count of my belt collection?", // Similar meaning, different words
            "the weather today is quite pleasant"           // Completely unrelated
        ]
        
        // Test individual exact match first to verify it works
        print("-------- TESTING INDIVIDUAL DOCUMENTS FIRST --------")
        let exactMatchId = try await vectorOnlyVectura.addDocument(text: documents[0]) // Use vectorOnlyVectura
        let exactMatchResults = try await vectorOnlyVectura.search(query: documents[0], numResults: 1) // Use vectorOnlyVectura
        
        print("[TEST] Individual exact match test:")
        if let result = exactMatchResults.first {
            print("\(result.score): \(result.text)")
            XCTAssertGreaterThan(result.score, 0.99, "Individual exact match score should be > 0.99")
        }
        
        // Reset to try with fresh database
        try await vectorOnlyVectura.reset() // Use vectorOnlyVectura
        
        // Add all documents in batch, but one by one to track IDs
        var docIds: [UUID] = []
        var docMap: [UUID: String] = [:] // To keep track of which ID maps to which text
        
        for doc in documents {
            let id = try await vectorOnlyVectura.addDocument(text: doc) // Use vectorOnlyVectura
            docIds.append(id)
            docMap[id] = doc
            
            // Verify each document can be found by its own text
            let verifyResults = try await vectorOnlyVectura.search(query: doc, numResults: 1) // Use vectorOnlyVectura
            print("[TEST] Adding document: \"\(doc)\"")
            if let verifyResult = verifyResults.first {
                print("  Verified with score \(verifyResult.score)")
                XCTAssertEqual(verifyResult.id, id, "Document should match its own ID")
                // Also check score is high for self-match
                XCTAssertGreaterThan(verifyResult.score, 0.99, "Self-match score should be > 0.99")
            } else {
                XCTFail("Could not find document by its own text: \(doc)")
            }
        }
        
        // Search with the exact text of the first document
        let query = "how many belts do i have?"
        let results = try await vectorOnlyVectura.search(query: query) // Use vectorOnlyVectura
        
        print("[TEST] Full search results for '\(query)':")
        for result in results {
            print("\(result.score): \(result.text) (ID: \(result.id))")
        }
        
        // Should get all documents since we set threshold to 0.0 and weight to 1.0 (no BM25 filtering)
        XCTAssertEqual(results.count, documents.count, "Should return all documents")
        
        // Find the position of the exact match document
        var exactMatchPosition: Int? = nil
        var exactMatchScore: Float? = nil
        
        for (index, result) in results.enumerated() {
            if result.text == documents[0] {
                exactMatchPosition = index
                exactMatchScore = result.score
                break
            }
        }
        
        // Assert we found the exact match AND it's the first result
        XCTAssertNotNil(exactMatchPosition, "Should find the exact match document in results")
        XCTAssertEqual(exactMatchPosition, 0, "Exact match should be the first result in pure vector search")
        
        if let position = exactMatchPosition, let score = exactMatchScore {
            print("[TEST] Exact match found at position \(position) with score \(score)")
            XCTAssertGreaterThan(score, 0.99, "Exact match score should be > 0.99") // Check score here
        } else {
            XCTFail("Could not find exact match in search results")
        }
        
        // Check that scores are in descending order
        for i in 1..<results.count {
            XCTAssertGreaterThanOrEqual(results[i-1].score, results[i].score, "Results should be in descending score order")
        }
        
        // Similar documents should have positive but lower scores
        if results.count > 1 {
            XCTAssertGreaterThan(results[1].score, 0.7, "Very similar document should have high score")
            XCTAssertLessThan(results[1].score, results[0].score, "Similar document should score less than exact match")
        }
        
        // The most dissimilar document should have the lowest score
        if let lastResult = results.last {
             // Check if the last result is indeed the unrelated document
             if lastResult.text == documents.last {
                 XCTAssertLessThan(lastResult.score, 0.7, "Unrelated document should have low score") // Adjusted threshold slightly
             } else {
                 // If the last result isn't the unrelated one, something might still be wrong with ranking
                 print("[WARN] Last result wasn't the expected unrelated document.")
             }
        }
        
        print("[TEST] Multiple document batch search scores:")
        for result in results {
            print("\(result.score): \(result.text)")
        }
    }
}
