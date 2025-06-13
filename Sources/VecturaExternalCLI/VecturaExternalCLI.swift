import ArgumentParser
import Foundation
import VecturaCore
import VecturaExternalKit

@available(macOS 14.0, iOS 17.0, tvOS 17.0, visionOS 1.0, watchOS 10.0, *)
@main
struct VecturaExternalCLI: AsyncParsableCommand {
    struct DocumentID: ExpressibleByArgument, Decodable {
        let uuid: UUID
        
        init(_ uuid: UUID) {
            self.uuid = uuid
        }
        
        init?(argument: String) {
            guard let uuid = UUID(uuidString: argument) else { return nil }
            self.uuid = uuid
        }
    }
    
    static let configuration = CommandConfiguration(
        commandName: "vectura-external",
        abstract: "A CLI tool for VecturaExternalKit vector database using pre-computed embeddings",
        subcommands: [Add.self, Search.self, Delete.self, Reset.self, Mock.self]
    )
    
    static func setupDB(dbName: String, dimension: Int, numResults: Int, threshold: Float) async throws
    -> VecturaExternalKit
    {
        let config = VecturaConfig(
            name: dbName,
            dimension: dimension,
            searchOptions: VecturaConfig.SearchOptions(
                defaultNumResults: numResults,
                minThreshold: threshold
            )
        )
        return try await VecturaExternalKit(config: config)
    }
}

@available(macOS 14.0, iOS 17.0, tvOS 17.0, visionOS 1.0, watchOS 10.0, *)
extension VecturaExternalCLI {
    struct Add: AsyncParsableCommand {
        static let configuration = CommandConfiguration(abstract: "Add documents with pre-computed embeddings")
        
        @Option(name: .shortAndLong, help: "Database name")
        var database: String = "external-cli-db"
        
        @Option(name: .shortAndLong, help: "Embedding dimension")
        var dimension: Int = 384
        
        @Argument(help: "Text to add")
        var text: String
        
        @Option(help: "Comma-separated embedding values")
        var embedding: String
        
        @Option(help: "Optional document ID")
        var id: String?
        
        @Option(help: "Metadata in key=value format")
        var metadata: [String] = []
        
        func run() async throws {
            let db = try await VecturaExternalCLI.setupDB(
                dbName: database, dimension: dimension, numResults: 10, threshold: 0.0)
            
            // Parse embedding
            let embeddingValues = embedding.components(separatedBy: ",").compactMap { Float($0.trimmingCharacters(in: .whitespaces)) }
            
            guard embeddingValues.count == dimension else {
                throw ValidationError("Embedding must have exactly \(dimension) values, got \(embeddingValues.count)")
            }
            
            // Parse metadata
            var metadataDict: [String: String]? = nil
            if !metadata.isEmpty {
                metadataDict = [:]
                for item in metadata {
                    let parts = item.split(separator: "=", maxSplits: 1)
                    if parts.count == 2 {
                        metadataDict![String(parts[0])] = String(parts[1])
                    }
                }
            }
            
            let docId = try await db.addDocumentWithEmbedding(
                text: text,
                embedding: embeddingValues,
                id: id.map { UUID(uuidString: $0) } ?? nil,
                metadata: metadataDict
            )
            
            print("Added document with ID: \(docId)")
        }
    }
    
    struct Search: AsyncParsableCommand {
        static let configuration = CommandConfiguration(abstract: "Search using a pre-computed query embedding")
        
        @Option(name: .shortAndLong, help: "Database name")
        var database: String = "external-cli-db"
        
        @Option(name: .shortAndLong, help: "Embedding dimension")
        var dimension: Int = 384
        
        @Option(help: "Comma-separated query embedding values")
        var embedding: String
        
        @Option(name: .shortAndLong, help: "Number of results")
        var numResults: Int = 5
        
        @Option(name: .shortAndLong, help: "Similarity threshold")
        var threshold: Float = 0.0
        
        @Option(help: "Metadata filter in key=value format")
        var filter: [String] = []
        
        func run() async throws {
            let db = try await VecturaExternalCLI.setupDB(
                dbName: database, dimension: dimension, numResults: numResults, threshold: threshold)
            
            // Parse query embedding
            let embeddingValues = embedding.components(separatedBy: ",").compactMap { Float($0.trimmingCharacters(in: .whitespaces)) }
            
            guard embeddingValues.count == dimension else {
                throw ValidationError("Query embedding must have exactly \(dimension) values, got \(embeddingValues.count)")
            }
            
            // Parse filter
            var filterDict: [String: String]? = nil
            if !filter.isEmpty {
                filterDict = [:]
                for item in filter {
                    let parts = item.split(separator: "=", maxSplits: 1)
                    if parts.count == 2 {
                        filterDict![String(parts[0])] = String(parts[1])
                    }
                }
            }
            
            let results = try await db.search(
                query: embeddingValues,
                numResults: numResults,
                threshold: threshold > 0 ? threshold : nil,
                filter: filterDict
            )
            
            if results.isEmpty {
                print("No results found.")
            } else {
                print("Found \(results.count) results:")
                for (index, result) in results.enumerated() {
                    print("\n\(index + 1). Score: \(String(format: "%.4f", result.score))")
                    print("   ID: \(result.id)")
                    print("   Text: \(result.text)")
                    if let metadata = result.metadata, !metadata.isEmpty {
                        print("   Metadata: \(metadata)")
                    }
                }
            }
        }
    }
    
    struct Delete: AsyncParsableCommand {
        static let configuration = CommandConfiguration(abstract: "Delete documents by metadata filter")
        
        @Option(name: .shortAndLong, help: "Database name")
        var database: String = "external-cli-db"
        
        @Option(name: .shortAndLong, help: "Embedding dimension")
        var dimension: Int = 384
        
        @Option(help: "Metadata filter in key=value format")
        var filter: [String] = []
        
        func run() async throws {
            let db = try await VecturaExternalCLI.setupDB(
                dbName: database, dimension: dimension, numResults: 10, threshold: 0.0)
            
            guard !filter.isEmpty else {
                throw ValidationError("At least one filter must be specified")
            }
            
            // Parse filter
            var filterDict: [String: String] = [:]
            for item in filter {
                let parts = item.split(separator: "=", maxSplits: 1)
                if parts.count == 2 {
                    filterDict[String(parts[0])] = String(parts[1])
                }
            }
            
            let beforeCount = db.documentCount()
            try await db.deleteDocuments(filter: filterDict)
            let afterCount = db.documentCount()
            
            print("Deleted \(beforeCount - afterCount) documents matching filter: \(filterDict)")
        }
    }
    
    struct Reset: AsyncParsableCommand {
        static let configuration = CommandConfiguration(abstract: "Reset the database (delete all documents)")
        
        @Option(name: .shortAndLong, help: "Database name")
        var database: String = "external-cli-db"
        
        @Option(name: .shortAndLong, help: "Embedding dimension")
        var dimension: Int = 384
        
        func run() async throws {
            let db = try await VecturaExternalCLI.setupDB(
                dbName: database, dimension: dimension, numResults: 10, threshold: 0.0)
            
            try await db.reset()
            print("Database '\(database)' has been reset.")
        }
    }
    
    struct Mock: AsyncParsableCommand {
        static let configuration = CommandConfiguration(abstract: "Add mock data with random embeddings for testing")
        
        @Option(name: .shortAndLong, help: "Database name")
        var database: String = "external-cli-db"
        
        @Option(name: .shortAndLong, help: "Embedding dimension")
        var dimension: Int = 384
        
        @Option(name: .shortAndLong, help: "Number of documents to add")
        var count: Int = 10
        
        func run() async throws {
            let db = try await VecturaExternalCLI.setupDB(
                dbName: database, dimension: dimension, numResults: 10, threshold: 0.0)
            
            let sampleTexts = [
                "The quick brown fox jumps over the lazy dog",
                "Machine learning is transforming the world",
                "Vector databases enable semantic search",
                "Swift is a powerful programming language",
                "Artificial intelligence is the future",
                "Data science drives business decisions",
                "Cloud computing scales applications",
                "Mobile apps connect people globally",
                "Open source software powers innovation",
                "Natural language processing understands text"
            ]
            
            var texts: [String] = []
            var embeddings: [[Float]] = []
            var metadatas: [[String: String]?] = []
            
            for i in 0..<count {
                let text = sampleTexts[i % sampleTexts.count] + " \(i)"
                
                // Generate random embedding
                let embedding = (0..<dimension).map { _ in Float.random(in: -1...1) }
                
                // Generate metadata
                let metadata = [
                    "source": "mock_data",
                    "index": "\(i)",
                    "category": ["tech", "ai", "programming", "data"][i % 4]
                ]
                
                texts.append(text)
                embeddings.append(embedding)
                metadatas.append(metadata)
            }
            
            let documentIds = try await db.addDocumentsWithEmbeddings(
                texts: texts,
                embeddings: embeddings,
                ids: nil,
                metadatas: metadatas
            )
            
            print("Added \(documentIds.count) mock documents to database '\(database)'")
            print("Example search: vectura-external search --embedding=\"" + (0..<dimension).map { _ in "0.1" }.joined(separator: ",") + "\"")
        }
    }
} 