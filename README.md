# VecturaKit

VecturaKit is a Swift-based vector database designed for on-device applications, enabling advanced user experiences through local vector storage and retrieval. Inspired by [Dripfarm's SVDB](https://github.com/Dripfarm/SVDB), **VecturaKit** utilizes `MLTensor` and [`swift-embeddings`](https://github.com/jkrukowski/swift-embeddings) for generating and managing embeddings. The framework offers two primary modules: `VecturaKit`, which supports diverse embedding models via `swift-embeddings`, and `VecturaMLXKit`, which leverages Apple's MLX framework for accelerated processing.

## Key Features

-   **On-Device Storage:** Stores and manages vector embeddings locally, enhancing privacy and reducing latency.
-   **Hybrid Search:** Combines vector similarity with BM25 text search for comprehensive and relevant search results (`VecturaKit`).
-   **Batch Processing:** Indexes documents in parallel for faster data ingestion.
-   **Persistent Storage:** Automatically saves and loads document data, preserving the database state across app sessions.
-   **Configurable Search:** Customizes search behavior with adjustable thresholds, result limits, and hybrid search weights.
-   **Custom Storage Location:** Specifies a custom directory for database storage.
-   **MLX Support:** Employs Apple's MLX framework for accelerated embedding generation and search operations (`VecturaMLXKit`).
-   **Memory Optimized:** Implements batch processing and efficient memory management to reduce memory consumption.
-   **CLI Tool:** Includes a command-line interface (CLI) for database management, testing, and debugging for both `VecturaKit` and `VecturaMLXKit`.
-   **External Embedding Support:** Accepts pre-computed embeddings from external sources, allowing for flexibility in embedding generation.

## Supported Platforms

-   macOS 14.0 or later
-   iOS 17.0 or later
-   tvOS 17.0 or later
-   visionOS 1.0 or later
-   watchOS 10.0 or later

## Installation

### Swift Package Manager

To integrate VecturaKit into your project using Swift Package Manager, add the following dependency in your `Package.swift` file:

```swift
dependencies: [
    .package(url: "https://github.com/rryam/VecturaKit.git", branch: "main"),
],
```

### Dependencies

VecturaKit relies on the following Swift packages:

-   [swift-embeddings](https://github.com/jkrukowski/swift-embeddings): Used in `VecturaKit` for generating text embeddings using various models.
-   [swift-argument-parser](https://github.com/apple/swift-argument-parser): Used for creating the command-line interface.
-   [mlx-swift-examples](https://github.com/ml-explore/mlx-swift-examples): Provides MLX-based embeddings and vector search capabilities, specifically for `VecturaMLXKit`.

## Usage

### Core VecturaKit

1.  **Import VecturaKit**

    ```swift
    import VecturaKit
    ```

2.  **Create Configuration and Initialize Database**

    ```swift
    import Foundation
    import VecturaKit

    let config = VecturaConfig(
        name: "my-vector-db",
        directoryURL: nil,  // Optional custom storage location
        dimension: 384,     // Matches the default BERT model dimension
        searchOptions: VecturaConfig.SearchOptions(
            defaultNumResults: 10,
            minThreshold: 0.7,
            hybridWeight: 0.5,  // Balance between vector and text search
            k1: 1.2,           // BM25 parameters
            b: 0.75
        )
    )

    let vectorDB = try await VecturaKit(config: config)
    ```

3.  **Add Documents**

    Single document:

    ```swift
    let text = "Sample text to be embedded"
    let documentId = try await vectorDB.addDocument(
        text: text,
        id: UUID(),  // Optional, will be generated if not provided
        model: .id("sentence-transformers/all-MiniLM-L6-v2")  // Optional, this is the default
    )
    ```

    Multiple documents in batch:

    ```swift
    let texts = [
        "First document text",
        "Second document text",
        "Third document text"
    ]
    let documentIds = try await vectorDB.addDocuments(
        texts: texts,
        ids: nil,  // Optional array of UUIDs
         model: .id("sentence-transformers/all-MiniLM-L6-v2") // Optional model
    )
    ```
    
    **Using External Embeddings:**
    
    Add a single document with a pre-computed embedding:
    
    ```swift
    // Pre-computed embedding from an external source (e.g., OpenAI, Cohere, custom model)
    let externalEmbedding: [Float] = [...] // Must match config.dimension
    
    let documentId = try await vectorDB.addDocumentWithEmbedding(
        text: "Document with external embedding",
        embedding: externalEmbedding,
        id: UUID()  // Optional, will be generated if not provided
    )
    ```
    
    Add multiple documents with pre-computed embeddings:
    
    ```swift
    let texts = [
        "First document with external embedding",
        "Second document with external embedding"
    ]
    
    // Pre-computed embeddings from an external source
    let externalEmbeddings: [[Float]] = [
        [...], // First embedding, must match config.dimension
        [...]  // Second embedding, must match config.dimension
    ]
    
    let documentIds = try await vectorDB.addDocumentsWithEmbeddings(
        texts: texts,
        embeddings: externalEmbeddings,
        ids: nil  // Optional array of UUIDs
    )
    ```

4.  **Search Documents**

    Search by text (hybrid search):

    ```swift
    let results = try await vectorDB.search(
        query: "search query",
        numResults: 5,      // Optional
        threshold: 0.8,     // Optional
        model: .id("sentence-transformers/all-MiniLM-L6-v2")  // Optional
    )

    for result in results {
        print("Document ID: \(result.id)")
        print("Text: \(result.text)")
        print("Similarity Score: \(result.score)")
        print("Created At: \(result.createdAt)")
    }
    ```

    Search by vector embedding:

    ```swift
    let results = try await vectorDB.search(
        query: embeddingArray,  // [Float] matching config.dimension
        numResults: 5,  // Optional
        threshold: 0.8  // Optional
    )
    ```
    
    **Search with External Embeddings:**
    
    ```swift
    // Pre-computed query embedding from an external source
    let externalQueryEmbedding: [Float] = [...] // Must match config.dimension
    
    let results = try await vectorDB.searchWithExternalEmbedding(
        queryText: "search query text", // Original text for hybrid search
        queryEmbedding: externalQueryEmbedding,
        numResults: 5,  // Optional
        threshold: 0.8  // Optional
    )
    ```

5.  **Document Management**

    Update document:

    ```swift
    try await vectorDB.updateDocument(
        id: documentId,
        newText: "Updated text",
        model: .id("sentence-transformers/all-MiniLM-L6-v2")  // Optional
    )
    ```

    Delete documents:

    ```swift
    try await vectorDB.deleteDocuments(ids: [documentId1, documentId2])
    ```

    Reset database:

    ```swift
    try await vectorDB.reset()
    ```
    
    Get document count:
    
    ```swift
    let count = vectorDB.documentCount()
    print("The database contains \(count) documents")
    ```
    
    Check if document exists:
    
    ```swift
    // Check if a single document exists
    let exists = vectorDB.documentExists(id: documentId)
    if exists {
        print("Document exists")
    }
    
    // Check if multiple documents exist
    let existenceMap = vectorDB.documentsExist(ids: [id1, id2, id3])
    for (id, exists) in existenceMap {
        print("Document \(id): \(exists ? "exists" : "does not exist")")
    }
    ```

## New Features in VecturaKit

### Document Metadata Support

- You can now store arbitrary key-value metadata (e.g., `[String: String]`) with each document.
- Metadata is included in all search results and can be used for filtering and deletion.

#### Adding Documents with Metadata

```swift
let meta: [String: String] = ["source": "fileA", "type": "note"]
let docId = try await vectorDB.addDocument(text: "My note", metadata: meta)
```

#### Metadata-Based Search

```swift
let results = try await vectorDB.search(query: "note", filter: ["source": "fileA"])
```

#### Metadata-Based Deletion

```swift
try await vectorDB.deleteDocuments(filter: ["source": "fileA"])
```

### Automatic File Chunking and Ingestion

- Ingest any file that can be read as plain text (e.g., .txt, .md, .swift, .py, etc.).
- The file is split into overlapping chunks, each chunk is stored as a document with metadata including the original file ID, chunk index, and chunk text.

#### Example: Ingesting a File

```swift
let fileURL = URL(fileURLWithPath: "/path/to/your/file.swift")
let chunkIDs = try await vectorDB.ingestFileChunks(
    fileURL: fileURL,
    chunkSize: 1000,   // Number of characters per chunk
    overlap: 100       // Overlap between chunks
)
```

#### Searching and Deleting by File Chunk Metadata

```swift
// Search for all chunks from a specific file
let results = try await vectorDB.search(query: "some code", filter: ["originalFileID": fileID.uuidString])

// Delete all chunks for a file
try await vectorDB.deleteDocuments(filter: ["originalFileID": fileID.uuidString])
```

### VecturaMLXKit (MLX Version)

VecturaMLXKit harnesses Apple's MLX framework for accelerated processing, delivering optimized performance for on-device machine learning tasks.

1.  **Import VecturaMLXKit**

    ```swift
    import VecturaMLXKit
    ```

2.  **Initialize Database**

    ```swift
    import VecturaMLXKit
    import MLXEmbedders

    let config = VecturaConfig(
      name: "my-mlx-vector-db",
      dimension: 768 //  nomic_text_v1_5 model outputs 768-dimensional embeddings
    )
    
    // Basic initialization
    let vectorDB = try await VecturaMLXKit(config: config, modelConfiguration: .nomic_text_v1_5)
    
    // Memory-optimized initialization with custom parameters
    let VectorDB = try await VecturaMLXKit(
        config: config, 
        modelConfiguration: .nomic_text_v1_5,
        maxBatchSize: 8,         // Process documents in smaller batches to reduce memory usage
        maxTokenLength: 256      // Limit token length to avoid large tensor allocations
    )
    ```

3.  **Add Documents**

    ```swift
    let texts = [
      "First document text",
      "Second document text",
      "Third document text"
    ]
    let documentIds = try await vectorDB.addDocuments(texts: texts)
    ```

    **Using External Embeddings with VecturaMLXKit:**
    
    ```swift
    // Pre-computed embeddings from an external source (e.g., OpenAI, Ollama)
    let texts = ["First document", "Second document"]
    let externalEmbeddings: [[Float]] = [...] // Must match the config.dimension
    
    let documentIds = try await vectorDB.addDocumentsWithEmbeddings(
        texts: texts,
        embeddings: externalEmbeddings,
        ids: nil // Optional array of UUIDs
    )
    
    // Add a single document with external embedding
    let singleDocId = try await vectorDB.addDocumentWithEmbedding(
        text: "Single document",
        embedding: externalEmbeddings[0],
        id: UUID() // Optional, will be generated if not provided
    )
    ```

4.  **Search Documents**

    ```swift
    let results = try await vectorDB.search(
        query: "search query",
        numResults: 5,      // Optional
        threshold: 0.8     // Optional
    )

    for result in results {
        print("Document ID: \(result.id)")
        print("Text: \(result.text)")
        print("Similarity Score: \(result.score)")
        print("Created At: \(result.createdAt)")
    }
    ```

    **Search with External Embeddings:**
    
    ```swift
    // Pre-computed query embedding from external provider
    let externalQueryEmbedding: [Float] = [...] // Must match config.dimension
    
    let results = try await vectorDB.searchWithExternalEmbedding(
        queryEmbedding: externalQueryEmbedding,
        numResults: 5,  // Optional
        threshold: 0.8  // Optional
    )
    ```

5.  **Document Management**

    Update document:

    ```swift
    try await vectorDB.updateDocument(
         id: documentId,
         newText: "Updated text"
     )
    ```

    Delete documents:

    ```swift
    try await vectorDB.deleteDocuments(ids: [documentId1, documentId2])
    ```

    Reset database:

    ```swift
    try await vectorDB.reset()
    ```

### Using External Embedding Providers

VecturaKit supports using embeddings from external providers like OpenAI and Ollama while still benefiting from its storage, retrieval, and hybrid search capabilities.

#### Integration with OpenAI Embeddings

```swift
import Foundation
import VecturaKit

// A simple OpenAI embedding client
class OpenAIEmbeddings {
    private let apiKey: String
    private let model: String
    private let dimension: Int
    
    init(apiKey: String, model: String = "text-embedding-3-small", dimension: Int = 1536) {
        self.apiKey = apiKey
        self.model = model
        self.dimension = dimension
    }
    
    func embed(text: String) async throws -> [Float] {
        let url = URL(string: "https://api.openai.com/v1/embeddings")!
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.addValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
        request.addValue("application/json", forHTTPHeaderField: "Content-Type")
        
        let payload: [String: Any] = [
            "model": model,
            "input": text
        ]
        request.httpBody = try JSONSerialization.data(withJSONObject: payload)
        
        let (data, _) = try await URLSession.shared.data(for: request)
        let json = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        
        guard let data = json?["data"] as? [[String: Any]],
              let embedding = data.first?["embedding"] as? [Double] else {
            throw NSError(domain: "OpenAIEmbedding", code: 1, userInfo: [NSLocalizedDescriptionKey: "Failed to parse OpenAI response"])
        }
        
        return embedding.map { Float($0) }
    }
    
    func embedBatch(texts: [String]) async throws -> [[Float]] {
        return try await withThrowingTaskGroup(of: [Float].self) { group in
            for text in texts {
                group.addTask { try await self.embed(text: text) }
            }
            
            var results: [[Float]] = []
            for try await embedding in group {
                results.append(embedding)
            }
            return results
        }
    }
}

// Usage with VecturaKit
func exampleWithOpenAI() async throws {
    // Setup OpenAI client
    let openAI = OpenAIEmbeddings(apiKey: "your-api-key")
    
    // Setup VecturaKit with matching dimension (1536 for OpenAI text-embedding-3-small)
    let config = VecturaConfig(name: "openai-vectura", dimension: 1536)
    let vectorDB = try await VecturaKit(config: config)
    
    // Add documents using OpenAI embeddings
    let texts = ["First document about AI", "Second document about databases"]
    let embeddings = try await openAI.embedBatch(texts: texts)
    
    let docIds = try await vectorDB.addDocumentsWithEmbeddings(
        texts: texts,
        embeddings: embeddings
    )
    
    // Search using OpenAI embedding
    let queryText = "Tell me about artificial intelligence"
    let queryEmbedding = try await openAI.embed(text: queryText)
    
    let results = try await vectorDB.searchWithExternalEmbedding(
        queryText: queryText,
        queryEmbedding: queryEmbedding,
        numResults: 5
    )
    
    for result in results {
        print("Document: \(result.text)")
        print("Score: \(result.score)")
    }
}
```

#### Integration with Ollama Embeddings

```swift
import Foundation
import VecturaMLXKit

class OllamaEmbeddings {
    private let baseURL: URL
    private let model: String
    private let dimension: Int
    
    init(baseURL: URL = URL(string: "http://localhost:11434")!, 
         model: String = "nomic-embed-text", 
         dimension: Int = 768) {
        self.baseURL = baseURL
        self.model = model
        self.dimension = dimension
    }
    
    func embed(text: String) async throws -> [Float] {
        let url = baseURL.appendingPathComponent("api/embeddings")
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.addValue("application/json", forHTTPHeaderField: "Content-Type")
        
        let payload: [String: Any] = [
            "model": model,
            "prompt": text
        ]
        request.httpBody = try JSONSerialization.data(withJSONObject: payload)
        
        let (data, _) = try await URLSession.shared.data(for: request)
        let json = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        
        guard let embedding = json?["embedding"] as? [Double] else {
            throw NSError(domain: "OllamaEmbedding", code: 1, userInfo: [NSLocalizedDescriptionKey: "Failed to parse Ollama response"])
        }
        
        return embedding.map { Float($0) }
    }
    
    func embedBatch(texts: [String]) async throws -> [[Float]] {
        var results: [[Float]] = []
        for text in texts {
            let embedding = try await embed(text: text)
            results.append(embedding)
        }
        return results
    }
}

// Usage with VecturaMLXKit
func exampleWithOllama() async throws {
    // Setup Ollama client
    let ollama = OllamaEmbeddings()
    
    // Setup VecturaMLXKit with matching dimension (768 for nomic-embed-text)
    let config = VecturaConfig(name: "ollama-vectura", dimension: 768)
    let vectorDB = try await VecturaMLXKit(config: config)
    
    // Add documents using Ollama embeddings
    let texts = ["First document about AI", "Second document about databases"]
    let embeddings = try await ollama.embedBatch(texts: texts)
    
    let docIds = try await vectorDB.addDocumentsWithEmbeddings(
        texts: texts,
        embeddings: embeddings
    )
    
    // Search using Ollama embedding
    let queryText = "Tell me about artificial intelligence"
    let queryEmbedding = try await ollama.embed(text: queryText)
    
    let results = try await vectorDB.searchWithExternalEmbedding(
        queryEmbedding: queryEmbedding,
        numResults: 5
    )
    
    for result in results {
        print("Document: \(result.text)")
        print("Score: \(result.score)")
    }
}
```

## Command Line Interface

VecturaKit includes a command-line interface for both the standard and MLX versions, facilitating easy database management.

**Standard CLI Tool**

```bash
# Add documents
vectura add "First document" "Second document" "Third document" \
  --db-name "my-vector-db" \
  --dimension 384 \
  --model-id "sentence-transformers/all-MiniLM-L6-v2"

# Search documents
vectura search "search query" \
  --db-name "my-vector-db" \
  --dimension 384 \
  --threshold 0.7 \
  --num-results 5 \
  --model-id "sentence-transformers/all-MiniLM-L6-v2"

# Update document
vectura update <document-uuid> "Updated text content" \
  --db-name "my-vector-db" \
  --dimension 384 \
  --model-id "sentence-transformers/all-MiniLM-L6-v2"

# Delete documents
vectura delete <document-uuid-1> <document-uuid-2> \
  --db-name "my-vector-db" \
  --dimension 384

# Reset database
vectura reset \
  --db-name "my-vector-db" \
  --dimension 384

# Run demo with sample data
vectura mock \
  --db-name "my-vector-db" \
  --dimension 384 \
  --threshold 0.7 \
  --num-results 10 \
  --model-id "sentence-transformers/all-MiniLM-L6-v2"
```

Common options:

-   `--db-name, -d`: Database name (default: "vectura-cli-db")
-   `--dimension, -v`: Vector dimension (default: 384)
-   `--threshold, -t`: Minimum similarity threshold (default: 0.7)
-   `--num-results, -n`: Number of results to return (default: 10)
-   `--model-id, -m`: Model ID for embeddings (default: "sentence-transformers/all-MiniLM-L6-v2")

**MLX CLI Tool**

```bash
# Add documents
vectura-mlx add "First document" "Second document" "Third document" \
  --db-name "my-mlx-vector-db" \
  --batch-size 8 \
  --max-token-length 256

# Search documents
vectura-mlx search "search query" \
  --db-name "my-mlx-vector-db" \
  --threshold 0.7 \
  --num-results 5 \
  --batch-size 8 \
  --max-token-length 256

# Update document
vectura-mlx update <document-uuid> "Updated text content" \
  --db-name "my-mlx-vector-db" \
  --batch-size 8 \
  --max-token-length 256

# Delete documents
vectura-mlx delete <document-uuid-1> <document-uuid-2> \
  --db-name "my-mlx-vector-db" \
  --batch-size 8 \
  --max-token-length 256

# Reset database
vectura-mlx reset --db-name "my-mlx-vector-db"

# Run demo with sample data
vectura-mlx mock \
  --db-name "my-mlx-vector-db" \
  --batch-size 8 \
  --max-token-length 256
```

MLX-specific options:

-   `--batch-size, -b`: Maximum batch size for processing (default: 16)
-   `--max-token-length, -t`: Maximum token length for documents (default: 512)

## License

VecturaKit is released under the MIT License. See the [LICENSE](LICENSE) file for more information. Copyright (c) 2025 Rudrank Riyam.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your improvements.

### Development

The project is structured as a Swift Package.  It includes the following key targets:

- `VecturaKit`: The core vector database library.
- `VecturaMLXKit`:  The MLX-accelerated version of the library.
- `vectura-cli`:  The command-line interface for `VecturaKit`.
- `vectura-mlx-cli`: The command-line interface for `VecturaMLXKit`.

To build and test the project, use the following commands:

```bash
swift build
swift test
```

The project also includes CI workflows defined in `.github/workflows` to automate building and testing on pull requests and pushes to the `main` branch.  The workflows require Xcode 16.1 and Swift 6.0.

Debugging configurations are provided in `.vscode/launch.json` for the `vectura-cli`.  These can be used to launch the CLI with the debugger attached.

### Continuous Integration

The project uses GitHub Actions for continuous integration. The following workflows are defined:

- `.github/workflows/build_and_test_mlx.yml`: Builds and tests the `VecturaMLXKit` target.
- `.github/workflows/build_and_test_vectura.yml`: Builds and tests the `VecturaKit` and `vectura-cli` targets.
- `.github/workflows/update-readme.yml`: Automatically updates the `README.md` file using a Python script that calls the Gemini AI model. This workflow is triggered on pushes to the `main` branch and creates a pull request with the updated README.
