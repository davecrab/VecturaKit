# VecturaKitLite

A lightweight Swift package providing a vector database implementation for use with pre-computed embeddings. This is a streamlined fork that focuses on storage and search capabilities without built-in embedding generation.

## Overview

VecturaKitLite is designed to work with external embedding providers (like SwiftAIKit) and provides efficient vector storage and similarity search capabilities.

**Libraries:**
- **VecturaCore**: Common functionality, protocols, and data structures
- **VecturaExternalKit**: Vector database implementation for pre-computed embeddings

## Platform Requirements

| Library | macOS | iOS | tvOS | visionOS | watchOS |
|---------|-------|-----|------|-----------|---------|
| VecturaCore | 14.0+ | 17.0+ | 17.0+ | 1.0+ | 10.0+ |
| VecturaExternalKit | 14.0+ | 17.0+ | 17.0+ | 1.0+ | 10.0+ |

## When to Use VecturaKitLite

Use VecturaKitLite when:
- You have an external embedding provider (e.g., SwiftAIKit, OpenAI API, etc.)
- You want minimal dependencies and lower platform requirements
- You need efficient vector storage and similarity search
- You want to decouple embedding generation from vector storage

## Installation

Add VecturaKitLite to your Swift package dependencies:

```swift
dependencies: [
    .package(url: "https://github.com/davecrab/VecturaKit.git", branch: "main")
]
```

Then import the libraries you need:

```swift
import VecturaExternalKit  // For vector database operations
import VecturaCore         // For shared types (VecturaConfig, etc.)
```

## Usage Example

```swift
import VecturaExternalKit
import VecturaCore

// Configure the vector database
let config = VecturaConfig(name: "my-db", dimension: 384)
let vectorDB = try await VecturaExternalKit(config: config)

// Add documents with pre-computed embeddings (from SwiftAIKit or other provider)
let embeddings: [[Float]] = [
    [0.1, 0.2, 0.3, ...], // 384-dimensional embedding
    [0.4, 0.5, 0.6, ...]  // Another 384-dimensional embedding
]

let docIds = try await vectorDB.addDocumentsWithEmbeddings(
    texts: ["Document 1", "Document 2"],
    embeddings: embeddings,
    metadatas: [["category": "tech"], ["category": "science"]]
)

// Search with a pre-computed query embedding
let queryEmbedding: [Float] = [0.1, 0.2, 0.3, ...] // Your query embedding
let results = try await vectorDB.search(
    query: queryEmbedding,
    numResults: 5,
    filter: ["category": "tech"]
)

for result in results {
    print("Score: \(result.score), Text: \(result.text)")
}
```

## Command Line Tool

A CLI tool is included for testing and experimentation:

```bash
# Add mock data with random embeddings
swift run vectura-cli mock --count 10 --dimension 384

# Search with an embedding
swift run vectura-cli search --embedding="0.1,0.2,0.3,..."

# Reset the database
swift run vectura-cli reset
```

## Features

- ✅ Vector similarity search with cosine similarity
- ✅ Metadata filtering
- ✅ Batch operations
- ✅ Persistent storage (JSON files)
- ✅ Hybrid search (vector + BM25 text search)
- ✅ Configurable search options
- ✅ Minimal dependencies
- ✅ Lower platform requirements (macOS 14+, iOS 17+)

## Integration with SwiftAIKit

VecturaKitLite is designed to work seamlessly with SwiftAIKit for embedding generation:

```swift
import SwiftAIKit
import VecturaExternalKit
import VecturaCore

// Generate embeddings with SwiftAIKit
let aiKit = SwiftAIKit()
let embeddings = try await aiKit.generateEmbeddings(texts: ["Hello world", "AI is amazing"])

// Store in VecturaKitLite
let config = VecturaConfig(name: "my-db", dimension: embeddings[0].count)
let vectorDB = try await VecturaExternalKit(config: config)

let ids = try await vectorDB.addDocumentsWithEmbeddings(
    texts: ["Hello world", "AI is amazing"],
    embeddings: embeddings
)
```

## License

This project is licensed under the MIT License.