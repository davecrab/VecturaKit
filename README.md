# VecturaKit

A Swift package providing vector database implementations with different embedding capabilities and platform requirements.

## Overview

VecturaKit is now organized into multiple libraries to support different embedding approaches and minimize platform version requirements:

- **VecturaCore**: Common functionality and protocols (macOS 14+, iOS 17+)
- **VecturaKit**: Full-featured vector database with swift-embeddings support (macOS 15+, iOS 18+)
- **VecturaMLXKit**: Vector database using MLX for Apple Silicon optimization (macOS 14+, iOS 17+)
- **VecturaExternalKit**: Lightweight vector database for pre-computed embeddings (macOS 14+, iOS 17+)

## Platform Requirements

| Library | macOS | iOS | tvOS | visionOS | watchOS |
|---------|-------|-----|------|-----------|---------|
| VecturaCore | 14.0+ | 17.0+ | 17.0+ | 1.0+ | 10.0+ |
| VecturaKit | 15.0+ | 18.0+ | 18.0+ | 2.0+ | 11.0+ |
| VecturaMLXKit | 14.0+ | 17.0+ | 17.0+ | 1.0+ | 10.0+ |
| VecturaExternalKit | 14.0+ | 17.0+ | 17.0+ | 1.0+ | 10.0+ |

## When to Use Each Library

### VecturaKit
Use when you need automatic text embedding generation and can target the latest platforms:
- Includes swift-embeddings for BERT, CLIP, and other transformer models
- Full feature set including hybrid search
- Requires macOS 15+ / iOS 18+

### VecturaMLXKit  
Use when you want optimized performance on Apple Silicon with lower platform requirements:
- Uses MLX for efficient inference on Apple chips
- Good balance of features and performance
- Requires macOS 14+ / iOS 17+

### VecturaExternalKit
Use when you have pre-computed embeddings and want the lowest platform requirements:
- Only handles pre-computed embeddings (no embedding generation)
- Minimal dependencies and platform requirements
- Perfect for apps that generate embeddings elsewhere
- Requires macOS 14+ / iOS 17+

## Installation

Add VecturaKit to your Swift package dependencies:

```swift
dependencies: [
    .package(url: "https://github.com/rryam/VecturaKit.git", from: "1.0.0")
]
```

Then import the specific library you need:

```swift
import VecturaExternalKit  // For pre-computed embeddings
import VecturaMLXKit      // For MLX-powered embeddings  
import VecturaKit         // For swift-embeddings
import VecturaCore        // For shared types only
```

## Usage Examples

### VecturaExternalKit (Pre-computed Embeddings)

```swift
import VecturaExternalKit
import VecturaCore

let config = VecturaConfig(name: "my-db", dimension: 384)
let vectorDB = try await VecturaExternalKit(config: config)

// Add documents with pre-computed embeddings
let embeddings = [
    [0.1, 0.2, 0.3, ...], // 384-dimensional embedding
    [0.4, 0.5, 0.6, ...]  // Another 384-dimensional embedding
]

let docIds = try await vectorDB.addDocumentsWithEmbeddings(
    texts: ["Document 1", "Document 2"],
    embeddings: embeddings,
    metadatas: [["category": "tech"], ["category": "science"]]
)

// Search with a pre-computed query embedding
let queryEmbedding = [0.1, 0.2, 0.3, ...] // Your query embedding
let results = try await vectorDB.search(
    query: queryEmbedding,
    numResults: 5,
    filter: ["category": "tech"]
)
```

### VecturaMLXKit (MLX Embeddings)

```swift
import VecturaMLXKit
import VecturaCore

let config = VecturaConfig(name: "mlx-db", dimension: 768)
let vectorDB = try await VecturaMLXKit(config: config)

// Add documents - embeddings generated automatically
let docIds = try await vectorDB.addDocuments(
    texts: ["Machine learning document", "AI research paper"],
    metadatas: [["source": "arxiv"], ["source": "nature"]]
)

// Search with text - embedding generated automatically
let results = try await vectorDB.search(
    query: "artificial intelligence",
    numResults: 10
)
```

### VecturaKit (Swift-Embeddings)

```swift
import VecturaKit
import VecturaCore

let config = VecturaConfig(name: "full-db", dimension: 384)
let vectorDB = try await VecturaKit(config: config)

// Add documents with automatic embedding generation
let docIds = try await vectorDB.addDocuments(
    texts: ["Neural networks", "Deep learning"],
    model: .id("sentence-transformers/all-MiniLM-L6-v2"),
    metadatas: [["type": "ml"], ["type": "dl"]]
)

// Also supports pre-computed embeddings
let precomputedIds = try await vectorDB.addDocumentsWithEmbeddings(
    texts: ["External embedding"],
    embeddings: [[0.1, 0.2, 0.3, ...]]
)
```

## Command Line Tools

Each library includes a CLI tool for testing and experimentation:

```bash
# VecturaExternalKit CLI
swift run vectura-external-cli mock --count 10 --dimension 384
swift run vectura-external-cli search --embedding="0.1,0.2,0.3,..."

# VecturaMLXKit CLI  
swift run vectura-mlx-cli add "Machine learning text"
swift run vectura-mlx-cli search "AI research"

# VecturaKit CLI
swift run vectura-cli add "Neural networks" --model-id "sentence-transformers/all-MiniLM-L6-v2"
swift run vectura-cli search "deep learning"
```

## Architecture

The new modular architecture separates concerns:

- **VecturaCore**: Contains shared protocols, data structures, and utilities
- **VecturaExternalKit**: Minimal implementation for pre-computed embeddings
- **VecturaMLXKit**: MLX-based embedding generation for Apple Silicon
- **VecturaKit**: Full-featured implementation with swift-embeddings

This allows you to choose the right balance of features vs. platform requirements for your use case.

## Features

All libraries support:
- ✅ Vector similarity search with cosine similarity
- ✅ Metadata filtering
- ✅ Batch operations
- ✅ Persistent storage (JSON files)
- ✅ Hybrid search (vector + BM25 text search)
- ✅ Configurable search options

Additional features by library:
- **VecturaKit**: Automatic embedding generation with Transformers
- **VecturaMLXKit**: Automatic embedding generation optimized for Apple Silicon
- **VecturaExternalKit**: Lightweight for pre-computed embeddings only

## License

This project is licensed under the MIT License.
