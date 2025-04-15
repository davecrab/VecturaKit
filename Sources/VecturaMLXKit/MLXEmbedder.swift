import Foundation
import MLX
import MLXEmbedders
import VecturaKit

@available(macOS 14.0, iOS 17.0, tvOS 17.0, visionOS 1.0, watchOS 10.0, *)
public class MLXEmbedder {
  private let modelContainer: ModelContainer
  private let configuration: ModelConfiguration
  
  // Configuration for memory management
  private let maxBatchSize: Int
  private let defaultMaxLength: Int

  public init(configuration: ModelConfiguration = .nomic_text_v1_5, 
              maxBatchSize: Int = 16,
              defaultMaxLength: Int = 512) async throws {
    self.configuration = configuration
    self.maxBatchSize = maxBatchSize
    self.defaultMaxLength = defaultMaxLength
    self.modelContainer = try await MLXEmbedders.loadModelContainer(configuration: configuration)
  }

  public func embed(texts: [String]) async -> [[Float]] {
    // For large batches, process in chunks to reduce memory pressure
    if texts.count > maxBatchSize {
      var allEmbeddings: [[Float]] = []
      allEmbeddings.reserveCapacity(texts.count)
      
      for i in stride(from: 0, to: texts.count, by: maxBatchSize) {
        let endIdx = min(i + maxBatchSize, texts.count)
        let batchTexts = Array(texts[i..<endIdx])
        let batchEmbeddings = await embedBatch(texts: batchTexts)
        allEmbeddings.append(contentsOf: batchEmbeddings)
        
        // Allow any temporary tensors to be cleaned up between batches
        await Task.yield()
      }
      
      return allEmbeddings
    } else {
      return await embedBatch(texts: texts)
    }
  }
  
  private func embedBatch(texts: [String]) async -> [[Float]] {
    await modelContainer.perform { (model: EmbeddingModel, tokenizer, pooling) -> [[Float]] in
      let inputs = texts.map {
        tokenizer.encode(text: $0, addSpecialTokens: true)
      }

      // Calculate a reasonable maxLength but cap it
      var maxLength = inputs.reduce(into: 16) { acc, elem in
        acc = max(acc, elem.count)
      }
      // NOTE: Currently, documents exceeding the token limit are truncated, not split.
      // This means that if a document is longer than defaultMaxLength tokens, 
      // only the first defaultMaxLength tokens are processed, and the rest are discarded.
      //
      // TODO: Future enhancement - implement document splitting for large texts:
      // - Split texts exceeding maxTokenLength into multiple chunks with some overlap
      // - Process each chunk separately and store as separate documents or
      // - Generate embeddings for each chunk and average/combine them
      maxLength = min(maxLength, self.defaultMaxLength)

      // Process inputs in smaller groups if needed
      var paddedInputs: [MLXArray] = []
      paddedInputs.reserveCapacity(inputs.count)
      
      for elem in inputs {
        // More efficient array allocation - pre-allocate once
        var paddedElements = elem
        let paddingCount = maxLength - elem.count
        
        if paddingCount > 0 {
          paddedElements.reserveCapacity(maxLength)
          paddedElements.append(contentsOf: Array(
            repeating: tokenizer.eosTokenId ?? 0,
            count: paddingCount))
        }
        
        paddedInputs.append(MLXArray(paddedElements))
      }
      
      let padded = stacked(paddedInputs)
      
      // Release individual MLXArrays as they're no longer needed
      paddedInputs = []

      let mask = (padded .!= tokenizer.eosTokenId ?? 0)
      let tokenTypes = MLXArray.zeros(like: padded)

      let result = pooling(
        model(padded, positionIds: nil, tokenTypeIds: tokenTypes, attentionMask: mask),
        normalize: true, applyLayerNorm: true
      )
      
      // Directly return array conversion to avoid keeping extra MLXArrays in memory
      return result.map { $0.asArray(Float.self) }
    }
  }

  public func embed(text: String) async throws -> [Float] {
    let embeddings = await embed(texts: [text])
    return embeddings[0]
  }
}
