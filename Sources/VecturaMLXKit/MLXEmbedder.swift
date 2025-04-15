import Foundation
import MLX
import MLXEmbedders
import VecturaKit

@available(macOS 14.0, iOS 17.0, tvOS 17.0, visionOS 1.0, watchOS 10.0, *)
public class MLXEmbedder: @unchecked Sendable {
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
      maxLength = min(maxLength, self.defaultMaxLength)
      
      // Pad and stack in one operation
      let padded = stacked(
        inputs.map { elem in
          MLXArray(
            elem + Array(
              repeating: tokenizer.eosTokenId ?? 0,
              count: maxLength - elem.count))
        })
      
      let mask = (padded .!= tokenizer.eosTokenId ?? 0)
      let tokenTypes = MLXArray.zeros(like: padded)

      // Use the pooling with eval() 
      let result = pooling(
        model(padded, positionIds: nil, tokenTypeIds: tokenTypes, attentionMask: mask),
        normalize: true, applyLayerNorm: true
      ).eval()
      
      // Convert to Float arrays - handle the result appropriately based on its type
      if let multiDimensionalArray = result as? [MLXArray] {
          return multiDimensionalArray.map { $0.asArray(Float.self) }
      } else if let singleArray = result as? MLXArray {
          // If it's a single MLXArray that contains multiple embeddings
          return singleArray.dimensions.count > 1 
              ? (0..<singleArray.shape[0]).map { singleArray[$0].asArray(Float.self) }
              : [singleArray.asArray(Float.self)]
      } else {
          // Fallback to empty result if the structure is unexpected
          return []
      }
    }
  }

  public func embed(text: String) async throws -> [Float] {
    let embeddings = await embed(texts: [text])
    return embeddings[0]
  }
}
