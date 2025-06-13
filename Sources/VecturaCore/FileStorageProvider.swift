import Foundation

/// A file-based implementation of VecturaStorage.
///
/// This storage provider persists VecturaDocuments as JSON files in a specified directory.
/// Each document is stored as a separate JSON file named with its UUID.
public class FileStorageProvider: VecturaStorage {
    private let storageDirectory: URL
    
    /// Initializes a FileStorageProvider with the given storage directory.
    ///
    /// - Parameter storageDirectory: The directory where document files will be stored.
    /// - Throws: An error if the storage directory cannot be created.
    public init(storageDirectory: URL) throws {
        self.storageDirectory = storageDirectory
        try Self.createStorageDirectoryIfNeededSync(at: storageDirectory)
    }
    
    /// Creates the storage directory if it doesn't already exist (synchronous version for init).
    private static func createStorageDirectoryIfNeededSync(at url: URL) throws {
        if !FileManager.default.fileExists(atPath: url.path(percentEncoded: false)) {
            try FileManager.default.createDirectory(
                at: url,
                withIntermediateDirectories: true,
                attributes: nil
            )
        }
    }
    
    /// Creates the storage directory if it doesn't already exist.
    public func createStorageDirectoryIfNeeded() async throws {
        try Self.createStorageDirectoryIfNeededSync(at: storageDirectory)
    }
    
    /// Loads all documents from the storage directory.
    ///
    /// - Returns: An array of VecturaDocument objects loaded from JSON files.
    /// - Throws: VecturaError.loadFailed if any document files cannot be loaded.
    public func loadDocuments() async throws -> [VecturaDocument] {
        let fileURLs = try FileManager.default.contentsOfDirectory(
            at: storageDirectory, includingPropertiesForKeys: nil
        )
        
        let decoder = JSONDecoder()
        var documents: [VecturaDocument] = []
        var loadErrors: [String] = []
        
        for fileURL in fileURLs where fileURL.pathExtension == "json" {
            do {
                let data = try Data(contentsOf: fileURL)
                let document = try decoder.decode(VecturaDocument.self, from: data)
                documents.append(document)
            } catch {
                loadErrors.append(
                    "Failed to load \(fileURL.lastPathComponent): \(error.localizedDescription)"
                )
            }
        }
        
        if !loadErrors.isEmpty {
            throw VecturaError.loadFailed(loadErrors.joined(separator: "\n"))
        }
        
        return documents
    }
    
    /// Saves a document to the storage directory.
    ///
    /// - Parameter document: The document to save as a JSON file.
    /// - Throws: An error if the document cannot be encoded or written to disk.
    public func saveDocument(_ document: VecturaDocument) async throws {
        let documentURL = storageDirectory.appendingPathComponent("\(document.id).json")
        let encoder = JSONEncoder()
        encoder.outputFormatting = .prettyPrinted
        let data = try encoder.encode(document)
        try data.write(to: documentURL)
    }
    
    /// Deletes a document file from the storage directory.
    ///
    /// - Parameter id: The UUID of the document to delete.
    /// - Throws: An error if the file cannot be deleted.
    public func deleteDocument(withID id: UUID) async throws {
        let documentURL = storageDirectory.appendingPathComponent("\(id).json")
        try FileManager.default.removeItem(at: documentURL)
    }
    
    /// Updates an existing document by replacing the file.
    ///
    /// - Parameter document: The updated document to save.
    /// - Throws: An error if the document cannot be updated.
    public func updateDocument(_ document: VecturaDocument) async throws {
        // For file-based storage, updating is the same as saving
        try await saveDocument(document)
    }
} 