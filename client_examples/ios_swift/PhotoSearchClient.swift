import Foundation

struct UploadResponse: Decodable {
    let cloudId: Int
    let userId: Int
    let localId: String
    let imageEmbeddingDim: Int

    enum CodingKeys: String, CodingKey {
        case cloudId = "cloud_id"
        case userId = "user_id"
        case localId = "local_id"
        case imageEmbeddingDim = "image_embedding_dim"
    }
}

struct SearchResponse: Decodable {
    let query: String
    let elapsedMs: Double
    let results: [SearchResult]

    enum CodingKeys: String, CodingKey {
        case query
        case elapsedMs = "elapsed_ms"
        case results
    }
}

struct SearchResult: Decodable {
    let cloudId: Int
    let localId: String
    let filePath: String
    let similarityScore: Double
    let captionSim: Double?
    let rerankScore: Double?
    let caption: String?

    enum CodingKeys: String, CodingKey {
        case cloudId = "cloud_id"
        case localId = "local_id"
        case filePath = "file_path"
        case similarityScore = "similarity_score"
        case captionSim = "caption_sim"
        case rerankScore = "rerank_score"
        case caption
    }
}

final class PhotoSearchClient {
    private let baseURL: URL

    init(baseURL: URL = URL(string: "http://localhost:8000")!) {
        self.baseURL = baseURL
    }

    func uploadPhoto(jpegData: Data, userId: Int, localId: String) async throws -> UploadResponse {
        let url = baseURL.appendingPathComponent("/api/photos/upload")
        var request = URLRequest(url: url)
        request.httpMethod = "POST"

        let boundary = "Boundary-\(UUID().uuidString)"
        request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")

        var body = Data()
        body.appendFormField(name: "user_id", value: "\(userId)", boundary: boundary)
        body.appendFormField(name: "local_id", value: localId, boundary: boundary)
        body.appendFileField(name: "file", filename: "photo.jpg", mimeType: "image/jpeg", data: jpegData, boundary: boundary)
        body.append("--\(boundary)--
".data(using: .utf8)!)
        request.httpBody = body

        let (data, _) = try await URLSession.shared.data(for: request)
        return try JSONDecoder().decode(UploadResponse.self, from: data)
    }

    func attachCaptions(cloudId: Int, captionA: String?, captionB: String?) async throws {
        let url = baseURL.appendingPathComponent("/api/photos/\(cloudId)/captions")
        var request = URLRequest(url: url)
        request.httpMethod = "POST"

        let boundary = "Boundary-\(UUID().uuidString)"
        request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")

        var body = Data()
        if let captionA { body.appendFormField(name: "caption_a", value: captionA, boundary: boundary) }
        if let captionB { body.appendFormField(name: "caption_b", value: captionB, boundary: boundary) }
        body.append("--\(boundary)--
".data(using: .utf8)!)
        request.httpBody = body

        _ = try await URLSession.shared.data(for: request)
    }

    func search(query: String, userId: Int, limit: Int = 10) async throws -> SearchResponse {
        let url = baseURL.appendingPathComponent("/api/photos/search")
        var request = URLRequest(url: url)
        request.httpMethod = "POST"

        let boundary = "Boundary-\(UUID().uuidString)"
        request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")

        var body = Data()
        body.appendFormField(name: "query", value: query, boundary: boundary)
        body.appendFormField(name: "user_id", value: "\(userId)", boundary: boundary)
        body.appendFormField(name: "limit", value: "\(limit)", boundary: boundary)
        body.appendFormField(name: "rerank", value: "true", boundary: boundary)
        body.appendFormField(name: "rerank_k", value: "50", boundary: boundary)
        body.appendFormField(name: "alpha", value: "0.7", boundary: boundary)
        body.appendFormField(name: "caption_style", value: "A", boundary: boundary)
        body.append("--\(boundary)--
".data(using: .utf8)!)
        request.httpBody = body

        let (data, _) = try await URLSession.shared.data(for: request)
        return try JSONDecoder().decode(SearchResponse.self, from: data)
    }
}

private extension Data {
    mutating func appendFormField(name: String, value: String, boundary: String) {
        append("--\(boundary)
".data(using: .utf8)!)
        append("Content-Disposition: form-data; name="\(name)"

".data(using: .utf8)!)
        append("\(value)
".data(using: .utf8)!)
    }

    mutating func appendFileField(name: String, filename: String, mimeType: String, data: Data, boundary: String) {
        append("--\(boundary)
".data(using: .utf8)!)
        append("Content-Disposition: form-data; name="\(name)"; filename="\(filename)"
".data(using: .utf8)!)
        append("Content-Type: \(mimeType)

".data(using: .utf8)!)
        append(data)
        append("
".data(using: .utf8)!)
    }
}
