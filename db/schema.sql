CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS users (
    user_id bigint PRIMARY KEY
);

CREATE TABLE IF NOT EXISTS photos (
    cloud_id bigint PRIMARY KEY GENERATED ALWAYS AS IDENTITY,
    user_id bigint NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    local_id text NOT NULL,
    file_path text NOT NULL,

    -- Qwen2-VL or external caption service output:
    -- {"A": "short visual caption", "B": "detailed visual caption"}
    caption jsonb,

    -- BGE-M3 text embeddings for caption rerank.
    -- These are not compatible with Chinese-CLIP image embeddings.
    caption_a_embedding vector(1024),
    caption_b_embedding vector(1024),
    caption_embedding_model text,

    -- Chinese-CLIP image embedding for first-stage vector retrieval.
    image_embedding vector(512) NOT NULL,

    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (user_id, local_id)
);

CREATE INDEX IF NOT EXISTS idx_photos_user_id ON photos(user_id);

CREATE INDEX IF NOT EXISTS idx_photos_image_embedding_hnsw
    ON photos USING hnsw (image_embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 200);

COMMENT ON COLUMN photos.image_embedding IS
    'Chinese-CLIP image embedding, 512 dimensions, L2-normalized.';
COMMENT ON COLUMN photos.caption_a_embedding IS
    'BGE-M3 text embedding for caption style A, 1024 dimensions, L2-normalized.';
COMMENT ON COLUMN photos.caption_b_embedding IS
    'BGE-M3 text embedding for caption style B, 1024 dimensions, L2-normalized.';
