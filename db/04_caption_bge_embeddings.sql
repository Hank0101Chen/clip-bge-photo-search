CREATE EXTENSION IF NOT EXISTS vector;

ALTER TABLE photos
    ADD COLUMN IF NOT EXISTS caption_a_embedding vector(1024),
    ADD COLUMN IF NOT EXISTS caption_b_embedding vector(1024),
    ADD COLUMN IF NOT EXISTS caption_embedding_model text;

COMMENT ON COLUMN photos.caption_a_embedding IS
    'BGE-M3 text embedding for caption style A, 1024 dimensions, L2-normalized.';
COMMENT ON COLUMN photos.caption_b_embedding IS
    'BGE-M3 text embedding for caption style B, 1024 dimensions, L2-normalized.';
