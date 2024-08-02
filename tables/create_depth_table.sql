CREATE TABLE IF NOT EXISTS crypto_depth (
    id SERIAL PRIMARY KEY,
    lastUpdateId BIGINT,
    bids TEXT,
    asks TEXT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    processed BOOLEAN DEFAULT FALSE
);
