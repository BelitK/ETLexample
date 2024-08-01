CREATE TABLE IF NOT EXISTS btcusdt_depth (
    id SERIAL PRIMARY KEY,
    last_update_id BIGINT,
    bids JSONB,
    asks JSONB,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    processed BOOLEAN DEFAULT FALSE
);
