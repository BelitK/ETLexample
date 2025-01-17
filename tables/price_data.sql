CREATE TABLE IF NOT EXISTS crypto_price_data (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(50),
    price VARCHAR(50),
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    processed BOOLEAN DEFAULT FALSE
);