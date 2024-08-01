CREATE TABLE IF NOT EXISTS btcusdt_trades (
    id SERIAL PRIMARY KEY,
    trade_id BIGINT,
    price NUMERIC,
    qty NUMERIC,
    quote_qty NUMERIC,
    time TIMESTAMP,
    is_buyer_maker BOOLEAN,
    is_best_match BOOLEAN,
    processed BOOLEAN DEFAULT FALSE
);
