CREATE TABLE IF NOT EXISTS btcusdt_aggtrades (
    id SERIAL PRIMARY KEY,
    agg_trade_id BIGINT,
    price NUMERIC,
    qty NUMERIC,
    first_trade_id BIGINT,
    last_trade_id BIGINT,
    time TIMESTAMP,
    is_buyer_maker BOOLEAN,
    processed BOOLEAN DEFAULT FALSE
);
