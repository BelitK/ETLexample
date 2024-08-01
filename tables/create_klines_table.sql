CREATE TABLE IF NOT EXISTS btcusdt_klines (
    id SERIAL PRIMARY KEY,
    open_time TIMESTAMP,
    open_price NUMERIC,
    high_price NUMERIC,
    low_price NUMERIC,
    close_price NUMERIC,
    volume NUMERIC,
    close_time TIMESTAMP,
    quote_asset_volume NUMERIC,
    number_of_trades INT,
    taker_buy_base_asset_volume NUMERIC,
    taker_buy_quote_asset_volume NUMERIC,
    ignore VARCHAR(255),
    processed BOOLEAN DEFAULT FALSE
);
