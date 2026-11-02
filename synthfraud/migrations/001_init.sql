-- One row per sampling request: who drew how many rows from which model, under what privacy budget.
-- Sampling from a differentially private model is post-processing and spends no further budget;
-- the log is for accountability, not for budget accounting.
CREATE TABLE generations (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    requested_by TEXT NOT NULL,
    model TEXT NOT NULL,
    model_sha256 TEXT NOT NULL,
    rows_requested BIGINT NOT NULL,
    fraud_type TEXT,
    epsilon TEXT NOT NULL
);
CREATE INDEX generations_created ON generations (created_at);
