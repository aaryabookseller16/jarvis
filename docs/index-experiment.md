# Index experiment: last 20 messages of a conversation

Run on 2026-10-07 against PostgreSQL 18.6 (Homebrew, Apple Silicon), database
`jarvis`, using a throwaway `bench` table (dropped afterwards; the real
`messages` table was not touched).

**Query under test**

```sql
SELECT id, content FROM bench
WHERE conversation_id = 42
ORDER BY id DESC LIMIT 20;
```

**Data:** 1,000,000 rows, `conversation_id` uniform random over 1..10,000
(~100 rows per conversation, scattered across the table), `content` = 100 bytes.

## Predictions vs results

Numbers are from the second (warm cache) run of `EXPLAIN (ANALYZE, BUFFERS)`.

| Setup | Prediction (scan / pages / ms) | Actual scan type | Pages | Execution time |
|---|---|---|---|---|
| (a) no extra index | _TODO_ | Parallel Index Scan Backward on `bench_pkey`, `Filter: conversation_id = 42` | 11,120 | 25.9 ms |
| (b) `(conversation_id, id)` | _TODO_ | Index Scan Backward on `bench_conv_id_idx`, `Index Cond: conversation_id = 42` | 23 | 0.013 ms |
| (c) `(id, conversation_id)` | _TODO_ | Index Scan Backward on `bench_id_conv_idx`, `Index Cond: conversation_id = 42` | 586 | 2.9 ms |

Cold first runs, for reference:

| Setup | Pages (hit / read from disk) | Execution time |
|---|---|---|
| (a) | 10,981 / 0 | 39.6 ms |
| (b) | 20 / 3 | 0.061 ms |
| (c) | 23 / 566 | 7.2 ms |

## Costs

| Measure | Prediction | Actual |
|---|---|---|
| Table size | | 135 MB |
| Index size | _TODO_ | `bench_pkey` 21 MB, `(id, conversation_id)` 21 MB, `(conversation_id, id)` 24 MB (each ~16–18% of the table) |
| Insert 100k rows, PK only | _TODO_ | 282 ms |
| Insert 100k rows, PK + `(conversation_id, id)` | _TODO_ | 540 ms (~1.9× slower) |

The second insert ran on a table of 1.1M rows rather than 1M; that has little
effect, so most of the slowdown is maintaining the extra index.

## What happened

- **(a)** The planner used the primary key index walked backward (newest `id`
  first) and filtered out non-matching rows: ~165k `Rows Removed by Filter` per
  process, ~495k in total across 2 workers + leader. This is the "phone book
  sorted by the wrong thing" problem, live.
- **(b)** All of conversation 42's entries are adjacent in the index, so Postgres
  seeks to the end of that group and reads 20 entries. 23 pages: a few index
  pages plus roughly one heap page per row, since the rows are scattered in the
  table. ~2,000× faster than (a).
- **(c)** Still walks every `id` backward, but `conversation_id` is checked
  inside the index (`Index Cond`, not `Filter`), so non-matching rows never
  cost a heap fetch. Index pages hold ~300 entries vs ~70 rows per table page,
  hence 586 pages instead of 11k. ~9× faster than (a) but ~200× slower than (b),
  and it still scans ~50k entries to find 20 matches.

## Gaps between prediction and result

_TODO: one sentence per gap._

## Reproducing

```sql
\timing on

CREATE TABLE bench (
    id SERIAL PRIMARY KEY,
    conversation_id INT NOT NULL,
    content TEXT NOT NULL
);
INSERT INTO bench (conversation_id, content)
SELECT (random() * 9999)::int + 1, repeat('x', 100)
FROM generate_series(1, 1000000);
ANALYZE bench;

-- (a): run the EXPLAIN twice, use the second
EXPLAIN (ANALYZE, BUFFERS)
SELECT id, content FROM bench
WHERE conversation_id = 42
ORDER BY id DESC LIMIT 20;

-- (b)
CREATE INDEX bench_conv_id_idx ON bench (conversation_id, id);
ANALYZE bench;
-- rerun EXPLAIN twice

-- (c)
DROP INDEX bench_conv_id_idx;
CREATE INDEX bench_id_conv_idx ON bench (id, conversation_id);
ANALYZE bench;
-- rerun EXPLAIN twice

-- sizes
SELECT pg_size_pretty(pg_relation_size('bench')) AS table_size,
       pg_size_pretty(pg_indexes_size('bench'))  AS index_size;

-- insert cost
DROP INDEX bench_id_conv_idx;
INSERT INTO bench (conversation_id, content)
SELECT (random() * 9999)::int + 1, repeat('x', 100) FROM generate_series(1, 100000);
CREATE INDEX bench_conv_id_idx ON bench (conversation_id, id);
INSERT INTO bench (conversation_id, content)
SELECT (random() * 9999)::int + 1, repeat('x', 100) FROM generate_series(1, 100000);

DROP TABLE bench;
```
