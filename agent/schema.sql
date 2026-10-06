CREATE TABLE conversations (
    id SERIAL PRIMARY KEY,
    title TEXT DEFAULT 'untitled',
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE messages (
    id SERIAL PRIMARY KEY,
    conversation_id INT NOT NULL REFERENCES conversations(id) ON DELETE RESTRICT,
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    kind TEXT NOT NULL CHECK (kind IN ('user_input', 'tool_call', 'failed_tool_call',
                                       'observation', 'correction', 'final_answer')),
    content TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX messages_conversation_id_id_idx ON messages (conversation_id, id);