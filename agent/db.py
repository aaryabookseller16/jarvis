import psycopg


def connect():
    """Open one connection to the jarvis database. Call once at startup."""
    return psycopg.connect("dbname=jarvis")


def insert_message(conn, conversation_id, role, kind, content):
    """Save one message. Not committed here: the caller commits or rolls back the whole turn."""
    conn.execute(
        "INSERT INTO messages (conversation_id, role, kind, content) "
        "VALUES (%s, %s, %s, %s)",
        (conversation_id, role, kind, content),
    )


def latest_or_new_conversation(conn):
    """Return the id of the most recent conversation, or create one if none exist."""
    row = conn.execute("""
    SELECT id FROM conversations
    ORDER BY id DESC
    LIMIT 1
    """).fetchone()

    if row is not None:
        return row[0]

    new_row = conn.execute(
        "INSERT INTO conversations DEFAULT VALUES RETURNING id"
    ).fetchone()
    conn.commit()
    return new_row[0]


def load_recent(conn, conversation_id, limit):
    """Return the last `limit` messages of a conversation, oldest first,
    as [{"role": ..., "content": ...}, ...], skipping failed_tool_call and correction rows."""
    rows = conn.execute("""
        SELECT role, content
        FROM messages
        WHERE conversation_id = %s
          AND kind NOT IN ('failed_tool_call', 'correction')
        ORDER BY id DESC
        LIMIT %s
    """, (conversation_id, limit)).fetchall()

    return [{"role": role, "content": content} for role, content in reversed(rows)]