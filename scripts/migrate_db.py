
import os
import sys
import sqlite3
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent))

def migrate():
    db_path = Path("data/app.db")
    if not db_path.exists():
        print(f"Database {db_path} not found. No migration needed.")
        return

    print(f"Migrating database: {db_path}")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        # Add arbiter_count and consensus_count to tasks table
        print("Adding consensus_count to tasks table...")
        cursor.execute("ALTER TABLE tasks ADD COLUMN consensus_count INTEGER DEFAULT 0")
        print("Adding arbiter_count to tasks table...")
        cursor.execute("ALTER TABLE tasks ADD COLUMN arbiter_count INTEGER DEFAULT 0")
        
        conn.commit()
        print("✅ Migration successful!")
    except sqlite3.OperationalError as e:
        if "duplicate column name" in str(e).lower():
            print("ℹ️ Columns already exist. Migration skipped.")
        else:
            print(f"❌ Migration error: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    migrate()
