"""Initialize database tables for the trading engine."""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend.db.database import engine, Base

def init_db():
    print("Creating all database tables...")
    Base.metadata.create_all(bind=engine)
    print("Done. Tables created successfully.")

if __name__ == "__main__":
    init_db()
