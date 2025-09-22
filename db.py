import os
from sqlalchemy import event
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy import Column, Integer, String, Float, DateTime, select, Boolean, Text
from sqlalchemy.sql import func

from core.log_config import logger

# 定义数据库模型
Base = declarative_base()

class RequestStat(Base):
    __tablename__ = 'request_stats'
    id = Column(Integer, primary_key=True)
    request_id = Column(String)
    endpoint = Column(String)
    client_ip = Column(String)
    process_time = Column(Float)
    first_response_time = Column(Float)
    provider = Column(String, index=True)
    model = Column(String, index=True)
    api_key = Column(String, index=True)
    is_flagged = Column(Boolean, default=False)
    text = Column(Text)
    prompt_tokens = Column(Integer, default=0)
    completion_tokens = Column(Integer, default=0)
    total_tokens = Column(Integer, default=0)
    timestamp = Column(DateTime(timezone=True), server_default=func.now(), index=True)

class ChannelStat(Base):
    __tablename__ = 'channel_stats'
    id = Column(Integer, primary_key=True)
    request_id = Column(String)
    provider = Column(String, index=True)
    model = Column(String, index=True)
    api_key = Column(String)
    provider_api_key = Column(String, nullable=True, index=True)
    success = Column(Boolean, default=False)
    timestamp = Column(DateTime(timezone=True), server_default=func.now(), index=True)

DISABLE_DATABASE = os.getenv("DISABLE_DATABASE", "false").lower() == "true"
db_engine = None
async_session = None

if not DISABLE_DATABASE:
    DB_TYPE = os.getenv("DB_TYPE", "sqlite").lower()
    is_debug = bool(os.getenv("DEBUG", False))
    logger.info(f"Using {DB_TYPE} database.")

    if DB_TYPE == "postgres":
        try:
            import asyncpg
        except ImportError:
            raise ImportError("asyncpg is not installed. Please install it with 'pip install asyncpg' to use PostgreSQL.")

        DB_USER = os.getenv("DB_USER", "postgres")
        DB_PASSWORD = os.getenv("DB_PASSWORD", "mysecretpassword")
        DB_HOST = os.getenv("DB_HOST", "localhost")
        DB_PORT = os.getenv("DB_PORT", "5432")
        DB_NAME = os.getenv("DB_NAME", "postgres")

        db_url = f"postgresql+asyncpg://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
        db_engine = create_async_engine(db_url, echo=is_debug)

    elif DB_TYPE == "sqlite":
        db_path = os.getenv('DB_PATH', './data/stats.db')
        data_dir = os.path.dirname(db_path)
        os.makedirs(data_dir, exist_ok=True)
        db_engine = create_async_engine('sqlite+aiosqlite:///' + db_path, echo=is_debug)

        @event.listens_for(db_engine.sync_engine, "connect")
        def set_sqlite_pragma_on_connect(dbapi_connection, connection_record):
            cursor = None
            try:
                cursor = dbapi_connection.cursor()
                cursor.execute("PRAGMA journal_mode=WAL;")
                cursor.execute("PRAGMA busy_timeout = 5000;")
            except Exception as e:
                logger.error(f"Failed to set PRAGMA for SQLite: {e}")
            finally:
                if cursor:
                    cursor.close()
    else:
        raise ValueError(f"Unsupported DB_TYPE: {DB_TYPE}. Please use 'sqlite' or 'postgres'.")

    async_session = sessionmaker(db_engine, class_=AsyncSession, expire_on_commit=False)
