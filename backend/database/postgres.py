import logging

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, declarative_base

from backend.config import get_settings

logger = logging.getLogger(__name__)

Base = declarative_base()

_engine = None
_SessionLocal = None


def _ensure_database_exists():
    """Create the target database if it doesn't exist (connects to default 'postgres' db)."""
    settings = get_settings()
    from urllib.parse import quote_plus
    pwd = quote_plus(settings.postgres_password)
    maintenance_url = f"postgresql://{settings.postgres_user}:{pwd}@{settings.postgres_host}:{settings.postgres_port}/postgres"
    try:
        eng = create_engine(maintenance_url, isolation_level="AUTOCOMMIT")
        with eng.connect() as conn:
            result = conn.execute(
                text("SELECT 1 FROM pg_database WHERE datname = :db"),
                {"db": settings.postgres_db},
            )
            if not result.fetchone():
                conn.execute(text(f'CREATE DATABASE "{settings.postgres_db}"'))
                logger.info("Created database '%s'", settings.postgres_db)
            else:
                logger.info("Database '%s' already exists", settings.postgres_db)
        eng.dispose()
    except Exception:
        logger.exception("Could not ensure database exists — will try to connect anyway")


def get_engine():
    global _engine
    if _engine is None:
        settings = get_settings()
        _engine = create_engine(
            settings.postgres_url,
            pool_pre_ping=True,
            pool_recycle=3600,
            pool_size=10,
            max_overflow=20,
            connect_args={"connect_timeout": 10},
            pool_timeout=15,
        )
    return _engine


def get_session_factory():
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(bind=get_engine(), autoflush=False, autocommit=False)
    return _SessionLocal


def get_db():
    """FastAPI dependency: yields a database session."""
    Session = get_session_factory()
    db = Session()
    try:
        yield db
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def _seed_admin():
    """Create the first admin user if ADMIN_USERNAME/ADMIN_PASSWORD are set and no users exist."""
    settings = get_settings()
    if not settings.admin_username or not settings.admin_password:
        return
    from backend.database.models import User
    from backend.auth import hash_password
    Session = get_session_factory()
    db = Session()
    try:
        if db.query(User).first() is not None:
            logger.info("Users already exist — skipping admin seed")
            return
        user = User(
            username=settings.admin_username.strip().lower(),
            password_hash=hash_password(settings.admin_password),
            role="admin",
        )
        db.add(user)
        db.commit()
        logger.info("Seeded admin user '%s'", user.username)
    except Exception:
        db.rollback()
        logger.exception("Failed to seed admin user")
    finally:
        db.close()


def init_db():
    """Create database (if needed), tables, and seed admin."""
    _ensure_database_exists()
    from backend.database.models import User, AuditLog  # noqa: F401
    Base.metadata.create_all(bind=get_engine())
    _seed_admin()
