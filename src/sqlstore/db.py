import configparser
import contextlib
import logging
from configparser import ConfigParser

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql.psycopg2 import PGDialect_psycopg2
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import Session
from sqlalchemy.exc import OperationalError, ProgrammingError

logger = logging.getLogger(__name__)

Base = declarative_base()


def db_config(filename: str, section="postgresql") -> dict:
    db_configparser = ConfigParser()
    try:
        with open(filename) as f:
            logger.info("opening config file")
            db_configparser.read_file(f)
    except IOError:
        logger.critical("no database ini file found!")
        raise
    try:
        db = dict(db_configparser[section])
    except configparser.NoSectionError:
        logger.critical(f"Section {section} not found in file {filename}")
        raise
    return db


def connect_to_db(config: dict = None) -> sa.engine.base.Engine:
    """
    Connects to db and return the engine object
    If database does not exist, creates database
    """
    url_object = sa.URL.create(
        "postgresql+psycopg2",
        username=config["user"],
        password=config["password"],
        host=config["host"],
        database=config["database"],
    )
    logger.info(f"trying to create engine object with {url_object}")
    engine = sa.create_engine(url_object)
    logger.info("engine object created")
    if not database_exists(url_object):
        logger.critical(f"database {config['database']} does not exist")
    return engine


@contextlib.contextmanager
def get_session(engine: sa.Engine, cleanup=False) -> Session:
    session = Session(bind=engine)
    logger.info("building database session")
    Base.metadata.create_all(engine)

    try:
        yield session
    except sa.exc.SQLAlchemyError as e:
        logger.critical(str(e))
        session.rollback()
        raise
    finally:
        session.close()

    if cleanup:
        logger.info("dropping all tables")
        Base.metadata.drop_all(engine)


@contextlib.contextmanager
def get_conn(engine: sa.engine.base.Engine, cleanup=False):
    conn = engine.connect()
    logger.info("creating all tables")
    Base.metadata.create_all(engine)

    yield conn
    conn.close()

    if cleanup:
        logger.info("dropping all tables")
        Base.metadata.drop_all(engine)


def database_exists(url: sa.URL) -> bool:
    database = url.database
    dialect = url.get_dialect()
    engine = None
    try:
        if not issubclass(dialect, PGDialect_psycopg2):
            logger.critical(f"unsupported dialect {dialect}")
            raise ValueError("unsupported dialect")

        text = f"SELECT 1 FROM pg_database WHERE datname={database}"
        for _ in (database, 'postgres', None):
            engine = sa.create_engine(url, echo=True)
            try:
                with engine.connect() as conn:
                    result = conn.scalar(sa.text(text))
                return bool(result)
            except (ProgrammingError, OperationalError):
                pass
        return False

    finally:
        if engine:
            engine.dispose()
