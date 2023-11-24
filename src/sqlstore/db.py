import configparser
import contextlib
import logging
from configparser import ConfigParser

from sqlalchemy import create_engine, URL, exc
from sqlalchemy.orm import declarative_base, Session

logger = logging.getLogger(__name__)


def db_config(filename="src/database.ini", section="postgresql") -> dict:
    db_configparser = ConfigParser()
    try:
        with open(filename) as f:
            logger.info("opening config file")
            db_configparser.read_file(f)
    except IOError:
        logger.critical(f"no dabase ini file found!")
        raise
    try:
        db = dict(db_configparser[section])
    except configparser.NoSectionError:
        logger.critical(f"Section {section} not found in file {filename}")
        raise
    return db


def connect_to_db():
    """Connect to db and return the engine object"""
    config: dict = db_config()
    url_object = URL.create(
        "postgresql+psycopg2",
        username=config["user"],
        password=config["password"],
        host=config["host"],
        database=config["database"],
    )
    logger.info(f"creating engine object with {url_object}")
    return create_engine(url_object, pool_size=20, max_overflow=10)


Base = declarative_base()
engine = connect_to_db()


@contextlib.contextmanager
def get_session(cleanup=False) -> Session:
    session = Session(bind=engine)
    logger.info(f"building database session")
    Base.metadata.create_all(engine)

    try:
        yield session
    except exc.SQLAlchemyError as e:
        logger.critical(str(e))
        session.rollback()
        raise
    finally:
        session.close()

    if cleanup:
        logger.info("dropping all tables")
        Base.metadata.drop_all(engine)


@contextlib.contextmanager
def get_conn(cleanup=False):
    conn = engine.connect()
    logger.info("creating all tables")
    Base.metadata.create_all(engine)

    yield conn
    conn.close()

    if cleanup:
        logger.info("dropping all tables")
        Base.metadata.drop_all(engine)
