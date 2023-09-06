import configparser
import logging
import contextlib

from sqlalchemy import create_engine, URL, exc
from sqlalchemy.orm import declarative_base, Session
from configparser import ConfigParser

logger = logging.getLogger(__name__)


def db_config(filename='src/database.ini', section='postgresql') -> dict:
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


def connect_to_db(filename):
    """Connect to db and return the engine object"""
    config: dict = db_config(filename)
    url_object = URL.create('postgresql+psycopg2',
                            username=config['user'],
                            password=config['password'],
                            host=config['host'],
                            database=config['database'],
                            )
    logger.info(f"creating engine object with {url_object}")
    return create_engine(url_object)


Base = declarative_base()
engine = connect_to_db(filename='../src/database.ini')


@contextlib.contextmanager
def get_session(cleanup=False):
    session = Session(bind=engine)
    logger.info(f"creating all tables")
    Base.metadata.create_all(engine)

    try:
        yield session
    except exc.SQLAlchemyError as e:
        logger.critical(e)
        session.rollback()
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
