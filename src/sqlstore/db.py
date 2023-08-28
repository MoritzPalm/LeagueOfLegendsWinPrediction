import configparser
import logging
import contextlib

from sqlalchemy import create_engine, MetaData, URL, exc
from sqlalchemy.orm import sessionmaker, declarative_base, Session
from configparser import ConfigParser, Error

logger = logging.getLogger(__name__)

# TODO: logging


def db_config(filename='database.ini', section='postgresql') -> dict:
    db_configparser = ConfigParser()
    try:
        with open(filename) as f:
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
    url_object = URL.create('postgresql+psycopg2',
                            username=config['user'],
                            password=config['password'],
                            host=config['host'],
                            database=config['database'],
                            )
    return create_engine(url_object, echo="debug")


Base = declarative_base()
engine = connect_to_db()


@contextlib.contextmanager
def get_session(cleanup=False):
    session = Session(bind=engine)
    Base.metadata.create_all(engine)

    try:
        yield session
    except exc.SQLAlchemyError as e:
        logger.critical(e)
        session.rollback()
    finally:
        session.close()

    if cleanup:
        Base.metadata.drop_all(engine)


@contextlib.contextmanager
def get_conn(cleanup=False):
    conn = engine.connect()
    Base.metadata.create_all(engine)

    yield conn
    conn.close()

    if cleanup:
        Base.metadata.drop_all(engine)
