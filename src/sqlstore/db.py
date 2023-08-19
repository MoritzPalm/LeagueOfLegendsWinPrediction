import logging
import contextlib

from sqlalchemy import create_engine, MetaData, URL
from sqlalchemy.orm import sessionmaker, declarative_base, Session
from configparser import ConfigParser, Error

logger = logging.getLogger(__name__)

# TODO: logging
def db_config(filename='database.ini', section='postgresql'):
    # create a parser
    db_configparser = ConfigParser()
    # read config file
    db_configparser.read(filename)
    # get section, default to postgresql
    db = {}
    if db_configparser.has_section(section):
        params = db_configparser.items(section)
        for param in params:
            db[param[0]] = param[1]
    else:
        raise Exception('Section {0} not found in the {1} file'.format(section, filename))
    return db


def connect_to_db():
    """Connect to db and return the engine object"""
    config = db_config()
    url_object = URL.create('postgresql+psycopg2',
                            username=config['user'],
                            password=config['password'],
                            host=config['host'],
                            database=config['database'],
                            )
    return create_engine(url_object, echo=True)


Base = declarative_base()
engine = connect_to_db()


@contextlib.contextmanager
def get_session(cleanup=False):
    session = Session(bind=engine)
    Base.metadata.create_all(engine)

    try:
        yield session
    except Exception:   # TODO: this should be a more specific exception
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
