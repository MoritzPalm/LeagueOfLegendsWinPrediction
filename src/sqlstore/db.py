import logging

from sqlalchemy import create_engine, MetaData, URL
from sqlalchemy.orm import sessionmaker, declarative_base
from configparser import ConfigParser, Error

logger = logging.getLogger(__name__)



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
logger.info('Database connection established')
metadata = MetaData()
Base.metadata.create_all(bind=engine)
Session = sessionmaker(bind=engine)
session = Session()
