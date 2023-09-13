import logging

import sqlalchemy.exc
from sqlalchemy.orm import Session
from sqlalchemy import exists

from src.sqlstore.champion import SQLChampion

logger = logging.getLogger(__file__)




