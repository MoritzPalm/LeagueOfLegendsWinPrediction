
from sqlalchemy import Table, Column, Integer, String, BigInteger, Boolean, ForeignKeyConstraint, Numeric

from src.sqlstore.db import Base


class SQLmatch(Base):
    __tablename__ = "match"

    platformId = Column("platformId", String(7), primary_key=True)
    gameId = Column("gameId", BigInteger, primary_key=True)
    seasonId = Column("seasonId", Integer)
    queueId = Column("queueId", Integer)
    gameVersion = Column("gameVersion", String(23))
    mapId = Column("mapId", Integer)
    gameDuration = Column("gameDuration", Integer)
    gameCreation = Column("gameCreation", BigInteger)
    lastUpdate = Column("lastUpdate", BigInteger)

    def __init__(self, platformId, gameId, seasonId, queueId, gameVersion, mapId, gameDuration, gameCreation,
                 lastUpdate):
        self.platformId = platformId
        self.gameId = gameId
        self.seasonId = seasonId
        self.queueId = queueId
        self.gameVersion = gameVersion
        self.mapId = mapId
        self.gameDuration = gameDuration
        self.gameCreation = gameCreation
        self.lastUpdate = lastUpdate

    def __repr__(self):
        return f'({self.platformId}) ({self.gameId}) ({self.gameCreation})'

