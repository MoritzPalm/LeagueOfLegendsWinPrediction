
from sqlalchemy import Table, Column, Integer, String, BigInteger, Boolean, ForeignKeyConstraint, Numeric, DateTime
from sqlalchemy.sql import func
from src.sqlstore.db import Base


class SQLmatch(Base):
    __tablename__ = "match"
    matchId = Column("matchId", String(20), primary_key=True)
    platformId = Column("platformId", String(7))
    gameId = Column("gameId", BigInteger)
    seasonId = Column("seasonId", Integer)
    queueId = Column("queueId", Integer)
    gameVersion = Column("gameVersion", String(23))
    mapId = Column("mapId", Integer)
    gameDuration = Column("gameDuration", Integer)
    gameCreation = Column("gameCreation", BigInteger)
    timeCreated = Column("timeCreated", DateTime(timezone=True), server_default=func.now())
    lastUpdate = Column("lastUpdate", DateTime(timezone=True), onupdate=func.now())

    def __init__(self, matchId: str, platformId: str, gameId: int, seasonId: int, queueId: int,
                 gameVersion: str, mapId: int, gameDuration: int, gameCreation: int):
        self.matchId = matchId
        self.platformId = platformId
        self.gameId = gameId
        self.seasonId = seasonId
        self.queueId = queueId
        self.gameVersion = gameVersion
        self.mapId = mapId
        self.gameDuration = gameDuration
        self.gameCreation = gameCreation

    def __repr__(self):
        return f'({self.platformId}) ({self.gameId}) ({self.gameCreation})'

