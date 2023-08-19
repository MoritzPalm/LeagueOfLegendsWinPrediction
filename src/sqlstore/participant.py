from sqlalchemy import Table, Column, Integer, String, BigInteger, Boolean, ForeignKeyConstraint, Numeric, DateTime
from sqlalchemy.sql import func
from src.sqlstore.db import Base


class SQLparticipant(Base):
    __tablename__ = "match_participant"

    participantId = Column("participantId", Integer, primary_key=True)
    platformId = Column("platformId", String(7), primary_key=True)
    gameId = Column("gameId", BigInteger, primary_key=True)
    championId = Column("championId", Integer)
    teamId = Column("teamID", Integer)
    spell1Id = Column("spell1Id", Integer)
    spell2Id = Column("spell2Id", Integer)

    def __init__(self, participantId: int, platformId: str, gameId: int, championId: int,
                 teamId: int, spell1Id: int, spell2Id: int):
        self.participantId = participantId
        self.platformId = platformId
        self.gameId = gameId
        self.championId = championId
        self.teamId = teamId
        self.spell1Id = spell1Id
        self.spell2Id = spell2Id

    def __repr__(self):
        return f"({self.platformId}) ({self.gameId}) {self.participantId} playing champion {self.championId}"

