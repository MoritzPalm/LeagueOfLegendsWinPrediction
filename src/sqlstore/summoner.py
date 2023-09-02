from sqlalchemy import Column, Integer, String, BigInteger, Boolean, ForeignKeyConstraint, DateTime
from sqlalchemy.sql import func
from src.sqlstore.db import Base


class SQLSummoner(Base):
    __tablename__ = "summoner"

    puuid = Column(String(78), primary_key=True)
    platformId = Column(String(7))
    summonerId = Column(String(63))
    accountId = Column(String(56))
    name = Column(String(60))
    summonerLevel = Column(Integer)
    timeCreated = Column(DateTime(timezone=True), server_default=func.now())
    lastUpdate = Column(DateTime(timezone=True), onupdate=func.now())

    def __init__(self, puuid: str, summonerId: str, accountId: str, name: str, summonerLevel: str):
        self.puuid = puuid
        self.summonerId = summonerId
        self.accountId = accountId
        self.name = name
        self.summonerLevel = summonerLevel

    def __repr__(self):
        return f"summoner {self.name} with puuid {self.puuid}"


class SQLSummonerLeague(Base):
    __tablename__ = "summoner_league"

    summonerId = Column(String(63), primary_key=True)
    platformId = Column(String(7), primary_key=True)
    leagueId = Column(String(70))
    queueType = Column(String(50))
    tier = Column(String(20))
    rank = Column(Integer)  # originally a string, is converted from roman numerals
    summonerName = Column(String(60))
    leaguePoints = Column(Integer)
    wins = Column(Integer)
    losses = Column(Integer)
    veteran = Column(Boolean)
    inactive = Column(Boolean)
    freshBlood = Column(Boolean)
    hotStreak = Column(Boolean)
    timeCreated = Column(DateTime(timezone=True), server_default=func.now())
    lastUpdate = Column(DateTime(timezone=True), onupdate=func.now())

    def __init__(self, summonerId: str, platformId: str, leagueId: str, queueType: str, tier: str, rank: int,
                 summonerName: str, leaguePoints: int, wins: int, losses: int, veteran: bool, inactive: bool,
                 freshBlood: bool, hotStreak: bool):
        self.summonerId = summonerId
        self.platformId = platformId
        self.leagueId = leagueId
        self.queueType = queueType
        self.tier = tier
        self.rank = rank
        self.summonerName = summonerName
        self.leaguePoints = leaguePoints
        self.wins = wins
        self.losses = losses
        self.veteran = veteran
        self.inactive = inactive
        self.freshBlood = freshBlood
        self.hotStreak = hotStreak

    def __repr__(self):
        return f"summoner {self.summonerName} with id {self.summonerId} has rank {self.tier} {self.rank} in queue " \
               f"{self.queueType} "


class SQLChampionMastery(Base):
    __tablename__ = "summoner_champion_mastery"

    puuid = Column(String(78), primary_key=True)
    championId = Column(Integer, primary_key=True)
    championPointsUntilNextLevel = Column(Integer)
    chestGranted = Column(Boolean)
    lastPlayTime = Column(Integer)  # in unix milliseconds time format
    championLevel = Column(Integer)
    summonerId = Column(String(63))
    championPoints = Column(Integer)
    championPointsSinceLastLevel = Column(Integer)
    tokensEarned = Column(Integer)  # tokens earned for champion at current championLevel. Is reset to 0 after championLevel increase
    timeCreated = Column(DateTime(timezone=True), server_default=func.now())
    lastUpdate = Column(DateTime(timezone=True), onupdate=func.now())

    def __init__(self, puuid: str, championId: str, championPointsUntilNextlevel: int, chestGranted: bool,
                 lastPlayTime: int, championLevel: int, summonerId: str, championPoints: int,
                 championPointsSinceLastLevel: int, tokensEarned: int):
        self.puuid = puuid
        self.championId = championId
        self.championPointsUntilNextLevel = championPointsUntilNextlevel
        self.chestGranted = chestGranted
        self.lastPlayTime = lastPlayTime
        self.championLevel = championLevel
        self.summonerId = summonerId
        self.championPoints = championPoints
        self.championPointsSinceLastLevel = championPointsSinceLastLevel

    def __repr__(self):
        return f"player {self.puuid} has level {self.championLevel} on champion {self.championId}"
