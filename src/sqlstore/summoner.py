from sqlalchemy.orm import mapped_column, relationship
from sqlalchemy import Integer, String, BigInteger, Boolean, ForeignKey, DateTime, Identity
from sqlalchemy.sql import func
import roman
from src.sqlstore.db import Base


class SQLSummoner(Base):
    __tablename__ = "summoner"

    puuid = mapped_column(String(78), primary_key=True)
    platformId = mapped_column(String(7), index=True)
    summonerId = mapped_column(String(63), index=True)
    accountId = mapped_column(String(56))
    name = mapped_column(String(100))
    summonerLevel = mapped_column(Integer)
    timeCreated = mapped_column(DateTime(timezone=True), server_default=func.now())
    lastUpdate = mapped_column(DateTime(timezone=True), onupdate=func.now())

    def __init__(self, puuid: str, platformId: str, summonerId: str, accountId: str, name: str, summonerLevel: str):
        self.puuid = puuid
        self.platformId = platformId
        self.summonerId = summonerId
        self.accountId = accountId
        self.name = name
        self.summonerLevel = summonerLevel

    def __repr__(self):
        return f"summoner {self.name} with puuid {self.puuid}"


class SQLSummonerLeague(Base):
    __tablename__ = "summoner_league"

    id = mapped_column(BigInteger, Identity(always=True), primary_key=True)
    puuid = mapped_column(String, ForeignKey("summoner.puuid"), nullable=False)
    summoner = relationship("SQLSummoner", backref="leagues")
    leagueId = mapped_column(String(70))
    queueType = mapped_column(String(50))
    tier = mapped_column(String(20), index=True)
    rank = mapped_column(Integer, index=True)  # originally a string, is converted from roman numerals
    summonerName = mapped_column(String(60))
    leaguePoints = mapped_column(Integer)
    wins = mapped_column(Integer)
    losses = mapped_column(Integer)
    veteran = mapped_column(Boolean)
    inactive = mapped_column(Boolean)
    freshBlood = mapped_column(Boolean)
    hotStreak = mapped_column(Boolean)
    timeCreated = mapped_column(DateTime(timezone=True), server_default=func.now())
    lastUpdate = mapped_column(DateTime(timezone=True), onupdate=func.now())

    def __init__(self, leagueId: str, queueType: str, tier: str, rank: int,
                 summonerName: str, leaguePoints: int, wins: int, losses: int, veteran: bool, inactive: bool,
                 freshBlood: bool, hotStreak: bool):
        self.leagueId = leagueId
        self.queueType = queueType
        self.tier = tier
        self.rank = roman.fromRoman(rank)
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

    id = mapped_column(BigInteger, Identity(always=True), primary_key=True)
    puuid = mapped_column(String, ForeignKey("summoner.puuid"), nullable=False)
    summoner = relationship("SQLSummoner", backref="mastery")
    championId = mapped_column(BigInteger, ForeignKey("champion.id"), nullable=False)
    champion = relationship("SQLChampion", backref="mastery")
    championPointsUntilNextLevel = mapped_column(Integer)
    chestGranted = mapped_column(Boolean)
    lastPlayTime = mapped_column(Integer)  # in unix milliseconds time format
    championLevel = mapped_column(Integer)
    summonerId = mapped_column(String(63))
    championPoints = mapped_column(Integer)
    championPointsSinceLastLevel = mapped_column(Integer)
    tokensEarned = mapped_column(Integer)  # tokens earned for champion at current championLevel. Is reset to 0 after championLevel increase
    timeCreated = mapped_column(DateTime(timezone=True), server_default=func.now())
    lastUpdate = mapped_column(DateTime(timezone=True), onupdate=func.now())
    winsLoses = mapped_column(String)
    championWinrate = mapped_column(String)
    kda = mapped_column(String)
    killsDeathsAssists = mapped_column(String)
    lp = mapped_column(String)
    maxKills = mapped_column(String)
    maxDeaths = mapped_column(String)
    cs = mapped_column(String)
    damage = mapped_column(String)
    gold = mapped_column(String)

    def __init__(self, championPointsUntilNextlevel: int, chestGranted: bool,
                 lastPlayTime: int, championLevel: int, summonerId: str, championPoints: int,
                 championPointsSinceLastLevel: int, tokensEarned: int, winsLoses: str, championWinrate: str,
                 kda: str, killsDeathsAssists: str, lp: str, maxKills: str, maxDeaths: str,
                 cs : str, damage: str, gold: str):
        self.championPointsUntilNextLevel = championPointsUntilNextlevel
        self.chestGranted = chestGranted
        self.lastPlayTime = lastPlayTime
        self.championLevel = championLevel
        self.summonerId = summonerId
        self.championPoints = championPoints
        self.championPointsSinceLastLevel = championPointsSinceLastLevel
        self.tokensEarned = tokensEarned
        self.winsLoses = winsLoses
        self.championWinrate = championWinrate
        self.kda = kda
        self.killsDeathsAssists = killsDeathsAssists
        self.lp = lp
        self.maxKills = maxKills
        self.maxDeaths = maxDeaths
        self.cs = cs
        self.damage = damage
        self.gold = gold

    def __repr__(self):
        return f"player {self.puuid} has level {self.championLevel} on champion {self.championId}"
