from sqlalchemy.orm import mapped_column, relationship
from sqlalchemy import Integer, String, BigInteger, Boolean, ForeignKey, DateTime, Identity, Float
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

    def get_training_data(self):
        return {'level': self.summonerLevel}


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
               f"{self.queueType}"

    def get_training_data(self):
        return {
            'tier': self.tier,
            'rank': self.rank,
            'leaguePoints': self.leaguePoints,
            'wins': self.wins,
            'losses': self.losses,
            'veteran': self.veteran,
            'inactive': self.inactive,
            'freshBlood': self.freshBlood,
            'hotStreak': self.hotStreak
        }


class SQLChampionMastery(Base):
    __tablename__ = "summoner_champion_mastery"

    id = mapped_column(BigInteger, Identity(always=True), primary_key=True)
    puuid = mapped_column(String, ForeignKey("summoner.puuid"), nullable=False)
    summoner = relationship("SQLSummoner", backref="mastery")
    championId = mapped_column(BigInteger, ForeignKey("champion.id"), nullable=False)
    champion = relationship("SQLChampion", backref="mastery")
    championPointsUntilNextLevel = mapped_column(Integer)
    chestGranted = mapped_column(Boolean)
    lastPlayTime = mapped_column(BigInteger)  # in unix milliseconds time format
    championLevel = mapped_column(Integer)
    summonerId = mapped_column(String(63))
    championPoints = mapped_column(Integer)
    championPointsSinceLastLevel = mapped_column(Integer)
    tokensEarned = mapped_column(
        Integer)  # tokens earned for champion at current championLevel. Is reset to 0 after championLevel increase
    wins = mapped_column(Integer)
    loses = mapped_column(Integer)
    championWinrate = mapped_column(Float)
    kda = mapped_column(Float)
    kills = mapped_column(Float)
    deaths = mapped_column(Float)
    assists = mapped_column(Float)
    lp = mapped_column(Integer)
    maxKills = mapped_column(Integer)
    maxDeaths = mapped_column(Integer)
    cs = mapped_column(Float)  # averaged
    damage = mapped_column(Float)  # averaged
    gold = mapped_column(Float)  # averaged
    timeCreated = mapped_column(DateTime(timezone=True), server_default=func.now())
    lastUpdate = mapped_column(DateTime(timezone=True), onupdate=func.now())

    def __init__(self, championPointsUntilNextlevel: int, chestGranted: bool,
                 lastPlayTime: int, championLevel: int, summonerId: str, championPoints: int,
                 championPointsSinceLastLevel: int, tokensEarned: int, wins: str = None, loses: str = None,
                 championWinrate: float = None,
                 kda: str = None, kills: float = None, deaths: float = None, assists: float = None, lp: int = None,
                 maxKills: int = None,
                 maxDeaths: int = None,
                 cs: float = None, damage: float = None, gold: float = None):
        self.championPointsUntilNextLevel = championPointsUntilNextlevel
        self.chestGranted = chestGranted
        self.lastPlayTime = lastPlayTime
        self.championLevel = championLevel
        self.summonerId = summonerId
        self.championPoints = championPoints
        self.championPointsSinceLastLevel = championPointsSinceLastLevel
        self.tokensEarned = tokensEarned
        self.wins = wins
        self.loses = loses
        self.championWinrate = championWinrate
        self.kda = kda
        self.kills = kills
        self.deaths = deaths
        self.assists = assists
        self.lp = lp
        self.maxKills = maxKills
        self.maxDeaths = maxDeaths
        self.cs = cs
        self.damage = damage
        self.gold = gold

    def __repr__(self):
        return f"player {self.puuid} has level {self.championLevel} on champion {self.championId}"

    def get_training_data(self):
        return {
            'lastPlayTime': self.lastPlayTime,
            'championLevel': self.championLevel,
            'championPoints': self.championPoints,
            'championPointsSinceLastLevel': self.championPointsSinceLastLevel,
            'tokensEarned': self.tokensEarned,
            'wins': self.wins,
            'loses': self.loses,
            'championWinrate': self.championWinrate,
            'kda': self.kda,
            'kills': self.kills,
            'deaths': self.deaths,
            'assists': self.assists,
            'lp': self.lp,
            'maxKills': self.maxKills,
            'maxDeaths': self.maxDeaths,
            'cs': self.cs,
            'damage': self.damage,
            'gold': self.gold,
        }
