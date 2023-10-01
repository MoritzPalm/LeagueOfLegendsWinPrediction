from typing import Any

from sqlalchemy import (
    Integer,
    String,
    Float,
    PickleType,
    DateTime,
    ForeignKey,
    Identity,
    BigInteger,
)
from sqlalchemy.orm import mapped_column, relationship, MappedColumn
from sqlalchemy.sql import func
from src.sqlstore.db import Base


class SQLChampion(Base):
    __tablename__ = "champion"

    id = mapped_column(BigInteger, Identity(always=True), primary_key=True)
    championNumber = mapped_column(Integer, nullable=False, index=True)
    seasonNumber = mapped_column(Integer, nullable=False, index=True)
    patchNumber = mapped_column(Integer, nullable=False, index=True)
    championName = mapped_column(String(100), nullable=False, index=True)
    championTitle = mapped_column(String(100))
    infoAttack = mapped_column(Integer)
    infoDefense = mapped_column(Integer)
    infoMagic = mapped_column(Integer)
    infoDifficulty = mapped_column(Integer)
    partype = mapped_column(
        String(150)
    )  # type of mana or energy (e.g. "Blood Well" for Aatrox)# TODO: fix partype
    # Maybe counters, abilities, Tier, maybe range, skill-shot-based, or not, cc-level.., trends in winrates,
    # role flexibility, new skin released (higher playrate)
    # -> this should not be saved in db, instead calculated server/analytics side imo
    tier = mapped_column(String(10), nullable=True)  # Represented as S,A,B,C,D,E, etc.
    win_rate = mapped_column(Float, nullable=True)  # Represented as a percent
    pick_rate = mapped_column(Float, nullable=True)  # Represented as a percent
    ban_rate = mapped_column(Float, nullable=True)  # Represented as a percent
    matches = mapped_column(Integer, nullable=True)  # Number of matches observed
    timeCreated = mapped_column(DateTime(timezone=True), server_default=func.now())
    lastUpdate = mapped_column(DateTime(timezone=True), onupdate=func.now())

    def __init__(
        self,
        championNumber: int,
        seasonNumber: int,
        patchNumber: int,
        championName: str,
        championTitle: str,
        infoAttack: int,
        infoDefense: int,
        infoMagic: int,
        infoDifficulty: int,
        partype: str = None,
        role: str = None,
        tier: str = None,
        win_rate: float = None,
        pick_rate: float = None,
        ban_rate: float = None,
        matches: int = None,
    ):
        """
        :param patchNumber: patch number identifying the data as champion data changes per patch
        :param seasonNumber: season number to uniquely identify the patch, see patchNumber
        :param championName: name of the champion
        :param championTitle:
        :param infoAttack:
        :param infoDefense:
        :param infoMagic:
        :param infoDifficulty:
        :param partype:
        :param role:
        :param tier:
        :param win_rate:
        :param pick_rate:
        :param ban_rate:
        :param matches:
        :return:
        """
        self.championNumber = championNumber
        self.championName = championName
        self.championTitle = championTitle
        self.infoAttack = infoAttack
        self.infoDefense = infoDefense
        self.infoMagic = infoMagic
        self.infoDifficulty = infoDifficulty
        self.partype = partype
        self.patchNumber = patchNumber
        self.seasonNumber = seasonNumber
        self.primaryRole = role
        self.tier = tier
        self.win_rate = win_rate
        self.pick_rate = pick_rate
        self.ban_rate = ban_rate
        self.matches = matches

    def repr(self):
        return f"<Champion {self.championName} ({self.key}) - {self.championTitle}>"

    def get_training_data(self):
        return {
            'champion_number': self.championNumber,
            'tier': self.tier,
            'win_rate': self.win_rate,
            'pick_rate': self.pick_rate,
            'ban_rate': self.ban_rate,
            'matches': self.matches
        }


class SQLChampionStats(Base):

    __tablename__ = "champion_stats"
    id = mapped_column(BigInteger, Identity(always=True), primary_key=True)
    championId = mapped_column(Integer, ForeignKey("champion.id"), nullable=False)
    champion = relationship("SQLChampion", backref="stats")
    patchNumber = mapped_column(Integer, nullable=False, index=True)
    seasonNumber = mapped_column(Integer, nullable=False, index=True)
    hp = mapped_column(Integer)
    hpperlevel = mapped_column(Integer)
    mp = mapped_column(Integer)
    mpperlevel = mapped_column(Integer)
    movespeed = mapped_column(Integer)
    armor = mapped_column(Integer)
    armorperlevel = mapped_column(Float)
    spellblock = mapped_column(Integer)
    spellblockperlevel = mapped_column(Float)
    attackrange = mapped_column(Integer)
    hpregen = mapped_column(Float)
    hpregenperlevel = mapped_column(Float)
    mpregen = mapped_column(Float)
    mpregenperlevel = mapped_column(Float)
    crit = mapped_column(Integer)
    critperlevel = mapped_column(Integer)
    attackdamage = mapped_column(Integer)
    attackdamageperlevel = mapped_column(Float)
    attackspeed = mapped_column(Float)
    timeCreated = mapped_column(DateTime(timezone=True), server_default=func.now())
    lastUpdate = mapped_column(DateTime(timezone=True), onupdate=func.now())

    def __init__(
        self,
        championId: int,
        patchNumber: int,
        seasonNumber: int,
        hp: int,
        hpperlevel: int,
        mp: int,
        mpperlevel: int,
        movespeed: int,
        armor: int,
        armorperlevel: float,
        spellblock: int,
        spellblockperlevel: float,
        attackrange: int,
        hpregen: float,
        hpregenperlevel: float,
        mpregen: float,
        mpregenperlevel: float,
        crit: int,
        critperlevel: int,
        attackdamage: int,
        attackdamageperlevel: float,
        attackspeed: float,
    ):
        self.championId = championId
        self.patchNumber = patchNumber
        self.seasonNumber = seasonNumber
        self.hp = hp
        self.hpperlevel = hpperlevel
        self.mp = mp
        self.mpperlevel = mpperlevel
        self.movespeed = movespeed
        self.armor = armor
        self.armorperlevel = armorperlevel
        self.spellblock = spellblock
        self.spellblockperlevel = spellblockperlevel
        self.attackrange = attackrange
        self.hpregen = hpregen
        self.hpregenperlevel = hpregenperlevel
        self.mpregen = mpregen
        self.mpregenperlevel = mpregenperlevel
        self.crit = crit
        self.critperlevel = critperlevel
        self.attackdamage = attackdamage
        self.attackdamageperlevel = attackdamageperlevel
        self.attackspeed = attackspeed

    def __repr__(self):
        return f"Champion stats of champion {self.championId}"


class SQLChampionRoles(Base):
    __tablename__ = "champion_roles"

    id = mapped_column(BigInteger, Identity(always=True), primary_key=True)
    championId = mapped_column(Integer, ForeignKey("champion.id"), nullable=False)
    champion = relationship("SQLChampion", backref="roles")
    role1 = mapped_column(String(20))
    role2 = mapped_column(String(20))
    role3 = mapped_column(String(20))
    timeCreated = mapped_column(DateTime(timezone=True), server_default=func.now())
    lastUpdate = mapped_column(DateTime(timezone=True), onupdate=func.now())

    def __init__(
        self, championId: int, role1: str, role2: str = None, role3: str = None
    ):
        self.championId = championId
        self.role1 = role1
        self.role2 = role2
        self.role3 = role3

    def __repr__(self):
        return f"champion {self.championId} with first role {self.role1}"


class SQLChampionTags(Base):
    __tablename__ = "champion_tags"

    id = mapped_column(BigInteger, Identity(always=True), primary_key=True)
    championId = mapped_column(BigInteger, ForeignKey("champion.id"), nullable=False)
    champion = relationship("SQLChampion", backref="tags")
    tag1 = mapped_column(String(20))
    tag2 = mapped_column(String(20))
    timeCreated = mapped_column(DateTime(timezone=True), server_default=func.now())
    lastUpdate = mapped_column(DateTime(timezone=True), onupdate=func.now())

    def __init__(self, championId: int, tag1, tag2):
        self.championId = championId
        self.tag1 = tag1
        self.tag2 = tag2

    def __repr__(self):
        return f"champion {self.championId} with first tag {self.tag1}"
