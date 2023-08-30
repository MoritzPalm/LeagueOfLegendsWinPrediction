from sqlalchemy import Column, Integer, String, Float, Text, CheckConstraint, PickleType
from sqlalchemy.sql import func
from src.sqlstore.db import Base


class SQLChampion(Base):
    __tablename__ = "champion"

    championId = Column(Integer, primary_key=True)
    championName = Column(String(100))
    championTitle = Column(String(100))
    infoAttack = Column(Integer)
    infoDefense = Column(Integer)
    infoMagic = Column(Integer)
    infoDifficulty = Column(Integer)
    tags = Column(PickleType)   # serialized list of tags (eg. [Marksman, Support] for Ashe)
    partype = Column(String(150))   # type of mana or energy (eg. "Blood Well" for Aatrox)
    patchWinRate = Column(Float, nullable=True)  # Represented as a percent
    patchPlayRate = Column(Float, nullable=True)  # Represented as a percent
    patchNumber = Column(Integer, nullable=True)    # this may be problematic if saving data from multiple seasons , consider adding season as column
    # TODO: should this be part of the primary key?
    primaryRole = Column(String(50), nullable=True)  # Top, Mid...
    # Maybe counters, abilities, Tier, maybe range, skill-shot-based, or not, cc-level.., trends in winrates,
    # role flexibility, new skin released (higher playrate)
    # -> this should not be saved in db, instead calculated server/analytics side imo

    def init(self, championId: int, championName: str, championTitle: str, infoAttack: int, infoDefense: int,
             infoMagic: int, infoDifficulty: int, tags, partype: str, patchWinRate: float = None,
             patchPlayRate: float = None, patchNumber: float = None, role: str = None):
        """

        :param championId:
        :param championName:
        :param championTitle:
        :param infoAttack:
        :param infoDefense:
        :param infoMagic:
        :param infoDifficulty:
        :param tags:
        :param partype:
        :param patchWinRate:
        :param patchPlayRate:
        :param patchNumber:
        :param role:
        :return:
        """
        self.championId = championId
        self.championName = championName
        self.championTitle = championTitle
        self.infoAttack = infoAttack
        self.infoDefense = infoDefense
        self.infoMagic = infoMagic
        self.infoDifficulty = infoDifficulty
        self.tags = tags
        self.partype = partype
        self.patchWinRate = patchWinRate
        self.patchPlayRate = patchPlayRate
        self.patchNumber = patchNumber
        self.primaryRole = role

    def repr(self):
        return f"<Champion {self.championName} ({self.key}) - {self.championTitle}>"


class SQLChampionStats(Base):

    __tablename__ = "champion_stats"

    championId = Column(Integer, primary_key=True)
    hp = Column(Integer)
    hpperlevel = Column(Integer)
    mp = Column(Integer)
    mpperlevel = Column(Integer)
    movespeed = Column(Integer)
    armor = Column(Integer)
    armorperlevel = Column(Float)
    spellblock = Column(Integer)
    spellblockperlevel = Column(Float)
    attackrange = Column(Integer)
    hpregen = Column(Float)
    hpregenperlevel = Column(Float)
    mpregen = Column(Float)
    mpregenperlevel = Column(Float)
    crit = Column(Integer)
    critperlevel = Column(Integer)
    attackdamage = Column(Integer)
    attackdamageperlevel = Column(Float)
    attackspeed = Column(Float)

    def __init__(self, championId: int, hp: int, hpperlevel: int, mp: int, mpperlevel: int, movespeed: int, armor: int,
                 armorperlevel: float, spellblock: int, spellblockperlevel: float, attackrange: int, hpregen: float,
                 hpregenperlevel: float, mpregen: float, mpregenperlevel: float, crit: int, critperlevel: int,
                 attackdamage: int, attackdamageperlevel: float, attackspeed: float):
        self.championId = championId
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
