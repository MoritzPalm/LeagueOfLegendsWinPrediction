from sqlalchemy import Column, Integer, String, Float, Text, CheckConstraint
from sqlalchemy.sql import func
from src.sqlstore.db import Base


class SQLChampion(Base):
    tablename = "champion"

    championId = Column(Integer, primary_key=True)
    championName = Column(String(100))
    championTitle = Column(String(100))
    # championLore = Column(Text)  # Text() for long descriptions # TODO: is this necessary considering the disk usage?
    skillLevel = Column(Integer, CheckConstraint('skillLevel>=1 AND skillLevel<=5'), nullable=True)
    patchWinRate = Column(Float, nullable=True)  # Represented as a percent
    patchPlayRate = Column(Float, nullable=True)  # Represented as a percent
    patchNumber = Column(String(20), nullable=True)  # Representing which game patch
    role = Column(String(50), nullable=True)  # Top, Mid...
    # Maybe counters, abilities, Tier, maybe range, skill-shot-based, or not, cc-level.., trends in winrates, role flexibility, new skin released (higher playrate)

    def init(self, championId, championName, championTitle, championLore=None, skillLevel=None,
                 patchWinRate=None, patchPlayRate=None, patchNumber=None, role=None):
        self.championId = championId
        self.championName = championName
        self.championTitle = championTitle
        self.championLore = championLore
        self.skillLevel = skillLevel
        self.patchWinRate = patchWinRate
        self.patchPlayRate = patchPlayRate
        self.patchNumber = patchNumber
        self.role = role

    def repr(self):
        return f"<Champion {self.championName} - {self.championTitle}>"