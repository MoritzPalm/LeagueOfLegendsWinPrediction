from sqlalchemy import Integer, String, BigInteger, Boolean, ForeignKey, DateTime, Float, Identity
from sqlalchemy.orm import mapped_column, relationship
from sqlalchemy.sql import func
from src.sqlstore.db import Base


# TODO: should this be a dataclass?


class SQLParticipant(Base):
    __tablename__ = "participant"
    id = mapped_column(BigInteger, Identity(always=True), primary_key=True)
    puuid = mapped_column(String(78), nullable=False)
    matchId = mapped_column(BigInteger, ForeignKey("match.id"), nullable=False)
    match = relationship("SQLMatch", backref="participantStats")
    participantId = mapped_column(Integer)
    timeCreated = mapped_column(DateTime(timezone=True), server_default=func.now())
    lastUpdate = mapped_column(DateTime(timezone=True), onupdate=func.now())

    def __init__(self, puuid: str, participantId: int):
        self.puuid = puuid
        self.participantId = participantId

    def __repr__(self):
        return f"Participant {self.participantId} with puuid {self.puuid} in match {self.matchId}"


class SQLparticipantStats(Base):
    __tablename__ = "participant_stats"
    id = mapped_column(BigInteger, Identity(always=True), primary_key=True)
    participantId = mapped_column(BigInteger, ForeignKey("participant.id"), nullable=False)
    participant = relationship("SQLParticipant", backref="stats")
    allInPings = mapped_column(Integer)
    assistMePings = mapped_column(Integer)
    assists = mapped_column(Integer)
    baitPings = mapped_column(Integer)
    baronKills = mapped_column(Integer)
    bountyLevel = mapped_column(Integer)
    champExperience = mapped_column(Integer)
    champLevel = mapped_column(Integer)
    championId = mapped_column(Integer)
    championName = mapped_column(String(30))
    championTransform = mapped_column(Integer)
    commandPings = mapped_column(Integer)
    consumablesPurchased = mapped_column(Integer)
    damageDealtToBuildings = mapped_column(Integer)
    damageDealtToObjectives = mapped_column(Integer)
    damageDealtToTurrets = mapped_column(Integer)
    damageSelfMitigated = mapped_column(Integer)
    dangerPings = mapped_column(Integer)
    deaths = mapped_column(Integer)
    detectorWardsPlaced = mapped_column(Integer)
    doubleKills = mapped_column(Integer)
    dragonKills = mapped_column(Integer)
    eligibleForProgression = mapped_column(Boolean)
    enemyMissingPings = mapped_column(Integer)
    enemyVisionPings = mapped_column(Integer)
    firstBloodAssist = mapped_column(Boolean)
    firstBloodKill = mapped_column(Boolean)
    firstTowerAssist = mapped_column(Boolean)
    firstTowerKill = mapped_column(Boolean)
    gameEndedInEarlySurrender = mapped_column(Boolean)  # TODO: this info should be placed in a general team or match table
    gameEndedInSurrender = mapped_column(Boolean)  # TODO: this info should be placed in a general team or match table
    getBackPings = mapped_column(Integer)
    goldEarned = mapped_column(Integer)
    goldSpent = mapped_column(Integer)
    holdPings = mapped_column(Integer)
    individualPosition = mapped_column(String(10))
    inhibitorKills = mapped_column(Integer)
    inhibitorTakedowns = mapped_column(Integer)
    inhibitorsLost = mapped_column(Integer)
    item0 = mapped_column(Integer)
    item1 = mapped_column(Integer)
    item2 = mapped_column(Integer)
    item3 = mapped_column(Integer)
    item4 = mapped_column(Integer)
    item5 = mapped_column(Integer)
    item6 = mapped_column(Integer)
    itemsPurchased = mapped_column(Integer)
    killingSprees = mapped_column(Integer)
    kills = mapped_column(Integer)
    lane = mapped_column(String(20))
    largestCriticalStrike = mapped_column(Integer)
    largestKillingSpree = mapped_column(Integer)
    largestMultiKill = mapped_column(Integer)
    longestTimeSpentLiving = mapped_column(Integer)
    magicDamageDealt = mapped_column(Integer)
    magicDamageDealtToChampions = mapped_column(Integer)
    magicDamageTaken = mapped_column(Integer)
    needVisionPings = mapped_column(Integer)
    neutralMinionsKilled = mapped_column(Integer)
    nexusKills = mapped_column(Integer)  # This mapped_column is probably only important for special gamemodes, consider deleting it
    nexusLost = mapped_column(Integer)  # This mapped_column is probably only important for special gamemodes, consider deleting it
    nexusTakedowns = mapped_column(
        Integer)  # This mapped_column is probably only important for special gamemodes, consider deleting it
    objectivesStolen = mapped_column(Integer)
    objectivesStolenAssists = mapped_column(Integer)
    onMyWayPings = mapped_column(Integer)
    participantId = mapped_column(Integer, nullable=False)
    pentaKills = mapped_column(Integer)
    # TODO: in matchDto are perks, which do not translate well into this table, consider putting those in separate table
    physicalDamageDealt = mapped_column(Integer)
    physicalDamageDealtToChampions = mapped_column(Integer)
    physicalDamageTaken = mapped_column(Integer)
    placement = mapped_column(Integer)
    playerAugment0 = mapped_column(Integer)
    playerAugment1 = mapped_column(Integer)
    playerAugment2 = mapped_column(Integer)
    playerAugment3 = mapped_column(Integer)
    playerAugment4 = mapped_column(Integer)
    playerSubteamId = mapped_column(Integer)
    profileIcon = mapped_column(Integer)
    pushPings = mapped_column(Integer)
    quadraKills = mapped_column(Integer)
    riotIdName = mapped_column(String(120))
    riotIdTagLine = mapped_column(String(120))
    role = mapped_column(String(20))
    sightWardsBoughtInGame = mapped_column(Integer)
    spell1Casts = mapped_column(Integer)
    spell2Casts = mapped_column(Integer)
    spell3Casts = mapped_column(Integer)
    spell4Casts = mapped_column(Integer)
    subteamPlacement = mapped_column(Integer)
    summoner1Casts = mapped_column(Integer)
    summoner1Id = mapped_column(Integer)
    summoner2Casts = mapped_column(Integer)
    summoner2Id = mapped_column(Integer)
    summonerId = mapped_column(String(100))
    summonerLevel = mapped_column(Integer)
    summonerName = mapped_column(String(150))
    teamEarlySurrendered = mapped_column(Boolean)
    teamId = mapped_column(Integer)
    teamPosition = mapped_column(String(20))
    timeCCingOthers = mapped_column(Integer)
    timePlayed = mapped_column(Integer)
    totalAllyJungleMinionsKilled = mapped_column(Integer)
    totalDamageDealt = mapped_column(Integer)
    totalDamageDealtToChampions = mapped_column(Integer)
    totalDamageShieldedOnTeammates = mapped_column(Integer)
    totalDamageTaken = mapped_column(Integer)
    totalEnemyJungleMinionsKilled = mapped_column(Integer)
    totalHeal = mapped_column(Integer)
    totalHealOnTeammates = mapped_column(Integer)
    totalMinionsKilled = mapped_column(Integer)
    totalTimeCCDealt = mapped_column(Integer)
    totalTimeSpentDead = mapped_column(Integer)
    totalUnitsHealed = mapped_column(Integer)
    tripleKills = mapped_column(Integer)
    trueDamageDealt = mapped_column(Integer)
    trueDamageDealtToChampions = mapped_column(Integer)
    trueDamageTaken = mapped_column(Integer)
    turretKills = mapped_column(Integer)
    turretTakedowns = mapped_column(Integer)
    turretsLost = mapped_column(Integer)
    unrealKills = mapped_column(Integer)
    visionClearedPings = mapped_column(Integer)
    visionScore = mapped_column(Integer)
    visionWardsBoughtInGame = mapped_column(Integer)
    wardsKilled = mapped_column(Integer)
    wardsPlaced = mapped_column(Integer)
    win = mapped_column(Boolean)
    timeCreated = mapped_column(DateTime(timezone=True), server_default=func.now())
    lastUpdate = mapped_column(DateTime(timezone=True), onupdate=func.now())

    def __init__(self, **kwargs):  # TODO: include challenges and perks
        for attr in ('allInPings', 'assistMePings', 'assists', 'baitPings', 'baronKills', 'basicPings', 'bountyLevel',
                     'champExperience', 'champLevel', 'championId', 'championName', 'championTransform', 'commandPings',
                     'consumablesPurchased', 'damageDealtToBuildings', 'damageDealtToObjectives',
                     'damageDealtToTurrets',
                     'damageSelfMitigated', 'dangerPings', 'deaths', 'detectorWardsPlaced', 'doubleKills',
                     'dragonKills',
                     'eligibleForProgression', 'enemyMissingPings', 'enemyVisionPings', 'firstBloodAssist',
                     'firstBloodKill', 'firstTowerAssist', 'firstTowerKill', 'gameEndedInEarlySurrender',
                     'gameEndedInSurrender', 'getBackPings', 'goldEarned', 'goldSpent', 'holdPings',
                     'individualPosition', 'inhibitorKills', 'inhibitorTakedowns', 'inhibitorsLost', 'item0', 'item1',
                     'item2', 'item3', 'item4', 'item5', 'item6', 'itemsPurchased', 'killingSprees', 'kills', 'lane',
                     'largestCriticalStrike', 'largestKillingSpree', 'largestMultiKill', 'longestTimeSpentLiving',
                     'magicDamageDealt', 'magicDamageDealtToChampions', 'magicDamageTaken', 'needVisionPings',
                     'neutralMinionsKilled', 'nexusKills', 'nexusLost', 'nexusTakedowns', 'objectivesStolen',
                     'objectivesStolenAssists', 'onMyWayPings', 'participantId', 'pentaKills', 'physicalDamageDealt',
                     'physicalDamageDealtToChampions', 'physicalDamageTaken', 'placement', 'playerAugment1',
                     'playerAugment2', 'playerAugment3', 'playerAugment4', 'playerSubteamId', 'profileIcon',
                     'pushPings',
                     'puuid', 'quadraKills', 'riotIdName', 'riotIdTagline', 'role', 'sightWardsBoughtInGame',
                     'spell1Casts', 'spell2Casts', 'spell3Casts', 'spell4Casts', 'subteamPlacement', 'summoner1Casts',
                     'summoner1Id', 'summoner2Casts', 'summoner2Id', 'summonerId', 'summonerLevel', 'summonerName',
                     'teamEarlySurrendered', 'teamId', 'teamPosition', 'timeCCingOthers', 'timePlayed',
                     'totalAllyJungleMinionsKilled', 'totalDamageDealt', 'totalDamageDealtToChampions',
                     'totalDamageShieldedOnTeammates', 'totalDamageTaken', 'totalEnemyJungleMinionsKilled',
                     'totalHeal', 'totalHealsOnTeammates', 'totalMinionsKilled', 'totalTimeCCDealt',
                     'totalTimeSpentDead', 'totalUnitsHealed', 'tripleKills', 'trueDamageDealt',
                     'trueDamageDealtToChampions', 'trueDamageTaken', 'turretKills', 'turretTakedowns',
                     'turretsLost', 'unrealKills', 'visionClearedPings', 'visionScore', 'visionWardsBoughtInGame',
                     'wardsKilled', 'wardsPlaced', 'win',):
            setattr(self, attr, kwargs.get(attr))

    def __repr__(self):
        return f"{self.platformId}_{self.gameId} player {self.puuid} with number {self.participantId}"


class SQLStatPerk(Base):
    __tablename__ = "participant_perk"
    id = mapped_column(BigInteger, Identity(always=True), primary_key=True)
    participantId = mapped_column(BigInteger, ForeignKey("participant.id"), nullable=False)
    participant = relationship("SQLParticipant", backref="perk")
    defense = mapped_column(Integer)
    flex = mapped_column(Integer)
    offense = mapped_column(Integer)
    timeCreated = mapped_column(DateTime(timezone=True), server_default=func.now())
    lastUpdate = mapped_column(DateTime(timezone=True), onupdate=func.now())

    def __init(self, puuid: str, platformId: str, gameId: str, defense: int, flex: int, offense: int):
        self.puuid = puuid
        self.platformId = platformId
        self.gameId = gameId
        self.defense = defense
        self.flex = flex
        self.offense = offense

    def __repr__(self):
        return f"{self.platformId}_{self.gameId} player {self.puuid} stats"


class SQLStyle(Base):
    __tablename__ = "participant_style"
    id = mapped_column(BigInteger, Identity(always=True), primary_key=True)
    participantId = mapped_column(BigInteger, ForeignKey("participant.id"), nullable=False)
    participant = relationship("SQLParticipant", backref="style")
    description = mapped_column(String(80))
    style = mapped_column(Integer)
    timeCreated = mapped_column(DateTime(timezone=True), server_default=func.now())
    lastUpdate = mapped_column(DateTime(timezone=True), onupdate=func.now())

    def __init__(self, description: str, style: int):
        self.description = description
        self.style = style

    def __repr__(self):
        return f"Perk {self.description} by participant {self.participantId}"


class SQLStyleSelection(Base):
    __tablename__ = "style_selection"

    id = mapped_column(BigInteger, Identity(always=True), primary_key=True)
    styleId = mapped_column(BigInteger, ForeignKey("participant_style.id"), nullable=False)
    style = relationship("SQLStyle", backref="selection")
    perk = mapped_column(Integer)
    var1 = mapped_column(Integer)
    var2 = mapped_column(Integer)
    var3 = mapped_column(Integer)
    timeCreated = mapped_column(DateTime(timezone=True), server_default=func.now())
    lastUpdate = mapped_column(DateTime(timezone=True), onupdate=func.now())

    def __init__(self, perk: int, var1: int, var2: int, var3: int):
        self.perk = perk
        self.var1 = var1
        self.var2 = var2
        self.var3 = var3

    def __repr__(self):
        return f"Style selection (style id: {self.styleId}) perk: {self.perk}"


class SQLChallenges(Base):
    __tablename__ = "participant_challenges"

    id = mapped_column(BigInteger, Identity(always=True), primary_key=True)
    participantId = mapped_column(BigInteger, ForeignKey("participant.id"), nullable=False)
    participant = relationship("SQLParticipant", backref="challenges")
    Assist12StreakCount = mapped_column(Integer)
    abilityUses = mapped_column(Integer)
    acesBefore15Minutes = mapped_column(Integer)
    alliedJungleMonsterKills = mapped_column(Integer)
    baronBuffGoldAdvantageOverThreshold = mapped_column(Integer)
    baronTakedowns = mapped_column(Integer)
    blastConeOppositeOpponentCount = mapped_column(Integer)
    bountyGold = mapped_column(Integer)
    buffsStolen = mapped_column(Integer)
    completeSupportQuestOnTime = mapped_column(Boolean)
    controlWardsPlaced = mapped_column(Integer)
    damagePerMinute = mapped_column(Float)
    damageTakenOnTeamPercentage = mapped_column(Float)
    dancedWithRiftHerald = mapped_column(Integer)
    deathsByEnemyChamps = mapped_column(Integer)
    dodgeSkillShotsSmallWindow = mapped_column(Integer)
    doubleAces = mapped_column(Integer)
    dragonTakedowns = mapped_column(Integer)
    earliestBaron = mapped_column(Float)
    earlyLaningPhaseGoldExpAdvantage = mapped_column(Integer)
    effectiveHealAndShielding = mapped_column(Float)
    elderDragonKillsWithOpposingSoul = mapped_column(Integer)
    elderDragonMultikills = mapped_column(Integer)
    enemyChampionImmobilizations = mapped_column(Integer)
    enemyJungleMonsterKills = mapped_column(Integer)
    epicMonsterKillsNearEnemyJungler = mapped_column(Integer)
    epicMonsterKillsWithin30SecondsOfSpawn = mapped_column(Integer)
    epicMonsterSteals = mapped_column(Integer)
    epicMonsterStolenWithoutSmite = mapped_column(Integer)
    firstTurretKilled = mapped_column(Boolean)
    flawlessAces = mapped_column(Integer)
    fullTeamTakedown = mapped_column(Integer)
    gameLength = mapped_column(Float)
    getTakedownsInAllLanesEarlyJungleAsLaner = mapped_column(Boolean)
    goldPerMinute = mapped_column(Float)
    hadOpenNexus = mapped_column(Boolean)
    highestCrowdControlScore = mapped_column(Integer)
    immobilizeAndKillWithAlly = mapped_column(Integer)
    initialBuffCount = mapped_column(Integer)
    initialCrabCount = mapped_column(Integer)
    jungleCsBefore10Minutes = mapped_column(Integer)
    junglerTakedownsNearDamagedEpicMonster = mapped_column(Integer)
    kTurretsDestroyedBeforePlatesFall = mapped_column(Integer)
    kda = mapped_column(Float)
    killAfterHiddenWithAlly = mapped_column(Integer)
    killParticipation = mapped_column(Float)
    killedChampTookFullTeamDamageSurvived = mapped_column(Integer)
    killingSprees = mapped_column(Integer)
    killsNearEnemyTurret = mapped_column(Integer)
    killsOnOtherLanesEarlyJungleAsLaner = mapped_column(Integer)
    killsOnRecentlyHealedByAramPack = mapped_column(Integer)
    killsUnderOwnTurret = mapped_column(Integer)
    killsWithHelpFromEpicMonster = mapped_column(Integer)
    knockEnemyIntoTeamAndKill = mapped_column(Integer)
    landSkillShotsEarlyGame = mapped_column(Integer)
    laneMinionsFirst10Minutes = mapped_column(Integer)
    laningPhaseGoldExpAdvantage = mapped_column(Integer)
    legendaryCount = mapped_column(Integer)
    lostAnInhibitor = mapped_column(Boolean)   # TODO: investigate if this is really boolean
    maxCsAdvantageOnLaneOpponent = mapped_column(Integer)
    maxKillDeficit = mapped_column(Integer)
    maxLevelLeadLaneOpponent = mapped_column(Integer)
    mejaisFullStackInTime = mapped_column(Integer)
    moreEnemyJungleThanOpponent = mapped_column(Integer)
    multiKillOneSpell = mapped_column(Integer)
    multiTurretRiftHeraldCount = mapped_column(Integer)
    multikills = mapped_column(Integer)
    multikillsAfterAggressiveFlash = mapped_column(Integer)
    mythicItemUsed = mapped_column(Integer)
    outerTurretExecutesBefore10Minutes = mapped_column(Integer)
    outnumberedKills = mapped_column(Integer)
    outnumberedNexusKill = mapped_column(Boolean)
    perfectDragonSoulsTaken = mapped_column(Integer)
    perfectGame = mapped_column(Boolean)
    pickKillWithAlly = mapped_column(Integer)
    playedChampSelectPosition = mapped_column(Boolean)
    poroExplosions = mapped_column(Integer)
    quickCleanse = mapped_column(Integer)
    quickFirstTurret = mapped_column(Boolean)
    quickSoloKills = mapped_column(Integer)
    riftHeraldTakedowns = mapped_column(Integer)
    saveAllyFromDeath = mapped_column(Integer)
    scuttleCrabKills = mapped_column(Integer)
    shortestTimeToAceFromFirstTakedown = mapped_column(Float)
    skillshotsDodged = mapped_column(Integer)
    skillshotsHit = mapped_column(Integer)
    snowballsHit = mapped_column(Integer)
    soloBaronKills = mapped_column(Integer)
    soloKills = mapped_column(Integer)
    stealthWardsPlaced = mapped_column(Integer)
    survivedSingleDigitHpCount = mapped_column(Integer)
    survivedThreeImmobilizesInFight = mapped_column(Integer)
    takedownOnFirstTurret = mapped_column(Integer)
    takedowns = mapped_column(Integer)
    takedownsAfterGainingLevelAdvantage = mapped_column(Integer)
    takedownsBeforeJungleMinionSpawn = mapped_column(Integer)
    takedownsFirstXMinutes = mapped_column(Integer)  # TODO: investigate how many minutes (10?)
    takedownsInAlcove = mapped_column(Integer)
    takedownsInEnemyFountain = mapped_column(Integer)
    teamBaronKills = mapped_column(Integer)
    teamDamagePercentage = mapped_column(Float)
    teamElderDragonKills = mapped_column(Integer)
    teamRiftHeraldKills = mapped_column(Integer)
    tookLargeDamageSurvived = mapped_column(Integer)
    turretPlatesTaken = mapped_column(Integer)
    turretTakedowns = mapped_column(Integer)
    turretsTakenWithRiftHerald = mapped_column(Integer)
    twentyMinionsIn3SecondsCount = mapped_column(Integer)
    twoWardsOneSweeperCount = mapped_column(Integer)
    unseenRecalls = mapped_column(Integer)
    visionScoreAdvantageLaneOpponent = mapped_column(Float)
    visionScorePerMinute = mapped_column(Float)
    wardTakedowns = mapped_column(Integer)
    wardTakedownsBefore20M = mapped_column(Integer)
    wardsGuarded = mapped_column(Integer)
    timeCreated = mapped_column(DateTime(timezone=True), server_default=func.now())
    lastUpdate = mapped_column(DateTime(timezone=True), onupdate=func.now())

    def __init__(self, **kwargs):
        for attr in ('puuid', 'platformId', 'gameId', 'Assist12StreakCount', 'abilityUses', 'acesBefore15Minutes',
                     'alliedJungleMonsterKills', 'baronBuffGoldAdvantageOverThreshold', 'baronTakedowns',
                     'blastConeOppositeOpponentCount', 'bountyGold', 'buffsStolen', 'completeSupportQuestOnTime',
                     'controlWardsPlaced', 'damagePerMinute', 'damageTakenOnTeamPercentage', 'dancedWithRiftHerald',
                     'deathsByEnemyChamps', 'dodgeSkillShotsSmallWindow', 'doubleAces', 'dragonTakedowns',
                     'earliestBaron', 'earlyLaningPhaseGoldExpAdvantage', 'effectiveHealAndShielding',
                     'elderDragonKillsWithOpposingSoul', 'elderDragonMultikills', 'enemyChampionImmobilizations',
                     'enemyJungleMonsterKills', 'epicMonsterKillsNearEnemyJungler',
                     'epicMonsterKillsWithin30SecondsOfSpawn', 'epicMonsterSteals', 'epicMonsterStolenWithoutSmite',
                     'firstTurretKilled', 'flawlessAces', 'fullTeamTakedown', 'gameLength',
                     'getTakedownsInAllLanesEarlyJungleAsLaner', 'goldPerMinute', 'hadOpenNexus',
                     'highestCrowdControlScore', 'immobilizeAndKillWithAlly', 'initialBuffCount', 'initialCrabCount',
                     'jungleCsBefore10Minutes', 'junglerTakedownsNearDamagedEpicMonster',
                     'kTurretsDestroyedBeforePlatesFall', 'kda', 'killAfterHiddenWithAlly', 'killParticipation',
                     'killedChampTookFullTeamDamageSurvived', 'killingSprees', 'killsNearEnemyTurret',
                     'killsOnOtherLanesEarlyJungleAsLaner', 'killsOnRecentlyHealedByAramPack', 'killsUnderOwnTurret',
                     'killsWithHelpFromEpicMonster', 'knockEnemyIntoTeamAndKill', 'landSkillShotsEarlyGame',
                     'laneMinionsFirst10Minutes', 'laningPhaseGoldExpAdvantage', 'legendaryCount', 'lostAnInhibitor',
                     'maxCsAdvantageOnLaneOpponent', 'maxKillDeficit', 'maxLevelLeadLaneOpponent',
                     'mejaisFullStackInTime', 'moreEnemyJungleThanOpponent', 'multiKillOneSpell',
                     'multiTurretRiftHeraldCount', 'multikills', 'multikillsAfterAggressiveFlash', 'mythicItemUsed',
                     'outerTurretExecutesBefore10Minutes', 'outnumberedKills', 'outnumberedNexusKill',
                     'perfectDragonSoulsTaken', 'perfectGame', 'pickKillWithAlly', 'playedChampSelectPosition',
                     'poroExplosions', 'quickCleanse', 'quickFirstTurret', 'quickSoloKills', 'riftHeraldTakedowns',
                     'saveAllyFromDeath', 'scuttleCrabKills', 'shortestTimeToAceFromFirstTakedown', 'skillshotsDodged',
                     'skillshotsHit', 'snowballsHit', 'soloBaronKills', 'soloKills', 'stealthWardsPlaced',
                     'survivedSingleDigitHpCount', 'survivedThreeImmobilizesInFight', 'takedownOnFirstTurret',
                     'takedowns', 'takedownsAfterGainingLevelAdvantage', 'takedownsBeforeJungleMinionSpawn',
                     'takedownsFirstXMinutes', 'takedownsInAlcove', 'takedownsInEnemyFountain', 'teamBaronKills',
                     'teamDamagePercentage', 'teamElderDragonKills', 'teamRiftHeraldKills',
                     'tookLargeDamageSurvived', 'turretPlatesTaken', 'turretTakedowns', 'turretsTakenWithRiftHerald',
                     'twentyMinionsIn3SecondsCount', 'twoWardsOneSweeperCount', 'unseenRecalls',
                     'visionScoreAdvantageLaneOpponent', 'visionScorePerMinute', 'wardTakedowns',
                     'wardTakedownsBefore20M', 'wardsGuarded'):
            if attr in ['completeSupportQuestOnTime', 'firstTurretKilled', 'getTakedownsInAllLanesEarlyJungleAsLaner',
                        'hadOpenNexus', 'lostAnInhibitor', 'outnumberedNexusKill', 'perfectGame',
                        'playedChampSelectPosition', 'quickFirstTurret']:
                setattr(self, attr, bool(kwargs.get(attr)))
            else:
                setattr(self, attr, kwargs.get(attr))

    def __repr__(self):
        return f"challenges in match {self.platformId}_{self.gameId} of player {self.puuid} with number {self.participantId}"


