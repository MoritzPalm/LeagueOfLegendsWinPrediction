import pickle

import sqlalchemy.orm

from src.sqlstore.timeline import (
    SQLTimeline,
    SQLFrame,
    SQLKillEvent,
    SQLTimelineDamageDealt,
    SQLTimelineDamageReceived,
    SQLEvent,
    SQLParticipantFrame,
)


def parse_timeline_data(
    session: sqlalchemy.orm.Session, platformId: str, gameId: int, timeline: dict
):
    current_timeline = SQLTimeline(
        platformId=platformId, gameId=gameId, frameInterval=timeline["frameInterval"]
    )
    session.add(current_timeline)
    for frameId, frame in enumerate(timeline["frames"]):
        frame_obj = SQLFrame(
            platformId=platformId,
            gameId=gameId,
            frameId=frameId,
            timestamp=timeline["frames"][frameId]["timestamp"],
        )
        current_timeline.frames.append(frame_obj)
        session.add(frame_obj)
        for eventId, event in enumerate(timeline["frames"][frameId]["events"]):
            if event["type"] in {
                "CHAMPION_KILL",
                "CHAMPION_SPECIAL_KILL",
                "TURRET_PLATE_DESTROYED",
                "BUILDING_KILL",
            }:
                assistingParticipantIds = pickle.dumps(
                    event.get("assistingParticipantIds"),
                    protocol=pickle.HIGHEST_PROTOCOL,
                )
                event_obj = SQLKillEvent(
                    assistingParticipantIds=assistingParticipantIds,
                    bounty=event.get("bounty"),
                    killStreakLength=event.get("killStreakLength"),
                    killerId=event.get("killerId"),
                    laneType=event.get("laneType"),
                    position=event.get("position"),
                    shutdownBounty=event.get("shutdownBounty"),
                    timestamp=event.get("timestamp"),
                    type=event.get("type"),
                    victimId=event.get("victimId"),
                )
                dmgDealt = event.get("victimDamageDealt")
                if dmgDealt is not None:
                    for dmg in dmgDealt:
                        dmgDealt_obj = SQLTimelineDamageDealt(
                            basic=dmg.get("basic"),
                            magicDamage=dmg.get("magicDamage"),
                            name=dmg.get("name"),
                            participantId=dmg.get("participantId"),
                            physicalDamage=dmg.get("physicalDamage"),
                            spellName=dmg.get("spellName"),
                            spellSlot=dmg.get("SpellSlot"),
                            trueDamage=dmg.get("trueDamage"),
                            type=dmg.get("type"),
                        )
                        event_obj.dmgdealt.append(dmgDealt_obj)
                dmgReceived = event.get("victimDamageReceived")
                if dmgReceived is not None:
                    for dmg in dmgReceived:
                        dmgReceived_obj = SQLTimelineDamageReceived(
                            basic=dmg.get("basic"),
                            magicDamage=dmg.get("magicDamage"),
                            name=dmg.get("name"),
                            participantId=dmg.get("participantId"),
                            physicalDamage=dmg.get("physicalDamage"),
                            spellName=dmg.get("spellName"),
                            spellSlot=dmg.get("SpellSlot"),
                            trueDamage=dmg.get("trueDamage"),
                            type=dmg.get("type"),
                        )
                        event_obj.dmgreceived.append(dmgReceived_obj)
                frame_obj.killevents.append(event_obj)
            else:
                event_obj = SQLEvent(
                    eventId=eventId,
                    timestamp=event.get("timestamp"),
                    type=event.get("type"),
                    participantId=event.get("participantId"),
                    itemId=event.get("itemId"),
                    skillSlot=event.get("skillSlot"),
                    creatorId=event.get("creatorId"),
                    teamId=event.get("teamId"),
                    afterId=event.get("afterId"),
                    beforeId=event.get("beforeId"),
                    wardType=event.get("wardType"),
                )
                frame_obj.events.append(event_obj)
            session.add(event_obj)
        for i, participantFrame in enumerate(
            timeline["frames"][frameId]["participantFrames"].items(), start=1
        ):
            participantFrameData = participantFrame[1]
            participantFrameData["platformId"] = platformId
            participantFrameData["gameId"] = gameId
            participantFrameData["frameId"] = frameId
            participantFrameData["participantId"] = i
            participantFrame_obj = SQLParticipantFrame(**participantFrameData)
            frame_obj.participantframe.append(participantFrame_obj)
            session.add(participantFrame_obj)
