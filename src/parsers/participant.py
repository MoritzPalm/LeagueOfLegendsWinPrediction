import sqlalchemy.orm

from src.sqlstore.match import (
    SQLMatch,
    SQLParticipant,
    SQLParticipantStats,
    SQLStatPerk,
    SQLStyle,
    SQLStyleSelection,
    SQLChallenges,
)


def parse_participant_data(
    session: sqlalchemy.orm.Session, match: SQLMatch, participant: dict
) -> None:
    """
    parses participant stats and adds it to sqlalchemy session
    :param session: sqlalchemy orm session
    :param match: SQLMatch to get id for foreign key
    :param participant: list of dicts containing participant stats
    :return: None
    """
    with session.no_autoflush:
        participant_obj = SQLParticipant(
            puuid=participant["puuid"], participantId=participant["participantId"]
        )
        match.participant.append(participant_obj)
        session.add(participant_obj)
        participantStats_obj = SQLParticipantStats(**participant)
        participant_obj.stats.append(participantStats_obj)
        session.add(participantStats_obj)
        statPerks = participant["perks"]["statPerks"]
        participantPerk_obj = SQLStatPerk(
            participant["puuid"],
            statPerks["defense"],
            statPerks["flex"],
            statPerks["offense"],
        )
        participant_obj.statPerks.append(participantPerk_obj)
        session.add(participantPerk_obj)
        styles = participant["perks"]["styles"]
        for style in styles:
            participantStyle_obj = SQLStyle(style["description"], style["style"])
            participant_obj.styles.append(participantStyle_obj)
            for selection in style["selections"]:
                participantStyleSelection_obj = SQLStyleSelection(
                    selection["perk"],
                    selection["var1"],
                    selection["var2"],
                    selection["var3"],
                )
                participantStyle_obj.selection.append(participantStyleSelection_obj)
                session.add(participantStyleSelection_obj)
        participant["challenges"]["Assist12StreakCount"] = participant["challenges"][
            "12AssistStreakCount"
        ]  # rename
        participantChallenges_obj = SQLChallenges(**participant["challenges"])
        participant_obj.challenges.append(participantChallenges_obj)
        session.add(participantChallenges_obj)
