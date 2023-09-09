import logging

from riotwatcher import LolWatcher

logger = logging.getLogger(__name__)


class MatchIdCrawler:
    """An automatic crawler for Riot MatchIDs.
       The crawler runs ``riotwatcher.LolWatcher`` under the hood.

       Attributes
       ----------
       api_key : str
           Your Riot API key. You must have a valid API key to access
           information through the Riot API. Defaults to None.
       region : str
           The region of interest, defaults to None.
       tier : str
           Tier level of the matches, defaults to None.
       queue : str
           The queue type of the matches, defaults to None.

       Notes
       -----
           The available options for ``region`` are

           >>> ["br1", "eun1", "euw1", "jp1", "kr", "la1", "la2",
           ...  "na1", "oc1", "ru", "tr1"]

           The available options for ``tier`` are

           >>> ["CHALLENGER", "GRANDMASTER", "MASTER",
           ...  "DIAMOND", "EMERALD", "PLATINUM", "GOLD", "SILVER",
           ...  "BRONZE", "IRON"]

           The available options for ``queue`` are

           >>> ["RANKED_SOLO_5x5", "RANKED_FLEX_SR",
           ...  "RANKED_FLEX_TT"]
       """

    region_options_ = ["br1",
                       "eun1", "euw1",
                       "jp1", "kr",
                       "la1", "la2",
                       "na1",
                       "oc1",
                       "ru",
                       "tr1"]

    tier_options_ = ["CHALLENGER",
                     "GRANDMASTER",
                     "MASTER",
                     "DIAMOND",
                     "EMERALD",
                     "PLATINUM",
                     "GOLD",
                     "SILVER",
                     "BRONZE",
                     "IRON"]

    queue_options_ = ["RANKED_SOLO_5x5",
                      "RANKED_FLEX_SR",
                      "RANKED_FLEX_TT"]

    def __init__(self, api_key: str, region: str = "euw1", tier: str = "CHALLENGER", queue: str = "RANKED_SOLO_5x5"):
        # Error checking
        # api_key
        if type(api_key) != str:
            raise TypeError("Invalid API key.")
        else:
            self.api_key = api_key
        # region
        if type(region) != str:
            raise TypeError("Invalid type for region.")
        elif region not in self.region_options_:
            raise ValueError("Invalid value for region. Must be one of " +
                             f"{self.region_options_} (case sensitive).")
        else:
            self.region = region
        # tier
        if type(tier) != str:
            raise TypeError("Invalid type for tier.")
        elif tier not in self.tier_options_:
            raise ValueError("Invalid value for tier. Must be one of " +
                             f"{self.tier_options_} (case sensitive).")
        else:
            self.tier = tier
        # queue
        if type(queue) != str:
            raise TypeError("Invalid type for queue.")
        elif queue not in self.queue_options_:
            raise ValueError("Invalid value for queue. Must be one of " +
                             f"{self.queue_options_} (case sensitive).")
        else:
            self.queue = queue

        self.watcher = LolWatcher(api_key=self.api_key)

    def getMatchIDs(self, n: int, match_per_id: int = 15,
                    cutoff: int = 16, excludingIDs: set = None) -> set:
        """
        n : int
            Number of matchIDs to be returned. If not enough matches can be found,
            it logs a warning and return the (smaller than wanted) set of matchIDs
        match_per_id : int
            The number of matches to be crawled for each unique
            account. Recommend to be a minimum of 15. Will handle
            the case if a player have played for less than the
            specified number of matches. Defaults to 15.
        cutoff : int
            The minimum number of minutes required for a match to
            be counted toward the final list. Defaults to 16.
        excludingIDs : set
            The set of matchIDs that should not be included in the output.
            Usually used for successive runs, where already downloaded matches
            should not be processed twice.
            A list will also work, but is much slower, so a set is preferred
        :return: set of matchIDs (str)
        """
        # error checking
        if n <= 0:
            raise ValueError("Invalid number of matched to be crawled.")
        if match_per_id <= 0:
            raise ValueError("Invalid number of match per account.")
        if cutoff < 0:
            raise ValueError("Invalid cutoff.")
        # variable definition
        if not excludingIDs:
            exclude_IDs = False
        else:
            exclude_IDs = True

        # Fetch a set of leagueIds
        # For highest tiers - LeagueLists
        if self.tier in ["CHALLENGER", "GRANDMASTER", "MASTER"]:
            # For challengers
            if self.tier == "CHALLENGER":
                league_list = self.watcher.league \
                    .challenger_by_queue(self.region,
                                         self.queue)
                # For grandmasters
            elif self.tier == "GRANDMASTER":
                league_list = self.watcher.league \
                    .grandmaster_by_queue(self.region,
                                          self.queue)
            # For masters
            else:
                league_list = self.watcher.league \
                    .masters_by_queue(self.region,
                                      self.queue)
            leagueIds = {league_list["leagueId"]}
        # For all others - LeagueEntries
        else:
            league_entries_set = self.watcher.league \
                .entries(self.region, self.queue,
                         self.tier, "I")
            leagueIds = set([entry["leagueId"] for entry in league_entries_set])

        visited_matchIds = set()
        # Iterate over the leagueIds to fetch leagueEntries
        for leagueId in leagueIds:
            entries = self.watcher.league.by_id(self.region, leagueId)['entries']
            # Then fetch summonerIds for each LeagueEntry
            for entry in entries:
                summonerId = entry['summonerId']
                # Then fetch puuid for that summonerIds
                puuid = self.watcher.summoner.by_id(self.region, summonerId)["puuid"]
                # Then fetch a list of matchIds for that puuid
                match_list = self.watcher.match.matchlist_by_puuid(region=self.region, puuid=puuid, count=100, queue=420)
                for i in range(min(match_per_id, len(match_list))):
                    matchId = match_list[i]
                    if matchId in visited_matchIds:
                        continue
                    if exclude_IDs and matchId in excludingIDs:
                        continue
                    visited_matchIds.add(matchId)
                    if len(visited_matchIds) >= n:
                        break
                if len(visited_matchIds) >= n:
                    break
            if len(visited_matchIds) >= n:
                break
        return visited_matchIds
