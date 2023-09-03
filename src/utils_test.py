import utils


def test_parse_game_version():
    matches = utils.parse_game_version("13.16.525.6443")
    assert int(matches.group(1)) == 13
    assert int(matches.group(2)) == 16


def test_get_season():
    assert utils.get_season("13.16.525.6443") == 13


def test_get_patch():
    assert utils.get_patch("13.16.525.6443") == 16


def test_separate_MatchID():
    platformId, gameId = utils.separateMatchID("EUW1_6577607618")
    assert platformId == "EUW1"
    assert gameId == 6577607618
