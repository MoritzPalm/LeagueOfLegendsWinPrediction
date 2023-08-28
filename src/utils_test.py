import utils


def test_get_season():
    assert utils.get_season("13.16.525.6443") == 13
