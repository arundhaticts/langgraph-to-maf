"""Test-suite noise -- must be EXCLUDED from extraction/orchestration."""


def test_login_success():
    assert 1 + 1 == 2


def helper_used_only_in_tests(x):
    return x * 2
