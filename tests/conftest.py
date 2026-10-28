def pytest_addoption(parser):
    parser.addoption(
        '--regen-references',
        action='store_true',
        help='Rewrite the stored results in tests/reference/ instead of comparing against them.',
    )
