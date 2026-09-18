"""Audit-folder name -> stable project slug.

Keyed by a distinctive substring rather than the full folder name: one of the
DAppSCAN folders contains a Cyrillic 'е' in "Runtime_Vеrification", and an
exact-match table silently drops that project on any machine that normalises
it differently.
"""
SLUGS = [
    ("LensProtocol",        "lens-protocol"),
    ("ElementFinance",      "element-finance"),
    ("88mph",               "88mph"),
    ("Opyn-Gamma",          "opyn-gamma"),
    ("PerpetualProtocolV2", "perp-v2-oracle"),
    ("Primitive",           "primitive-rmm"),
    ("Seaport",             "seaport"),
    ("UniswapV3Core",       "uniswap-v3-core"),
    ("YieldProtocol",       "yield-fydai"),
    ("EIP-4337",            "eip4337"),
    ("Notional",            "notional"),
    ("Ribbon",              "ribbon-v2"),
    ("Across",              "across-v2"),
]


def slug_for(folder: str):
    for key, slug in SLUGS:
        if key in folder:
            return slug
    return None


# Projects held out as the test split. Held out whole, so no contract from a
# test project can appear in training under a different file name.
TEST_PROJECTS = {"lens-protocol", "yield-fydai"}
VAL_PROJECTS = {"primitive-rmm", "perp-v2-oracle", "seaport"}
