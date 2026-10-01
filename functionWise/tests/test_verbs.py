from functionWise import verbs as V


def test_function_name_from_tokenised_code():
    assert V.function_name("function mintToken ( address t ) public { }") == "mintToken"
    assert V.function_name("contract Foo { }") is None


def test_leading_verb_skips_noise_prefixes():
    assert V.leading_verb("disableInterface") == "disable"
    assert V.leading_verb("_setOwner") == "set"
    assert V.leading_verb("safeTransferFrom") == "transfer"
    assert V.leading_verb("doMint") == "mint"


def test_antonyms_are_bidirectional():
    assert "disable" in V.ANTONYMS["enable"]
    assert "enable" in V.ANTONYMS["disable"]
    assert "get" in V.ANTONYMS["set"]


def test_says_accepts_inflections_and_synonyms():
    assert V.says("Disables an interface.", "disable")
    assert V.says("Disabled the interface", "disable")
    assert V.says("Returns the owner", "get")
    assert not V.says("Returns the owner", "mint")


def test_inverts_only_when_the_verb_is_absent():
    # the real SmartDoc failures
    assert V.inverts("Enables an interface .", "disable")
    assert V.inverts("Retrieves the registration info .", "set")
    assert V.inverts("Set the price of a deed .", "get")
    # both present is a hedge, not an inversion
    assert not V.inverts("Disables an interface, undoing enable.", "disable")
    # unrelated wording is a miss, not an inversion
    assert not V.inverts("Changes the thing.", "mint")


def test_measure_counts_three_buckets():
    m = V.measure([("disable", "Enables an interface ."),
                   ("mint", "Mints new tokens ."),
                   ("burn", "Moves some value ."),
                   ("set", None)])
    assert m["n"] == 3
    assert m["named"] == 1
    assert m["inverted"] == 1
    assert m["examples"][0][0] == "disable"
