import re


def is_container_permit(objects):
    """
    Check whether permit is for a container based on the 'objecten' column.
    """
    container_words = [
        "puinbak",
        "container",
        "keet",
        "cabin",
    ]

    regex_pattern = re.compile(r"(?i)(" + "|".join(container_words) + r")")
    try:
        for obj in objects:
            if bool(regex_pattern.search(obj["object"])):
                return True
    except Exception as e:
        print(f"There was an exception in the is_container_permit function: {e}")

    return False


values_to_test = [
    {"object": "Schaftkeet"},
    {"object": "Mobiel toilet"},
    {"object": "Schaftkeet"},
    {"object": "Container overig"},
    {"object": "Container 1,8 x 6,5 m"},
    {"object": "Puin- of afvalcontainer"},
    {"object": "Bouwhekken"},
    {"object": "Puin- of afvalcontainer"},
    {"object": "Container overig"},
    {"object": "Container overig"},
    {"object": "Schaftkeet"},
    {"object": "Bouwhekken"},
    {"object": "Mobiel toilet"},
    {"object": "Mobiel toilet"},
    {"object": "Schaftkeet"},
    {"object": "Schaftkeet"},
]


def test_is_container_permit():
    results = []
    for value in values_to_test:
        result = is_container_permit([value])
        results.append(result)

    assert results == [
        True,
        False,
        True,
        True,
        True,
        True,
        False,
        True,
        True,
        True,
        True,
        False,
        False,
        False,
        True,
        True,
    ]
