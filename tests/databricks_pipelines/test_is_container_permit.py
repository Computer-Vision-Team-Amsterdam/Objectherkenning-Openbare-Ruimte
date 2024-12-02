from objectherkenning_openbare_ruimte.databricks_pipelines.post_processing_pipeline.data_enrichment_step.components.decos_data_connector import (
    DecosDataHandler,
)

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
        result = DecosDataHandler.is_container_permit([value])
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
