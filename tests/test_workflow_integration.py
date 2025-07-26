from prefect.testing.utilities import prefect_test_harness
from src.workflow import main_flow


def test_workflow_integration():
    with prefect_test_harness():
        main_flow()
