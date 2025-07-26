from prefect import flow, task
from src.train import train


@task
def run_training():
    train()


@flow
def main_flow():
    run_training()


if __name__ == "__main__":
    main_flow()
