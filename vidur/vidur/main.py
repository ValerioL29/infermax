import pdb
from vidur.config import SimulationConfig
from vidur.simulator import Simulator
from vidur.utils.random import set_seeds


def main() -> None:
    config: SimulationConfig = SimulationConfig.create_from_cli_args()
    #pdb.set_trace()  # Breakpoint 1: Check config after creation

    set_seeds(config.seed)

    simulator = Simulator(config)
    #pdb.set_trace()  # Breakpoint 2: Check simulator before running
    simulator.run()


if __name__ == "__main__":
    main()
