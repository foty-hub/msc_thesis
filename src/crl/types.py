from typing import Literal

ClassicControl = Literal[
    "CartPole-v1", "Acrobot-v1", "LunarLander-v3", "MountainCar-v0"
]

AgentTypes = Literal["vanilla", "cql", "ddqn"]
ScoringMethod = Literal["monte_carlo", "td"]
CalibMethods = Literal["nocalib", "ccdisc", "ccnn"]
