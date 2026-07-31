from typing import Literal

EnvironmentName = Literal[
    "CartPole-v1",
    "Acrobot-v1",
    "LunarLander-v3",
    "MountainCar-v0",
    "MinAtar/Breakout-v1",
]

# Kept as an alias so the existing scripts do not need a broad type-only rename.
ClassicControl = EnvironmentName

AgentTypes = Literal["vanilla", "cql", "ddqn"]
ScoringMethod = Literal["monte_carlo", "td"]
RepresentationMethod = Literal["pca", "input_pca", "q_values"]
