"""Constants for the ROBIN rl algorithms package."""

TRAINER_SEED_RANK_MULTIPLIER = 1000
EVALUATOR_SEED_RANK_MULTIPLIER = 100_000

HIDDEN_SIZE = 256
LOG_STD_MAX = 2
LOG_STD_MIN = -5

IS_COOPERATIVE = {
    'iql_sac': False,
    'vdn': True
}
