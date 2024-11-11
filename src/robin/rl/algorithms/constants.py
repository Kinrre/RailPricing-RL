"""Constants for the ROBIN rl algorithms package."""

DEFAULT_OUTPUT_DIR = 'models'

HIDDEN_SIZE = 256
LOG_STD_MAX = 2
LOG_STD_MIN = -5

IS_COOPERATIVE = {
    'iql_sac': False,
    'vdn': True
}
