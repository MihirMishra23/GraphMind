def compare_kg_to_world_state(world_kg: WorldKG, world_state: dict) -> dict:
    """
    Compute simple metrics:
    - Precision/recall of object locations.
    - Accuracy of door states (open/closed/locked).
    - Accuracy of inventory.
    Return metrics as a dict.
    """


def evaluate_agent_over_episodes(env_factory, num_episodes: int = 5) -> dict:
    """
    Run multiple episodes, aggregate:
    - Average total reward
    - Average steps per episode
    - KG accuracy metrics (when available)
    """
