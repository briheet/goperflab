from typing import List, Optional
from openenv.core.env_server.types import Action, Observation, State


class GoPerfAction(Action):
    actionType: str


class GoPerfObservation(Observation):
    None


class GoPerfState(State):
    None
