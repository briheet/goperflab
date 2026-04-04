from openenv.core.client_types import StepResult
from openenv.core.env_client import EnvClient

from models import GoPerfAction, GoPerfObservation, GoPerfState


class GoPerfEnv(EnvClient[GoPerfAction, GoPerfObservation, GoPerfState]):
    def _step_payload(self, action: GoPerfAction) -> dict:
        return action.model_dump(exclude_none=True)

    def _parse_result(self, payload: dict) -> StepResult:
        obs_data = payload.get("observation", {})
        observation = GoPerfObservation(
            **obs_data,
            done=payload.get("done", False),
            reward=payload.get("reward"),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: dict) -> GoPerfState:
        return GoPerfState(**payload)
