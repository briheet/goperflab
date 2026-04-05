from openenv.core.env_server import create_fastapi_app
import uvicorn

from .env import GoPerfEnvironment
from models import GoPerfAction, GoPerfObservation

app = create_fastapi_app(GoPerfEnvironment, GoPerfAction, GoPerfObservation)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


def main() -> None:
    uvicorn.run("server.app:app", host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
