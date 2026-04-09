from fastapi import APIRouter

from backend_app.api.llm_api.newUpdateFile.new_updateFile import new_updateFile_router
from backend_app.api.llm_api.universal.universal_router import universal_router
from backend_app.api.llm_api.ingest.ingest import ingest_router
from backend_app.api.voice.voice_router import voice_router
from backend_app.api.video.video_router import video_router

api_router = APIRouter()

api_router.include_router(new_updateFile_router, prefix="/new_update_file", tags=["new_update_file"])

api_router.include_router(ingest_router, prefix="/ingest", tags=["ingest"])

# universal enterprise assistant
api_router.include_router(universal_router, prefix="/universal", tags=["universal"])

# voice generation service
api_router.include_router(voice_router, prefix="/voice", tags=["voice"])

# video generation service
api_router.include_router(video_router, prefix="/video", tags=["video"])
