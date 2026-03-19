from fastapi import APIRouter


from backend_app.api.llm_api.newChat.new_chat import new_chat_router
from backend_app.api.llm_api.newUpdateFile.new_updateFile import new_updateFile_router
from backend_app.api.llm_api.enterprise.enterprise_router import enterprise_router
from backend_app.api.llm_api.travel.travel_router import travel_router
from backend_app.api.llm_api.universal.universal_router import universal_router
from backend_app.api.llm_api.geo.geo_router import geo_router
from backend_app.api.voice.voice_router import voice_router
from backend_app.api.video.video_router import video_router

api_router = APIRouter()


# new chat
api_router.include_router(new_chat_router, prefix="/new_chat", tags=["new_chat"])

api_router.include_router(new_updateFile_router, prefix="/new_update_file", tags=["new_update_file"])

# enterprise RAG assistant
api_router.include_router(enterprise_router, prefix="/enterprise", tags=["enterprise"])

# travel route planner assistant
api_router.include_router(travel_router, prefix="/travel", tags=["travel"])

# universal enterprise assistant
api_router.include_router(universal_router, prefix="/universal", tags=["universal"])

# voice generation service
api_router.include_router(voice_router, prefix="/voice", tags=["voice"])

# video generation service
api_router.include_router(video_router, prefix="/video", tags=["video"])

# GEO optimization assistant
api_router.include_router(geo_router, prefix="/geo", tags=["geo"])