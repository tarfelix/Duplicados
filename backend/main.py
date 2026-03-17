import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.config import get_settings
from backend.database.postgres import init_db
from backend.routers import auth_router, users_router, activities_router, groups_router

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: create tables
    init_db()
    logging.info("Database tables initialized")
    yield
    # Shutdown
    logging.info("Shutting down")


app = FastAPI(
    title="Verificador de Duplicidade API",
    version="2.0.0",
    lifespan=lifespan,
)

settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router.router)
app.include_router(users_router.router)
app.include_router(activities_router.router)
app.include_router(groups_router.router)


@app.get("/api/health")
def health():
    return {"status": "ok"}
