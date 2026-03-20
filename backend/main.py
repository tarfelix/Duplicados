import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.config import get_settings
from backend.database.postgres import init_db
from backend.routers import auth_router, users_router, activities_router, groups_router

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        init_db()
        logging.info("Database tables initialized")
    except Exception:
        logging.exception("Failed to initialize database — app may not function correctly")
    yield
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

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    body = await request.body()
    logger.error(
        "Validation error on %s %s — body=%r, errors=%s",
        request.method, request.url.path, body[:500], exc.errors(),
    )
    return JSONResponse(status_code=422, content={"detail": exc.errors()})


app.include_router(auth_router.router)
app.include_router(users_router.router)
app.include_router(activities_router.router)
app.include_router(groups_router.router)


@app.get("/api/health")
def health():
    return {"status": "ok"}
