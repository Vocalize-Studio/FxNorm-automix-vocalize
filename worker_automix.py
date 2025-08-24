
import os
import json
import uuid
import base64
import asyncio
import tempfile
from typing import Optional, Dict

# Third-party imports
import aio_pika
import asyncpg
import signal
from contextlib import suppress
import aiormq
import soundfile as sf
from dotenv import load_dotenv
from loguru import logger
from minio import Minio

# Local application imports
from automix.inference_wrapper import FxNormAutomixWrapper, Progress

# Load environment variables from .env file
load_dotenv()

# ----------------------------- #
# --- Configuration from Env -- #
# ----------------------------- #
# RabbitMQ
RMQ_URL = os.getenv("RMQ_URL", "amqp://guest:guest@localhost/")
DB_URL = os.getenv("DATABASE_URL", "postgresql://user:password@localhost/dbname")
SVC_QUEUE = os.getenv("SVC_QUEUE", "automix.jobs")
CTRL_QUEUE = os.getenv("CTRL_QUEUE", "automix.control")
EVENTS_EX = os.getenv("EVENTS_EX", "automix.events")

# MinIO S3 Storage
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "localhost:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
MINIO_SECURE = os.getenv("MINIO_SECURE", "False").lower() in ('true', '1', 't')
MINIO_REGION = os.getenv("MINIO_REGION", "us-east-1")
MINIO_BUCKET = os.getenv("MINIO_BUCKET", "automix-jobs")

# Worker Configuration
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "/tmp/automix-output")
HEARTBEAT_SECS = int(os.getenv("HEARTBEAT_SECS", "45"))
AUTOMIX_CONFIG_PATH = os.getenv("AUTOMIX_CONFIG_PATH", "./configs/ISMIR/ours_S_Lb.py")
AUTOMIX_MODEL_PATH = os.getenv("AUTOMIX_MODEL_PATH", "./trainings/results/ours_S_Lb/net_mixture.dump")

# ----------------- #
# --- Logging ----- #
# ----------------- #
logger.remove()
logger.add(
    sink=lambda msg: print(msg, end=""),
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
           "<level>{level: <8}</level> | "
           "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
           "<level>{message}</level>",
    colorize=True,
)

# ----------------- #
# --- MinIO Helpers - #
# ----------------- #
_minio_client = None

def get_minio() -> Minio:
    global _minio_client
    if _minio_client is None:
        _minio_client = Minio(
            MINIO_ENDPOINT, access_key=MINIO_ACCESS_KEY, secret_key=MINIO_SECRET_KEY, secure=MINIO_SECURE, region=MINIO_REGION
        )
    return _minio_client

def ensure_bucket_exists():
    cli = get_minio()
    if not cli.bucket_exists(MINIO_BUCKET):
        logger.info(f"Creating MinIO bucket '{MINIO_BUCKET}'")
        cli.make_bucket(MINIO_BUCKET, location=MINIO_REGION)

def object_key_for_job(job_id: str) -> str:
    return f"jobs/{job_id}/mixed_output.wav"

def upload_file_to_minio(local_path: str, job_id: str) -> str:
    ensure_bucket_exists()
    cli = get_minio()
    obj_key = object_key_for_job(job_id)
    logger.info(f"Uploading '{local_path}' to MinIO as '{MINIO_BUCKET}/{obj_key}'...")
    cli.fput_object(MINIO_BUCKET, obj_key, file_path=local_path, content_type="audio/wav")
    return f"minio://{MINIO_BUCKET}/{obj_key}"

async def download_from_minio(uri: str, local_dir: str) -> str:
    if not uri.startswith("minio://"):
        raise ValueError(f"URI must start with minio://, got {uri}")
    
    bucket_name, _, object_key = uri[len("minio://"):].partition('/')
    local_path = os.path.join(local_dir, os.path.basename(object_key))
    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    logger.info(f"Downloading '{uri}' to '{local_path}'...")
    cli = get_minio()
    cli.fget_object(bucket_name, object_key, local_path)
    return local_path

# ----------------- #
# --- The Worker -- #
# ----------------- #
class AutomixWorker:
    def __init__(self, device: str = "cuda"):
        self.mixer = FxNormAutomixWrapper(
            config_path=AUTOMIX_CONFIG_PATH,
            model_path=AUTOMIX_MODEL_PATH,
            device=device
        )
        self.cancel_flags: Dict[str, asyncio.Event] = {}
        self._stop = asyncio.Event()
        self._conn: Optional[aio_pika.RobustConnection] = None
        self._ch: Optional[aio_pika.RobustChannel] = None
        self._db_pool: Optional[asyncpg.Pool] = None

    async def _get_db_pool(self) -> asyncpg.Pool:
        if self._db_pool is None:
            self._db_pool = await asyncpg.create_pool(DB_URL)
        return self._db_pool

    async def run(self):
        logger.info(f"Connecting to RabbitMQ: {RMQ_URL}")
        self._conn = await aio_pika.connect_robust(RMQ_URL)
        self._ch = await self._conn.channel()
        await self._ch.set_qos(prefetch_count=1)
        self._db_pool = await self._get_db_pool()

        logger.info("Declaring RabbitMQ topology...")
        await self._ch.declare_queue(SVC_QUEUE, durable=True)
        events_ex = await self._ch.declare_exchange(EVENTS_EX, aio_pika.ExchangeType.TOPIC, durable=True)

        q = await self._ch.get_queue(SVC_QUEUE)
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        logger.success(f"✅ READY. Waiting for jobs on '{SVC_QUEUE}'...")

        try:
            async with q.iterator() as it:
                async for msg in it:
                    if self._stop.is_set(): break
                    async with msg.process():
                        job = json.loads(msg.body.decode("utf-8"))
                        job_id = job["job_id"]
                        params = job["params"]
                        logger.info(f"📥 Received job {job_id}")
                        self.cancel_flags[job_id] = asyncio.Event()
                        try:
                            await self._process_job(events_ex, job_id, params)
                            logger.info(f"✅ Finished job {job_id}")
                        except Exception as e:
                            logger.exception(f"❌ Job {job_id} failed: {e}")
                            await self._update_job_status(job_id, 'error')
                            await self._publish(events_ex, job_id, "error", {"message": str(e)})
                        finally:
                            self.cancel_flags.pop(job_id, None)
        finally:
            await self._graceful_close()

    async def _graceful_close(self):
        logger.info("Shutting down...")
        self._stop.set()
        if self._conn: await self._conn.close()
        if self._db_pool: await self._db_pool.close()
        logger.info("Shutdown complete.")

    async def _publish(self, ex: aio_pika.Exchange, job_id: str, kind: str, payload: dict):
        body = {"v": 1, "kind": kind, "job_id": job_id, **payload}
        await ex.publish(
            aio_pika.Message(body=json.dumps(body).encode(), content_type="application/json"),
            routing_key=f"{job_id}.{kind}"
        )

    async def _update_job_status(self, job_id: str, status: str, final_uri: Optional[str] = None):
        db_pool = await self._get_db_pool()
        async with db_pool.acquire() as db_conn:
            if status == 'success':
                await db_conn.execute(
                    "UPDATE jobs SET status = 'success', finished_at = now(), result_uri = $2 WHERE id = $1",
                    uuid.UUID(job_id), final_uri
                )
            else:
                 await db_conn.execute(
                    "UPDATE jobs SET status = $2, finished_at = now() WHERE id = $1",
                    uuid.UUID(job_id), status
                )

    async def _process_job(self, ex: aio_pika.Exchange, job_id: str, p: dict):
        await self._update_job_status(job_id, 'running')
        
        loop = asyncio.get_running_loop()
        def progress_cb_threadsafe(prog: Progress):
            payload = {"pct": prog.pct, "status": prog.status, "meta": prog.meta}
            asyncio.run_coroutine_threadsafe(self._publish(ex, job_id, "progress", payload), loop)

        job_temp_dir = tempfile.mkdtemp(prefix=f"automix_{job_id}_")
        try:
            # Download all stems from MinIO
            stem_uris = p.get('stems', {})
            local_stem_paths = {}
            for name, uri in stem_uris.items():
                if uri:
                    local_stem_paths[name] = await download_from_minio(uri, job_temp_dir)
                else:
                    local_stem_paths[name] = None

            # The model expects all possible stems, so we create an ordered dict
            ordered_stems = {stem_name: local_stem_paths.get(stem_name) for stem_name in self.mixer.config.STEM_ORDER}

            output_path = os.path.join(job_temp_dir, "mixed_output.wav")

            # Run blocking model inference in a separate thread
            await asyncio.to_thread(
                self.mixer.mix,
                output_path=output_path,
                stem_paths=ordered_stems,
                progress_cb=progress_cb_threadsafe
            )

            # Upload result to MinIO
            final_uri = upload_file_to_minio(output_path, job_id)
            await self._update_job_status(job_id, 'success', final_uri)
            await self._publish(ex, job_id, "done", {"uri": final_uri})

        finally:
            # Clean up temporary directory
            try:
                import shutil
                shutil.rmtree(job_temp_dir)
                logger.info(f"Cleaned up temp directory: {job_temp_dir}")
            except Exception as e:
                logger.warning(f"Failed to clean up temp directory {job_temp_dir}: {e}")

# ----------------- #
# --- Entrypoint -- #
# ----------------- #
if __name__ == "__main__":
    worker = AutomixWorker(device=os.getenv("DEVICE", "cuda"))
    loop = asyncio.get_event_loop()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda: asyncio.create_task(worker._graceful_close()))

    try:
        loop.run_until_complete(worker.run())
    finally:
        loop.close()
