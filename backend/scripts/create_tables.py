import asyncio

from app.db.base import Base
from app.db.models import *  # noqa: F403
from app.db.session import engine


async def main():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    print("Database tables created.")


if __name__ == "__main__":
    asyncio.run(main())