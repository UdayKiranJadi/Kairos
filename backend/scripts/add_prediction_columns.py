import asyncio

from sqlalchemy import text

from app.db.session import engine


SQL_STATEMENTS = [
    """
    ALTER TABLE predictions
    ADD COLUMN IF NOT EXISTS horizon_minutes INTEGER;
    """,
    """
    ALTER TABLE predictions
    ADD COLUMN IF NOT EXISTS probability_up DOUBLE PRECISION;
    """,
    """
    ALTER TABLE predictions
    ADD COLUMN IF NOT EXISTS predicted_class INTEGER;
    """,
]


async def main():
    async with engine.begin() as conn:
        for statement in SQL_STATEMENTS:
            await conn.execute(text(statement))

    print("Prediction table columns updated.")


if __name__ == "__main__":
    asyncio.run(main())