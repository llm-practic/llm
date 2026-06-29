from VLLMwrapperClass import VLLMWrapper
import aiohttp
import os
import asyncio

async def rag():
    async with aiohttp.ClientSession() as session:
        checker = VLLMWrapper(
            prompt=(
                """Ты эксперт по программированию. Отвечай коротко и по существу на русском языке"""
            ),
            session=session,
            model=os.getenv("MODEL", "cyankiwi/gemma-4-26B-A4B-it-AWQ-8bit"),
            base_url=os.getenv("BASE_URL", "http://localhost:8000/v1"),
            extra_params={
                "temperature": 0.3,
                "top_p": 0.3,
                "max_tokens": 1024,
        }
        )
        res =  await checker.chat_completion_one("Расскажи про рекурсивные функции")
        print(res)


if __name__ == "__main__":
    asyncio.run(rag())