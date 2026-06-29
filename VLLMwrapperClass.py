import os
import asyncio
import aiohttp
import json
from typing import List, Dict, Any, Optional, Union
from dotenv import load_dotenv
import logging
from pathlib import Path
import stanza
import base64
import uuid

load_dotenv()

# ---------- Logging ----------
logging.basicConfig(
    format="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("stats_async.log", mode="w", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

# ---------- Defaults from env ----------
DEFAULT_BASE_URL = os.getenv("BASE_URL", "http://localhost:8000/v1")
DEFAULT_MODEL = os.getenv("MODEL")


class VLLMWrapper:

    def __init__(
        self,
        prompt: Optional[str],
        session: aiohttp.ClientSession,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        extra_params: Optional[Dict[str, Any]] = None,
        timeout_s: Optional[float] = 120.0,
    ) -> None:
        self.prompt = prompt or ""
        self.session = session
        self.base_url = (base_url or "").rstrip("/")  # avoid double slashes
        self.model = model or DEFAULT_MODEL
        self.extra_params = extra_params or {}
        self.timeout = timeout_s
        self.super_base = "http://localhost:8000"
        logger.info("Created a prompt object (model=%s, base_url=%s)", self.model, self.base_url)

        # ---------------------- helpers ----------------------
    @staticmethod
    def chunk_text_by_chars(text: str, chunk_size: int = 2000, overlap: int = 200) -> List[str]:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")
        if overlap < 0:
            raise ValueError("overlap must be >= 0")
        if not text:
            return []

        chunks: List[str] = []
        n = len(text)
        start = 0
        while start < n:
            end = min(start + chunk_size, n)
            chunks.append(text[start:end])
            if end == n:
                break
            # ensure forward progress even if overlap >= chunk_size
            next_start = end - overlap
            start = next_start if next_start > start else end
        return chunks

    @staticmethod
    def read_text(file_path: str) -> str:
        p = Path(file_path)
        suffix = p.suffix.lower()
        logger.info("Extracting text")
        if suffix == ".txt":
            return p.read_text(encoding="utf-8", errors="ignore")

        elif suffix == ".pdf":
            try:
                from pypdf import PdfReader
            except Exception:
                try:
                    from PyPDF2 import PdfReader
                except Exception as e:
                    raise RuntimeError(
                        "Для чтения PDF установите 'pypdf' (или 'PyPDF2')."
                    ) from e
            reader = PdfReader(str(p))
            texts = []
            for page in reader.pages:
                t = page.extract_text() or ""
                texts.append(t)
            logger.info("PDF extraction complete")
            return "\n".join(texts)

        elif suffix == ".docx":
            try:
                import docx
            except Exception as e:
                raise RuntimeError(
                    "Для чтения DOCX установите 'python-docx'."
                ) from e
            doc = docx.Document(str(p))
            logger.info("Docx extraction complete")
            return "\n".join(par.text for par in doc.paragraphs)

        else:
            raise ValueError(f"Неподдерживаемый формат: {suffix}. Ожидаются .txt, .pdf или .docx")

    async def _read_json(self, resp: aiohttp.ClientResponse) -> Any:
        text = await resp.text()
        if not (200 <= resp.status < 300):
            try:
                j = json.loads(text)
                msg = j.get("error", {}).get("message") or j
            except Exception:
                msg = text
            raise RuntimeError(f"HTTP {resp.status}: {msg}")
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            raise RuntimeError("Response is not valid JSON")

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}" if path.startswith("/") else f"{self.base_url}/{path}"

    async def get(self, path: str, params: Optional[Dict[str, Any]] = None,
                  headers: Optional[Dict[str, str]] = None) -> Any:
        url = self._url(path)

        headers = {"Content-Type": "application/json", **(headers or {})}
        async with self.session.get(url, params=params, headers=headers, timeout=self.timeout) as resp:
            return await self._read_json(resp)

    async def post(self, path: str, payload: Dict[str, Any], headers: Optional[Dict[str, str]] = None) -> Any:
        url = self.base_url+path

        headers = {"Content-Type": "application/json", **(headers or {})}
        async with self.session.post(url, json=payload, headers=headers, timeout=self.timeout) as resp:
            return await self._read_json(resp)

    async def tokenize(self, text: str, *, model: Optional[str] = None, path: str = "/tokenize") -> List[int]:
        payload = {"text": text, "model": model or self.model}
        async with self.session.post("http://localhost:8000/tokenize", json=payload, timeout=self.timeout) as resp:
            return await self._read_json(resp)
        if isinstance(data, dict):
            if isinstance(data.get("tokens"), list):
                return list(map(int, data["tokens"]))
        if isinstance(data.get("input_ids"), list):
            return list(map(int, data["input_ids"]))
        raise RuntimeError("Unexpected tokenizer response")

    async def detokenize(self, tokens: List[int], *, model: Optional[str] = None, path: str = "/detokenize") -> str:
        url = f"{self.super_base}{path}"
        payload = {"tokens": list(map(int, tokens)), "model": model or self.model}
        data = await self._post_abs(url, payload)
        if isinstance(data, dict):
            if isinstance(data.get("text"), str):
                return data["text"]
        if isinstance(data.get("decoded"), str):
            return data["decoded"]
        raise RuntimeError("Unexpected detokenizer response")

    async def embeddings_root(self, inputs: Union[str, List[str]], *, model: Optional[str] = None,
                              path: str = "/embeddings") -> List[List[float]]:
        if isinstance(inputs, str):
            _inputs = [inputs]
        else:
            _inputs = list(inputs)
        payload = {"model": model or self.model, "input": _inputs}
        logger.info(f"Embedding sent to {model}")
        data = await self.post(path, payload)
        if isinstance(data, dict) and isinstance(data.get("data"), list):
            return [row["embedding"] for row in data["data"]]
        raise RuntimeError("Unexpected embeddings response format")

    async def chat_completion_one(self, user_content: str) -> str:
        """Send a single user message and return assistant text content.

        Raises RuntimeError on non-2xx responses or malformed JSON.
        """
        messages: List[Dict[str, str]] = []
        if self.prompt:
            messages.append({"role": "system", "content": self.prompt})
        messages.append({"role": "user", "content": user_content})

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
        }
        if self.extra_params:
            payload.update(self.extra_params)
        logger.info(f'Sending a message to LLM: {payload["messages"]}')
        data = await self.post(f"/chat/completions", payload)
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            raise RuntimeError(f"Unexpected response format: missing choices[0].message.content")

    async def chat_completion_batch(
            self,
            user_contents: List[str],
            *,
            concurrency: int = 5,
            return_exceptions: bool = False,
    ) -> List[Optional[str]]:
        if concurrency <= 0:
            raise ValueError("concurrency must be > 0")

        sem = asyncio.Semaphore(concurrency)

        async def _run(idx: int, content: str):
            async with sem:
                try:
                    logger.info(f"Batch item {idx} started")
                    result = await self.chat_completion_one(content)
                    logger.info(f"Batch item {idx} finished")
                    return result
                except Exception as e:  # noqa: BLE001
                    logger.exception("Batch item %d failed: %s", idx, e)
                    if return_exceptions:
                        return e
                    raise
        tasks = [asyncio.create_task(_run(i, c)) for i, c in enumerate(user_contents)]
        results = await asyncio.gather(*tasks, return_exceptions=return_exceptions)
        return list(results)

    async def get_gigachat_token(
            self,
            session: aiohttp.ClientSession,
            *,
            client_id: str,
            client_secret: str,
            scope: str = "GIGACHAT_API_B2B",
            ssl: bool = False,
    ) -> str:
        logger.info(f"Getting auth token for {client_id}")
        url = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"

        auth = base64.b64encode(f"{client_id}:{client_secret}".encode("utf-8")).decode("utf-8")

        headers = {
            "Authorization": f"Basic {auth}",
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
            "RqUID": str(uuid.uuid4()),
        }

        data = {"scope": scope}

        async with session.post(url, headers=headers, data=data, ssl=ssl) as resp:
            payload = await self._read_json(resp)

        token = payload.get("access_token")
        if not token:
            raise RuntimeError(f"Не удалось получить access_token: {payload}")

        return token

    async def gigachat_chat_completion_auto_token(
            self,
            *,
            user_content: str,
            model: str = "GigaChat-Max",
            scope: str = "GIGACHAT_API_B2B",
            temperature: float = 0.2,
            top_p: float = 0.9,
            max_tokens: int = 512,
            ssl: bool = False,
    ) -> str:
        access_token = await self.get_gigachat_token(
            self.session,
            client_id=os.getenv("CLIENT_ID"),
            client_secret=os.getenv("CLIENT_SECRET"),
            scope=scope,
            ssl=ssl,
        )
        logger.info(f"Sending a message to gigachat_api")
        url = "https://gigachat.devices.sberbank.ru/api/v1/chat/completions"

        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        messages: List[Dict[str, str]] = []
        if self.prompt:
            messages.append({"role": "system", "content": self.prompt})
        messages.append({"role": "user", "content": user_content})
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "stream": False,
        }

        async with self.session.post(url, headers=headers, json=payload, ssl=ssl) as resp:
            data = await self._read_json(resp)

        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            raise RuntimeError(f"Unexpected GigaChat response: {data}")

    async def gigachat_chat_completion_batch_auto_token(
            self,
            *,
            user_contents: List[str],
            model: str = "GigaChat-Max",
            scope: str = "GIGACHAT_API_B2B",
            temperature: float = 0.2,
            top_p: float = 0.9,
            max_tokens: int = 512,
            concurrency: int = 4,
            ssl: bool = False,
            return_exceptions: bool = False,
    ) -> List[Optional[str]]:
        access_token = await self.get_gigachat_token(
            self.session,
            client_id=os.getenv("CLIENT_ID"),
            client_secret=os.getenv("CLIENT_SECRET"),
            scope=scope,
            ssl=ssl,
        )
        logger.info(f"Sending a batch message to gigachat_api")
        url = "https://gigachat.devices.sberbank.ru/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        sem = asyncio.Semaphore(concurrency)

        async def _one(idx: int, content: str):
            async with sem:
                try:
                    messages: List[Dict[str, str]] = []
                    logger.info(f"Sending a batch message to gigachat_api. Message index = {idx}")
                    if self.prompt:
                        messages.append({"role": "system", "content": self.prompt})

                    messages.append({"role": "user", "content": content})

                    payload = {
                        "model": model,
                        "messages": messages,
                        "temperature": temperature,
                        "top_p": top_p,
                        "max_tokens": max_tokens,
                        "stream": False,
                    }

                    async with self.session.post(url, headers=headers, json=payload, ssl=ssl) as resp:
                        data = await self._read_json(resp)

                    return data["choices"][0]["message"]["content"]

                except Exception as e:
                    if return_exceptions:
                        return e
                    raise

        tasks = [asyncio.create_task(_one(i, content)) for i, content in enumerate(user_contents)]
        results = await asyncio.gather(*tasks, return_exceptions=return_exceptions)
        return list(results)

    async def gigachat_list_models(
            self,
            *,
            scope: str = "GIGACHAT_API_B2B",
            ssl: bool = False,
    ) -> List[Dict[str, Any]]:
        access_token = await self.get_gigachat_token(
            self.session,
            client_id=os.getenv("CLIENT_ID"),
            client_secret=os.getenv("CLIENT_SECRET"),
            scope=scope,
            ssl=ssl,
        )

        url = "https://gigachat.devices.sberbank.ru/api/v1/models"

        headers = {
            "Authorization": f"Bearer {access_token}",
            "Accept": "application/json",
        }

        async with self.session.get(url, headers=headers, ssl=ssl) as resp:
            data = await self._read_json(resp)

        return data.get("data", [])


    async def list_models(self) -> List[Dict[str, Any]]:
        data = await self.get("/models")
        try:
            return list(data.get("data", []))
        except AttributeError:
            raise RuntimeError("Unexpected /models response format")

    @staticmethod
    def extract_entyties_sentences(filename: str):
        nlp1 = stanza.Pipeline(lang='ru', processors='tokenize,ner')
        doc = nlp1(VLLMWrapper.read_text(filename))
        allowed_types = {"PER", "ORG", "LOC", "GPE"}
        sent_entities = []
        for sent in doc.sentences:
            entities = [f"text: {ent.text}, type: {ent.type}" for ent in sent.ents if ent.type in allowed_types]
            if entities:
                tmp = " ".join(entities)
                sent_entities.append(f"sentence: {sent.text}, entities: {tmp}")
        return sent_entities


#768