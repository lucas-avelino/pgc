import aiohttp
import random


class Client:
    def __init__(self, urls: list[str]):
        self.urls = urls
        self._session: aiohttp.ClientSession | None = None

    def random_url(self) -> str:
        return random.choice(self.urls)

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None:
            self._session = aiohttp.ClientSession()
        return self._session

    async def get(self, key: str) -> dict:
        session = await self._get_session()
        async with session.get(f"{self.random_url()}/get", params={"key": key}) as response:
            return await response.json()

    async def get_near(self, key: str, radius: int) -> dict:
        session = await self._get_session()
        async with session.get(
            f"{self.random_url()}/get_near", params={"key": key, "r": radius}
        ) as response:
            return await response.json()

    async def put(self, key: str, value: str) -> dict:
        session = await self._get_session()
        async with session.post(
            f"{self.random_url()}/put", json={"key": key, "value": value}
        ) as response:
            return await response.json()

    async def close(self) -> None:
        if self._session:
            await self._session.close()
            self._session = None