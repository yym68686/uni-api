import asyncio
import os
import sys

import httpx

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import main


def test_healthz_does_not_require_api_key():
    async def run_request():
        transport = httpx.ASGITransport(app=main.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            return await client.get("/healthz")

    response = asyncio.run(run_request())

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "version": main.VERSION}
