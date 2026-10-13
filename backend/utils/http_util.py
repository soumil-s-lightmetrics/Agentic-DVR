import aiohttp
import asyncio
import os
from utils.html_util import convert_p_to_div_with_style

# Async HTTP request with retries
async def fetch_with_retry(url, method="GET", params=None, payload=None, headers=None, auth=None, retries=3):
    print(f"[API call] Method: {method}, URL: {url}, Params: {params}")
    async with aiohttp.ClientSession() as session:
        for attempt in range(retries):
            try:
                async with session.request(method, url, params=params, json=payload, headers=headers, auth=auth) as response:
                    if response.status == 429:  # Rate limit
                        retry_after = int(response.headers.get("Retry-After", 10))
                        print(f"Rate limit hit. Retrying after {retry_after} seconds...")
                        await asyncio.sleep(retry_after)
                        continue
                    response.raise_for_status()
                    return await response.json()
            except aiohttp.ClientError as e:
                print(f"HTTP error (attempt {attempt + 1}/{retries}): {e}")
                if attempt < retries - 1:
                    await asyncio.sleep(2)
                else:
                    raise