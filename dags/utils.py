import aiohttp
import asyncio
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def fetch_url(session, url):
    max_retries = 5
    backoff_factor = 2
    for attempt in range(max_retries):
        try:
            async with session.get(url) as response:
                if response.status == 429:
                    retry_after = response.headers.get("Retry-After")
                    if retry_after:
                        retry_after = int(retry_after)
                    else:
                        retry_after = backoff_factor ** attempt
                    logger.warning(f"Rate limit exceeded. Retrying after {retry_after} seconds...")
                    await asyncio.sleep(retry_after)
                elif response.status >= 400:
                    response.raise_for_status()
                else:
                    return await response.json()
        except aiohttp.ClientResponseError as e:
            logger.error(f"Request failed: {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(backoff_factor ** attempt)
            else:
                raise
        except aiohttp.ClientError as e:
            logger.error(f"Client error: {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(backoff_factor ** attempt)
            else:
                raise
    raise Exception(f"Failed to fetch {url} after {max_retries} attempts")

async def fetch_data(url):
    async with aiohttp.ClientSession() as session:
        return await fetch_url(session, url)

def fetch_data_sync(url):
    return asyncio.run(fetch_data(url))