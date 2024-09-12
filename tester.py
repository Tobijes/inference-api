import asyncio
import httpx
import random

with open("./texts.txt") as f:
    texts = f.readlines()

base_url = "http://localhost:8000"

concurrency = 20
client = httpx.AsyncClient()

async def actor(endpoint):
    while True:
        sample_size = random.randint(1,10)
        sample = random.choices(texts, k=sample_size)

        try:
            r = await client.post(base_url + endpoint, json=sample, timeout=60.0)
            print(f"Request with {len(sample)} done")
        except Exception as e:
            print("Error:", e)
        # Sleep 0-5 seconds
        await asyncio.sleep(random.random() * 1)



async def main():
    for i in range(concurrency):
        asyncio.create_task(actor("/passage"))
        asyncio.create_task(actor("/query"))
    

loop = asyncio.get_event_loop()
loop.create_task(main())
loop.run_forever()