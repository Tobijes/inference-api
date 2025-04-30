# For testing run commands from project root
# python -m build && pip install dist/*-*.whl && python test/smoke.py
# For uninstalling again:
# pip uninstall inference_api && rm -rf dist

import asyncio

from inference_api import InferenceAPI, InferenceModel

class TestModel(InferenceModel):

    @InferenceModel.task()
    def predict(self, text):
        return text

async def main():
    app = InferenceAPI(model_type=TestModel)
    result = await app.submit_task(TestModel.predict, "Hello, World")
    assert result == "Hello, World"

if __name__ == "__main__":
    asyncio.run(main())
    print("Test completed successfully")