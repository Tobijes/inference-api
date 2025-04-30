# Mess with path to get example to import package
import sys, os
sys.path.append(os.path.abspath("../.."))
sys.path.append(os.path.abspath(".."))

import torch
from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline
from sonar.models.blaser.loader import load_blaser_model
from inference_api.model import InferenceModel
from settings import ModelSettings

Vector = list[float]


class SonarBlaserText(InferenceModel):
    model_metrics_timing_buckets = [10, 50, 100, 250, 500, 1000, 2500, 5000]
    settings: ModelSettings

    def __init__(self) -> None:      
        super().__init__() 
        self.logger.info("Loading models...")

        self.model_sonar = TextToEmbeddingModelPipeline(encoder="text_sonar_basic_encoder", tokenizer="text_sonar_basic_encoder", device=torch.device(self.device))
        self.logger.info("SONAR model initiated on %s", self.model_sonar.device)

        self.model_blaser = load_blaser_model("blaser_2_0_qe").eval()
        self.logger.info("BLASER model initiated on CPU")

    def infer(self, data: list, task: str, **kwargs):
        match task:
            case "SONAR":
                return self.sonar_embeddings(data, **kwargs)
            case "BLASER":
                return self.blaser_score(data)
            

    def sonar_embeddings(self, sentences: list[str], src_lang: str) -> list[Vector]:
        embeddings = self.model_sonar.predict(sentences, source_lang=src_lang, batch_size=self.settings.MAX_BATCH_SIZE, target_device=torch.device("cpu"))
        print(embeddings, flush=True)
        return embeddings
    
    def blaser_score(self, embeddings: list[tuple[Vector, Vector]]):
        src_embeddings, mt_embeddings = zip(*embeddings) #  [(1,2),(1,3),(1,4)] -> (1,1,1), (2,3,4)
        print(len(src_embeddings), len(mt_embeddings), flush=True)
        with torch.inference_mode():
            print(self.model_blaser(src=src_embeddings[0], mt=mt_embeddings[0]), flush=True)
            print(self.model_blaser(src=src_embeddings[0], mt=mt_embeddings[0]).item(), flush=True)
            scores = self.model_blaser(src=src_embeddings, mt=mt_embeddings).item()
        print(scores)
        return scores
    

