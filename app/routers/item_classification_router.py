import asyncio
from enum import Enum
import json
import logging
import os
import time
from typing import Dict, List
import aiohttp
from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException
import httpx
from pydantic import BaseModel, Field
import torch
from transformers import (
    DistilBertForSequenceClassification,
    AutoModelForCausalLM,
    DistilBertTokenizer,
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
from huggingface_hub import HfFolder
from transformers import pipeline

import app

load_dotenv()
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("item-classification-logger")
item_classification_router = APIRouter()

# HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

HfFolder.save_token("HUGGINGFACE_TOKEN")
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"  # for MPS fallback


class ItemType(Enum):
    FOOD = "food"
    BEVERAGE = "beverage"


class Item(BaseModel):
    # id: str
    item_name: str
    accounting_group: str


class ClassifyItemsRequest(BaseModel):
    items: List[Item]


class ClassifyItemsResponseDto:
    def __init__(self, item_name, food_score, beverage_score):
        self.item_name = item_name
        self.food_score = str(food_score)
        self.beverage_score = str(beverage_score)


LLAMA3_8B = "meta-llama/Meta-Llama-3-8B"
DISTILBERT = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
# pipe = pipeline(
#     "text-classification",
#     model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
# )

try:
    # tokenizer = AutoTokenizer.from_pretrained(LLAMA3_8B)
    tokenizer = DistilBertTokenizer.from_pretrained(DISTILBERT)
except Exception as e:
    raise RuntimeError(f"Error initializing tokenizer: {e}")
# https://pytorch.org/docs/stable/notes/mps.html
try:
    # model = AutoModelForCausalLM.from_pretrained(LLAMA3_8B)
    model = DistilBertForSequenceClassification.from_pretrained(
        DISTILBERT
    )  # sentiment analysis
    # if torch.backends.mps.is_built() and torch.backends.mps.is_available():
    #     device = torch.device("mps")
    # elif torch.cuda.is_available():
    #     device = torch.device("cuda")
    # else:
    device = torch.device("cpu")
    model.to(device)
except Exception as e:
    raise RuntimeError(f"Error initializing model: {e}")


# @item_classification_router.post("/classify-item", response_model=str)
async def classify_item_llamapy(item: Item):
    prompt = f"Classify the following item as food or beverage: {item.item_name}."
    inputs = tokenizer(prompt, return_tensors="pt")

    for key in inputs:  # moving each tensor to the device
        inputs[key] = inputs[key].to(device)
    try:
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            predicted_class_idx = logits.argmax(-1).item()

            if predicted_class_idx == 0:
                item_type = ItemType.FOOD
            elif predicted_class_idx == 1:
                item_type = ItemType.BEVERAGE
            else:
                raise HTTPException(status_code=500, detail="Unexpected model output")

            return item_type.value
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# https://huggingface.co/facebook/bart-large-mnli
classifier_bart_large_mnli = pipeline(
    "zero-shot-classification", model="facebook/bart-large-mnli"
)


# using pipeline to abstract away everything that needs to be set up manually
# @item_classification_router.post("/classify-item")
async def classify_item(item: Item):
    candidate_labels = [ItemType.FOOD.value, ItemType.BEVERAGE.value]
    try:
        results = classifier_bart_large_mnli(item.item_name, candidate_labels)
        labels = results["labels"]
        scores = results["scores"]
        # classifier seems to return results where the order of the labels directly correspondto the order of the scores
        food_score = scores[labels.index("food")]
        beverage_score = scores[labels.index("beverage")]
        return {"food_score": str(food_score), "beverage_score": str(beverage_score)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# tokenizer = AutoTokenizer.from_pretrained("MoritzLaurer/deberta-v3-large-zeroshot-v2.0")
# model = AutoModelForSequenceClassification.from_pretrained("MoritzLaurer/deberta-v3-large-zeroshot-v2.
# @item_classification_router.post("/classify-item")
# async def classify_item_debert(item: Item):
#     candidate_labels = [ItemType.FOOD.value, ItemType.BEVERAGE.value]
#     try:
#         encoded_input = tokenizer(
#             item.item_name,
#             candidate_labels,
#             padding=True,
#             truncation=True,
#             return_tensors="pt",
#             max_length=512,
#         )
#         with torch.no_grad():  # infering here no need for gradient calcl
#             outputs = model(**encoded_input)

#         entailment_cls_token_id = model.config.label2id.get(
#             "entailment", 1  # default to 1 if its not specified
#         )
#         scores = torch.softmax(outputs.logits[:, entailment_cls_token_id], dim=1)

#         # we have to extract score now

#      except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# https://huggingface.co/MoritzLaurer/deberta-v3-large-zeroshot-v2.0
classifier_deberta_large_v3_zero_shot = pipeline(
    "zero-shot-classification", model="MoritzLaurer/deberta-v3-large-zeroshot-v2.0"
)


@item_classification_router.post("/classify-items")
async def classify_items_debert_pipeline(items: ClassifyItemsRequest):
    classifyItemsResponseDto = []
    candidate_labels = [ItemType.FOOD.value, ItemType.BEVERAGE.value]
    # it should provide acontext on how the model should understand the labels in relation to the input text
    hypothesis_template = "Given that this item is from the '{0}' category, it is typically consumed as a type of {{}}."
    for item in items.items:
        try:
            formated_hypo = hypothesis_template.format(item.accounting_group)
            results = classifier_deberta_large_v3_zero_shot(
                item.item_name,
                candidate_labels,
                hypothesis_template=formated_hypo,
                multi_label=False,  # False;theyre mutually exclusive; one or the other;
            )
            labels = results["labels"]
            scores = results["scores"]
            # classifier seems to return results where the order of the labels directly correspondto the order of the scores

            food_score = scores[labels.index("food")]
            beverage_score = scores[labels.index("beverage")]
            item_result = ClassifyItemsResponseDto(
                item_name=item.item_name,
                food_score=food_score,
                beverage_score=beverage_score,
            )
            classifyItemsResponseDto.append(item_result)
        except Exception as e:
            classifyItemsResponseDto.append(
                {"item_name": item.item_name, "error": str(e)}
            )

    return classifyItemsResponseDto


AZURE_OPENAI_API_KEY = ""
MAX_TOKENS_PER_MINUTE = 120000
MAX_REQUESTS_PER_MINUTE = 720
# Azure OpenAI Studio>Deployments>test-deployment-gpt35turbo
# Properties:
# Model name: gpt-35-turbo
# Model version: 0301
# Version update policy: Once a new default version is available.
# Deployment type: Standard
# Content Filter: Default
# Tokens per Minute Rate Limit (thousands): 120
# Rate limit (Tokens per minute): 120000
# Rate limit (Requests per minute): 720


#   https://learn.microsoft.com/en-us/azure/ai-services/openai/chatgpt-quickstart?source=recommendations&tabs=command-line%2Cpython-new&pivots=rest-api#rest-api


async def classify_req(client, items_batch):
    url = "https://hos-dev-ai-test-instance.openai.azure.com/openai/deployments/test-deployment-gpt35turbo/chat/completions?api-version=2023-03-15-preview"
    # TODO WHY DOESNT VERSION 2024-05-13 not wokfor 4o ......
    urlgpt4o = "https://hos-dev-ai-test-instance.openai.azure.com/openai/deployments/test-deployment-gpt4o/chat/completions?api-version=2023-03-15-preview"

    headers = {
        "Content-Type": "application/json",
        "api-key": f"{AZURE_OPENAI_API_KEY}",
    }
    messages = [
        {
            "role": "system",
            "content": 'You are an assistant that classifies items as either food or beverage based on their names and accounting group. Example JSON output format expected:\n{\n "item_name": "",\n  "category": "",\n}',
        }
    ]
    items_content = {
        "items": [
            {
                "item_name": item.item_name,
                "accounting_group": item.accounting_group,
            }
            for item in items_batch
        ]
    }
    messages.append({"role": "user", "content": json.dumps(items_content)})
    # TODO: store in a cache to make sure if its already done lets not ask to categorize again (lets say a new item is created and we already categorized it)
    # TODO: save on token on json struct
    # TODO: save the in the db with a focus on on the name (so nothing to do with ID just the name and the category)
    # TODO: later on compare your pizza versus pizza of the peers.
    # give me the food taxonomy for all my locations, this information will be no precise
    # im creating a pizza suggest the price for this pizza
    request_to_azureopenai = {
        "messages": messages,
        "n": 1,
        "response_format": {
            "type": "json_object"
        },  # "Invalid parameter: 'response_format' of type 'json_object' is not supported with this model.",
        # TODO: IMPORTANT ITS gpt-35-turbo (0125) NEW https://learn.microsoft.com/en-us/azure/ai-services/openai/concepts/models#gpt-35 SO NOT THE DEFAULT
        # "temperature": 0.7,
        # "max_tokens": 150,
    }
    logging.info(f"Request payload: {request_to_azureopenai}")

    try:
        response = await client.post(
            urlgpt4o, json=request_to_azureopenai, headers=headers
        )
        response_json = response.json()
        logging.info(f"Received response: {response_json}")
        if response.status_code != 200:
            logging.error(f"Error response from Azure OpenAI: {response_json}")
            response.raise_for_status()
        return response_json
    except httpx.HTTPStatusError as e:
        logging.error(
            f"HTTP error: {e.response.text} (Status: {e.response.status_code})"
        )
        raise HTTPException(
            status_code=e.response.status_code, detail=f"HTTP error: {e.response.text}"
        )
    except httpx.RequestError as e:
        logging.error(f"Request error: {e}")
        raise HTTPException(status_code=500, detail=f"Request error: {e}")
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")


async def rate_limited_classify(client, item_batches):
    semaphore = asyncio.Semaphore(5)
    completion_tasks = []

    total_requests = 0
    total_tokens = 0
    start_time = time.time()

    async def classify_with_semaphore(item_batch):
        nonlocal total_requests, total_tokens, start_time  # so its works accross mutple tasks
        async with semaphore:
            if (
                total_requests >= MAX_REQUESTS_PER_MINUTE
                or total_tokens >= MAX_TOKENS_PER_MINUTE
            ):
                logger.info("Reach max req or token!")
                elapsed_time = time.time() - start_time
                if elapsed_time < 60:
                    logging.info(
                        f"rate limit reached. Sleeping for {60 - elapsed_time} sec"
                    )
                    await asyncio.sleep(60 - elapsed_time)
                total_requests = 0
                total_tokens = 0
                start_time = time.time()

            logging.info(
                f"seding request number {total_requests + 1} with num items: {len(item_batch)}"
            )
            response = await classify_req(client, item_batch)
            # seems like its usage.complete_tokens or total_tokens (to let them tell us)
            usage = response.get("usage", {})
            total_tokens += usage.get("total_tokens", 0)
            total_requests += 1
            logging.info(
                f"total requests sent: {total_requests}, total tokes used: {total_tokens}"
            )
            return response

    for item_batch in item_batches:
        completion_tasks.append(classify_with_semaphore(item_batch))

    try:
        results = await asyncio.gather(*completion_tasks)
        classified_items = []
        for result in results:
            if "choices" in result:
                for choice in result["choices"]:
                    # Maybe validate it hte json by cathing a JSON decodeError then retry that whole batch
                    content = json.loads(choice["message"]["content"])
                    classified_items.append(content)
        return classified_items
    except Exception as e:
        logging.error(f"GG. exc: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@item_classification_router.post("/azure/classify-items")
async def classify_via_azure_in_batches(request: ClassifyItemsRequest):
    # lets try batches of 100
    item_batches = [
        request.items[i : i + 100] for i in range(0, len(request.items), 100)
    ]
    async with httpx.AsyncClient(http2=True) as client:
        return await rate_limited_classify(client, item_batches)
