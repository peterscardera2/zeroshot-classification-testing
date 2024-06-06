from enum import Enum
import os
from typing import Dict, List
from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException
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

item_classification_router = APIRouter()

# HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

HfFolder.save_token("HUGGINGFACE_TOKEN")
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"  # for MPS fallback


class ItemType(Enum):
    FOOD = "food"
    BEVERAGE = "beverage"


class Item(BaseModel):
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
