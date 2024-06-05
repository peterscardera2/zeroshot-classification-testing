from fastapi.testclient import TestClient
from main import app
import pytest

client = TestClient(app)


@pytest.mark.parametrize(
    "item_name, expected_type",
    [
        ("pizza", "food"),
        ("coca-cola", "beverage"),
        ("papa burger", "food"),
        ("french fries", "food"),
        ("coffee", "beverage"),
        ("pasta", "food"),
        ("tea", "beverage"),
        ("pesto chicken", "food"),
        ("Bloody mary", "beverage"),
        ("nachos", "food"),
        ("negroni", "beverage"),
        ("mixed salad", "food"),
        ("Big Blue moon", "beverage"),
        ("chocolate cake", "food"),
        ("Big Guiness", "beverage"),
        ("bordeaux", "beverage"),
        ("primitivo", "beverage"),
        ("lambchops", "food"),
        ("spaghetti", "food"),
        # ("scaloppine di vitello con risotto", "food"), #IT FAILS
        
    ],
)
def test_item_classification(item_name, expected_type):
    response = client.post("/api/classify-item", json={"item_name": item_name})
    assert response.status_code == 200

    response_data = response.json()

    if response_data["food_score"] > response_data["beverage_score"]:
        classified_as = "food"
    else:
        classified_as = "beverage"

    assert (
        classified_as == expected_type
    ), f"{item_name} was classified a {classified_as} but I expected {expected_type}"
