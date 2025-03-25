import torch
from transformers import DistilBertTokenizer, DistilBertModel
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Initialize DistilBERT tokenizer and model
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertModel.from_pretrained('distilbert-base-uncased')

# Define Eisenhower category prompts (acting as "hypotheses" for zero-shot)
eisenhower_prompts = {
    "Urgent & Important (Do First)": "This task is critical and must be done immediately to avoid serious consequences.",
    "Not Urgent but Important (Schedule)": "This task is important for long-term goals but can be planned for later.",
    "Urgent but Not Important (Delegate)": "This task needs quick attention but can be handled by someone else.",
    "Not Urgent & Not Important (Eliminate)": "This task is trivial and does not contribute to key objectives."
}

# Function to get embeddings from DistilBERT
def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    # Use the [CLS] token embedding (first token)
    embedding = outputs.last_hidden_state[:, 0, :].numpy()
    return embedding

# Custom zero-shot classification function
def classify_task(task):
    # Get embedding for the task
    task_embedding = get_embedding(task)

    # Get embeddings for each category prompt
    prompt_embeddings = {category: get_embedding(prompt) for category, prompt in eisenhower_prompts.items()}

    # Calculate cosine similarity between task and each prompt
    similarities = {}
    for category, prompt_embedding in prompt_embeddings.items():
        similarity = cosine_similarity(task_embedding, prompt_embedding)[0][0]
        similarities[category] = similarity

    # Return the category with the highest similarity
    best_category = max(similarities, key=similarities.get)
    return best_category

# Example usage
if __name__ == "__main__":
    # Test tasks
    test_tasks = [
        "Urgently fix critical server outage",
        "Plan next quarter's strategy",
        "Schedule urgent team meeting",
        "Organize old files"
    ]

    for task in test_tasks:
        category = classify_task(task)
        print(f"Task: '{task}' -> Category: '{category}'")