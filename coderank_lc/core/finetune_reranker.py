# ==============================================
# Fine-tuning Reranker Model using Human Feedback
# ==============================================

from coderank_lc.core.astra_store import list_recent_feedback
from sentence_transformers import InputExample, losses, SentenceTransformer
from torch.utils.data import DataLoader
import os

OUTPUT_DIR = "models/fine_tuned_reranker"
MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def prepare_training_data(limit=1000):
    """Convert human feedback pairs into InputExamples."""
    feedback_docs = list_recent_feedback(limit)
    examples = []
    for doc in feedback_docs:
        query = doc.get("query")
        a_text, b_text = doc.get("text_a"), doc.get("text_b")
        preferred = doc.get("preferred")
        if not all([query, a_text, b_text, preferred]):
            continue

        # Label 1 if human prefers A, else 0
        if preferred == "A":
            examples.append(InputExample(texts=[query, a_text], label=1.0))
            examples.append(InputExample(texts=[query, b_text], label=0.0))
        else:
            examples.append(InputExample(texts=[query, b_text], label=1.0))
            examples.append(InputExample(texts=[query, a_text], label=0.0))

    return examples


def fine_tune_reranker(batch_size=8, epochs=2, limit=1000):
    """Fine-tune the reranker using human feedback data."""
    print(f"Loading base reranker: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    print("Preparing training data from Astra feedback...")
    train_examples = prepare_training_data(limit)
    if not train_examples:
        print("No feedback data available for fine-tuning.")
        return

    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
    train_loss = losses.CosineSimilarityLoss(model)

    print(f"Starting fine-tuning with {len(train_examples)} examples...")
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=epochs,
        warmup_steps=int(0.1 * len(train_dataloader)),
        output_path=OUTPUT_DIR,
        show_progress_bar=True,
    )

    print(f"Fine-tuned reranker saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    fine_tune_reranker(batch_size=8, epochs=3, limit=1000)
