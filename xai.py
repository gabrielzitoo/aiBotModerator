import os
from shap import Explanation
import shap
import torch
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification

import matplotlib.pyplot as plt

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def predict(texts):
    if not isinstance(texts, list):
        texts = texts.tolist() if hasattr(texts, 'tolist') else [str(texts)]

    model = BertForSequenceClassification.from_pretrained("./bert-imdb/checkpoint-40000")
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=256,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    probs = torch.nn.functional.softmax(logits, dim=1)

    return probs.numpy()

def evaluateXAI(dataset):
    print("xai_results")
    output_folder = "xai_results"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    explainer = shap.Explainer(predict, tokenizer)
    sample_texts = dataset["train"].shuffle(seed=42).select(range(50))["review_text"]
    shap_values = explainer(sample_texts)

    # Helper function to join path
    def save_path(filename):
        return os.path.join(output_folder, filename)

    # --- GLOBAL PLOTS ---
    
    # Global Bar Plot
    shap.plots.bar(shap_values.abs.mean(0), show=False)
    plt.savefig(save_path("shap_bar_global.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # Beeswarm Plot
    shap.plots.beeswarm(shap_values.abs.mean(0), show=False)
    plt.savefig(save_path("shap_beeswarm.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # --- HTML PLOTS ---
    
    # Save text explanation (First sample only)
    html_primeiro = shap.plots.text(shap_values[0], display=False)
    with open(save_path("exemplo_0_text.html"), "w", encoding="utf-8") as f:
        f.write(html_primeiro)

    # Save text explanation (All samples)
    html_content = shap.plots.text(shap_values, display=False)
    with open(save_path("shap_text.html"), "w", encoding="utf-8") as f:
        f.write(html_content)

    # --- LOCAL PLOTS (Example 0) ---

    exp_0_spoiler = Explanation(
        values=shap_values[0, :, 1].values,      # Only the 'Spoiler' class values
        base_values=shap_values[0, :, 1].base_values, # Only the 'Spoiler' base value
        data=shap_values[0, :, 1].data,          # The actual tokens/words
        display_data=shap_values[0, :, 1].data   # For display in the plot
    )

    # 2. Individual Bar plot
    try:
        shap.plots.bar(exp_0_spoiler, show=False)
        plt.savefig(save_path("exemplo_0_bar.png"), bbox_inches="tight")
        plt.close()
    except Exception as e:
        print(f"Bar plot failed: {e}. Moving to waterfall...")

    # 3. Waterfall plot
    try:
        shap.plots.waterfall(exp_0_spoiler, show=False)
        plt.savefig(save_path("exemplo_0_waterfall.png"), bbox_inches="tight")
        plt.close()
    except Exception as e:
        print(f"Waterfall plot failed: {e}")

    # --- VARIATIONS ---
    
    for count in [15, 20]:
        shap.plots.bar(shap_values.abs.mean(0), max_display=count, show=False)
        plt.savefig(save_path(f"shap_bar_{count}.png"), bbox_inches="tight")
        plt.close()