import gradio as gr
from transformers import pipeline

# Load the fill-mask pipeline with the specified model
mask_filler = pipeline(
    "fill-mask", model="RiKrim/distilbert-base-uncased-finetuned-imdb"
)

# Define a function that takes an input string with a masked token and returns the top predictions
def fill_mask(text):
    results = mask_filler(text)
    return {result['sequence']: result['score'] for result in results}

# Create a Gradio interface
inputs = gr.Textbox(lines=2, placeholder="Enter a sentence with a [MASK] token")
outputs = gr.Label(num_top_classes=5)

gr.Interface(fn=fill_mask, inputs=inputs, outputs=outputs, title="Fill Mask with DistilBERT", description="Enter a sentence with a [MASK] token to see the model's predictions.").launch()
