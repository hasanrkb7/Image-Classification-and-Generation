# Title: Task 9 - Prompt Engineering UI System (With Result Storage)

import gradio as gr
import pandas as pd
import os

# File to store results
RESULT_FILE = "../results/results.csv"

# Create file if not exists
if not os.path.exists(RESULT_FILE):
    df = pd.DataFrame(columns=["Prompt", "Technique", "Output", "Notes"])
    df.to_csv(RESULT_FILE, index=False)

def generate_prompt(prompt, technique):
    
    if technique == "Original":
        return prompt
        
    elif technique == "Zero-shot":
        return f"Answer clearly:\n{prompt}"
        
    elif technique == "One-shot":
        return f"""Example:
Q: What is AI?
A: AI simulates human intelligence.

Now answer:
Q: {prompt}
A:"""
        
    elif technique == "Few-shot":
        return f"""Examples:
Q: What is AI?
A: AI simulates human intelligence.

Q: What is ML?
A: ML is a subset of AI.

Now answer:
Q: {prompt}
A:"""
        
    elif technique == "Role-based":
        return f"You are an expert AI assistant. {prompt}"
        
    elif technique == "Constraint-based":
        return f"{prompt}\nAnswer in 2 lines only."
        
    return prompt


def save_result(prompt, technique, output, notes):
    df = pd.read_csv(RESULT_FILE)
    new_row = pd.DataFrame([[prompt, technique, output, notes]],
                           columns=["Prompt", "Technique", "Output", "Notes"])
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(RESULT_FILE, index=False)
    return "Saved successfully!"


interface = gr.Interface(
    fn=generate_prompt,
    inputs=[
        gr.Textbox(label="Enter Prompt"),
        gr.Dropdown(
            ["Original", "Zero-shot", "One-shot", "Few-shot", "Role-based", "Constraint-based"],
            label="Technique"
        )
    ],
    outputs=gr.Textbox(label="Generated Prompt"),
    title="Task 9 - Prompt Engineering Tester"
)

save_ui = gr.Interface(
    fn=save_result,
    inputs=[
        gr.Textbox(label="Original Prompt"),
        gr.Textbox(label="Technique"),
        gr.Textbox(label="LLM Output (Paste from ChatGPT)"),
        gr.Textbox(label="Notes (Quality, clarity, etc.)")
    ],
    outputs=gr.Textbox(label="Status"),
    title="Save Results"
)

app = gr.TabbedInterface([interface, save_ui], ["Test Prompt", "Save Results"])

app.launch()