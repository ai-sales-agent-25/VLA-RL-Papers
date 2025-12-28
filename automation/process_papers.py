import os
import json
import google.generativeai as genai
from pathlib import Path
import shutil

# Setup Gemini
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

# Detailed System Instruction with your favorite examples
SYSTEM_INSTRUCTION = """
You are an expert Robotics AI researcher specializing in VLA models and RL. 
Your task is to classify papers based on the VLA-RL Sorting Rubric.

CATEGORIES:
1. Systems & Scale (The "Heavy Lifters"): Training infra, GRPO, batch sizes.
2. Algorithmic Foundations (The "Math Architects"): Flow matching, SDEs, loss functions.
3. Semantic Reasoning (The "Thinkers"): CoT, sub-goals, System 2, tree search.
4. Robustness & Reliability (The "Shields"): OOD, sim-to-real, precision, messy environments.
5. Speed & Deployment (The "Fast Movers"): Hz, latency, quantization, edge compute.

OUTPUT FORMAT (STRICT JSON):
{
  "title": "Full Paper Title",
  "category": "Category Name",
  "bottleneck": "The specific Wall addressed (Hardware, Optimization, Complexity, Generalization, or Latency)",
  "justification": "Detailed explanation of why it fits this category based on the rubric.",
  "evidence": "Specific terminology or quotes found in the paper.",
  "key_concepts": ["concept1", "concept2"]
}

EXAMPLE OF DESIRED DEPTH:
Title: Hierarchical VLA Model Using Success and Failure Demonstrations (VINE)
Category: Semantic Reasoning (The "Thinkers")
Primary Bottleneck Addressed: The Complexity Wall.
Justification: This paper introduces a "System 2" for high-level reasoning and planning. It uses tree search, subgoals, and "failure-aware reasoning" to think through steps before execution. By using a scene-graph abstraction to propose subgoals, it addresses the logic and dependencies of long-horizon tasks.
Evidence: "High-level reasoning (System 2)," "Subgoal transitions," "Tree search," "2D scene-graph abstraction," "Long-horizon planning," and "Deliberation."

CRITICAL: Return ONLY a raw JSON object. No markdown, no conversational text.
"""

model = genai.GenerativeModel(
    model_name='gemini-3-flash-preview',
    system_instruction=SYSTEM_INSTRUCTION
)

def process_new_papers():
    inbox = Path("inbox")
    data_file = Path("data/papers.json")
    
    if not data_file.exists():
        data_file.parent.mkdir(exist_ok=True)
        database = []
    else:
        with open(data_file, "r") as f:
            database = json.load(f)

    existing_filenames = {item.get('filename') for item in database}

    for pdf_path in inbox.glob("*.pdf"):
        if pdf_path.name == ".gitkeep": continue
        
        if pdf_path.name in existing_filenames:
            print(f"Skipping {pdf_path.name} - Already processed.")
            pdf_path.unlink()
            continue
        
        print(f"--- Analyzing: {pdf_path.name} ---")
        try:
            # Upload the PDF
            paper_file = genai.upload_file(path=pdf_path)
            
            # Request classification
            prompt = "Read this paper and provide a high-depth JSON classification following the rubric and formatting rules."
            response = model.generate_content([paper_file, prompt])
            
            # Clean and parse JSON
            clean_text = response.text.strip()
            if clean_text.startswith("```json"): clean_text = clean_text[7:-3]
            elif clean_text.startswith("```"): clean_text = clean_text[3:-3]
            
            paper_data = json.loads(clean_text)
            
            # Folder Organization
            category_name = paper_data["category"]
            # Extract name to handle '4. Robustness & Reliability' vs 'Robustness & Reliability'
            clean_category = category_name.split(". ")[-1].split(" (")[0].strip()
            
            archive_path = Path("papers") / clean_category
            archive_path.mkdir(parents=True, exist_ok=True)
            
            # Move File
            target_path = archive_path / pdf_path.name
            shutil.move(str(pdf_path), str(target_path))
            
            # Add metadata for GitHub
            paper_data["filename"] = pdf_path.name
            paper_data["github_link"] = f"https://github.com/ai-sales-agent-25/VLA-RL-Papers/blob/main/{str(target_path)}"
            
            database.append(paper_data)
            print(f"Successfully filed under: {clean_category}")

        except Exception as e:
            print(f"Error parsing JSON: {str(e)}")
            print(f"Raw Response: {response.text}")

    # Save database
    with open(data_file, "w") as f:
        json.dump(database, f, indent=2)

if __name__ == "__main__":
    process_new_papers()
