import os
import json
import google.generativeai as genai
from pathlib import Path

# Setup Gemini using the model from your screenshot
genai.configure(api_key=os.environ["GEMINI_API_KEY"])
model = genai.GenerativeModel('gemini-3-flash-preview')

RUBRIC_PROMPT = """
You are an expert Robotics AI researcher. Your job is to read the attached paper and classify it into ONE of these categories based on its primary focus:

1. Systems and Scale (Focus on training infra, memory, distributed computing)
2. Algorithmic Foundations (Focus on loss functions, math, Flow matching, SDEs)
3. Semantic Reasoning (Focus on long-horizon planning, CoT, sub-goals)
4. Robustness and Reliability (Focus on sim-to-real, domain shift, active correction)
5. Speed and Deployment (Focus on Hz, latency, quantization, edge compute)

Return ONLY a JSON object with this exact structure:
{
  "title": "Full Paper Title",
  "category": "Exact Category Name from the list above",
  "summary": "One sentence summary of the paper's contribution",
  "key_concepts": ["concept1", "concept2", "concept3"]
}
"""

def process_new_papers():
    inbox = Path("inbox")
    data_file = Path("data/papers.json")
    
    # Ensure directory structure exists
    if not data_file.exists():
        data_file.parent.mkdir(exist_ok=True)
        database = []
    else:
        with open(data_file, "r") as f:
            database = json.load(f)

    # Scan for new PDFs
    for pdf_path in inbox.glob("*.pdf"):
        print(f"--- Analyzing: {pdf_path.name} ---")
        
        try:
            # 1. Upload PDF to Gemini
            paper_file = genai.upload_file(path=pdf_path)
            
            # 2. Generate Classification
            response = model.generate_content([paper_file, RUBRIC_PROMPT])
            
            # 3. Clean and parse JSON
            raw_json = response.text.replace("```json", "").replace("```", "").strip()
            paper_data = json.loads(raw_json)
            
            # 4. Folder Organization Logic
            category_dir_name = paper_data["category"]
            # Remove any numbering if the model includes it (e.g. "3. Semantic Reasoning" -> "Semantic Reasoning")
            clean_category = category_dir_name.split(". ")[-1] if ". " in category_dir_name else category_dir_name
            
            archive_path = Path("papers") / clean_category
            archive_path.mkdir(parents=True, exist_ok=True)
            
            # 5. Move File and Update Database
            target_path = archive_path / pdf_path.name
            pdf_path.rename(target_path)
            
            paper_data["filename"] = pdf_path.name
            paper_data["github_link"] = f"https://github.com/ai-sales-agent-25/VLA-RL-Papers/blob/main/{str(target_path)}"
            database.append(paper_data)
            
            print(f"Filed under: {clean_category}")

        except Exception as e:
            print(f"Error processing {pdf_path.name}: {str(e)}")

    # Save updated database
    with open(data_file, "w") as f:
        json.dump(database, f, indent=2)

if __name__ == "__main__":
    process_new_papers()
