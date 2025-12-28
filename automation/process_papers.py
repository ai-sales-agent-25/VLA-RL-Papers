import os
import json
import google.generativeai as genai
from pathlib import Path
import shutil

# Setup Gemini
genai.configure(api_key=os.environ["GEMINI_API_KEY"])
model = genai.GenerativeModel('gemini-3-flash-preview')

RUBRIC_PROMPT = """
You are an expert Robotics AI researcher. Read the attached paper and return ONLY a JSON object:
{
  "title": "Full Paper Title",
  "category": "Systems and Scale | Algorithmic Foundations | Semantic Reasoning | Robustness and Reliability | Speed and Deployment",
  "summary": "One sentence summary",
  "key_concepts": ["concept1", "concept2"]
}
"""

def process_new_papers():
    inbox = Path("inbox")
    data_file = Path("data/papers.json")
    
    if not data_file.exists():
        data_file.parent.mkdir(exist_ok=True)
        database = []
    else:
        with open(data_file, "r") as f:
            database = json.load(f)

    # Track filenames already in database to avoid duplicates
    existing_filenames = {item.get('filename') for item in database}

    for pdf_path in inbox.glob("*.pdf"):
        if pdf_path.name == ".gitkeep": continue
        
        # --- DUPLICATION SHIELD ---
        if pdf_path.name in existing_filenames:
            print(f"Skipping {pdf_path.name} - Already processed. Cleaning inbox.")
            pdf_path.unlink() # Just delete the local copy
            continue
        
        print(f"--- Analyzing: {pdf_path.name} ---")
        try:
            # 1. Upload and Analyze
            paper_file = genai.upload_file(path=pdf_path)
            response = model.generate_content([paper_file, RUBRIC_PROMPT])
            
            # 2. Parse Data
            raw_json = response.text.replace("```json", "").replace("```", "").strip()
            paper_data = json.loads(raw_json)
            
            # 3. Organize Folder
            clean_category = paper_data["category"].split(". ")[-1]
            archive_path = Path("papers") / clean_category
            archive_path.mkdir(parents=True, exist_ok=True)
            
            # 4. Move File
            target_path = archive_path / pdf_path.name
            shutil.move(str(pdf_path), str(target_path)) # Using shutil for better reliability
            
            # 5. Update Metadata
            paper_data["filename"] = pdf_path.name
            paper_data["github_link"] = f"https://github.com/ai-sales-agent-25/VLA-RL-Papers/blob/main/{str(target_path)}"
            database.append(paper_data)
            print(f"Successfully filed under: {clean_category}")

        except Exception as e:
            print(f"Error: {str(e)}")

    # Save final database
    with open(data_file, "w") as f:
        json.dump(database, f, indent=2)

if __name__ == "__main__":
    process_new_papers()
