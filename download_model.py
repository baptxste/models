import os
import argparse
from gliner import GLiNER

def main(model_name, save_dir):
    print(f"Downloading model '{model_name}'...")
    # This downloads the model from Hugging Face Hub
    model = GLiNER.from_pretrained(model_name)
    
    print(f"Saving model to local directory '{save_dir}'...")
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and save a GLiNER model locally.")
    parser.add_argument("--model", type=str, default="urchade/gliner_medium-v2.1", help="HuggingFace model name")
    parser.add_argument("--dir", type=str, default="downloaded_models", help="Directory to save the model")
    args = parser.parse_args()
    
    main(args.model, args.dir)
