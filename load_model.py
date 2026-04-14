from gliner import GLiNER
import argparse

def main(load_dir):
    print(f"Loading GLiNER model from local directory '{load_dir}'...")
    # Load the model from the local directory instead of Hugging Face Hub
    model = GLiNER.from_pretrained(load_dir, local_files_only=True)
    
    # Test the model with some text
    text = "The quick brown fox jumps over the lazy dog in New York."
    labels = ["animal", "location"]
    
    print(f"\nTesting model with text: '{text}'")
    print(f"Labels to detect: {labels}")
    
    entities = model.predict_entities(text, labels)
    for entity in entities:
        print(f"Detected: {entity['text']} -> {entity['label']} (Score: {entity['score']:.4f})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load a locally saved GLiNER model.")
    parser.add_argument("--dir", type=str, default="downloaded_models", help="Local directory containing the model")
    args = parser.parse_args()
    
    main(args.dir)
