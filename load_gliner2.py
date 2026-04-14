from gliner2 import GLiNER2
import argparse

def main(load_dir):
    print(f"Loading GLiNER2 model from local directory '{load_dir}'...")
    model = GLiNER2.from_pretrained(load_dir, local_files_only=True)
    
    text = "The quick brown fox jumps over the lazy dog in New York."
    labels = ["animal", "location"]
    
    print(f"\nTesting model with text: '{text}'")
    print(f"Labels to detect: {labels}")
    
    entities = model.extract_entities(text, labels)
    for entity in entities:
        print(f"Detected: {entity['text']} -> {entity['label']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load a locally saved GLiNER2 model.")
    parser.add_argument("--dir", type=str, default="downloaded_gliner2_models", help="Local directory containing the model")
    args = parser.parse_args()
    
    main(args.dir)
