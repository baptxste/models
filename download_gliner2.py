import argparse
from huggingface_hub import snapshot_download

def main(repo_id, save_dir):
    print(f"Downloading repository '{repo_id}' using huggingface_hub...")
    # This bypasses PyTorch dependency completely, just copies the files
    snapshot_download(repo_id=repo_id, local_dir=save_dir)
    print(f"Model saved to '{save_dir}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="fastino/gliner2-base-v1")
    parser.add_argument("--dir", type=str, default="downloaded_gliner2_models")
    args = parser.parse_args()
    main(args.model, args.dir)
