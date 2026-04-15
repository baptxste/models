from huggingface_hub import snapshot_download
 
snapshot_download(
    repo_id="cross-encoder/ms-marco-MiniLM-L-6-v2",
    local_dir="download_crossencoder/cross-encoder-ms-marco-MiniLM-L-6-v2"
)

print("Modèle téléchargé.")