"""
Modal deployment for the Streamlit UI.

Deploys ``streamlit_app.py`` as a web server on Modal. Reads Modal proxy-auth
credentials from a Modal Secret named ``modal-proxy-auth`` (keys: MODAL_KEY,
MODAL_SECRET).

ONE-TIME SETUP
--------------
In the Modal dashboard:

    Secrets → New Secret
        Name : modal-proxy-auth
        Key  : MODAL_KEY      value: wk-xxxxxxxxxxxxxxxxxxxx
        Key  : MODAL_SECRET   value: ws-xxxxxxxxxxxxxxxxxxxx

(Token pair from Settings → Proxy Auth Tokens.)

DEPLOY
------
    modal deploy deploy/modal_streamlit.py

Modal prints the live URL — pattern is
    https://harsh1243--text-to-sql-ui.modal.run
"""

import modal

# Remote paths inside the image. We lay out /root/app/ so the Streamlit
# process can find the bundled `retriver` package on PYTHONPATH.
APP_DIR = "/root/app"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        # Streamlit + its one runtime dep
        "streamlit==1.36.0",
        "requests==2.32.3",
        # Retriever deps — needed because the retriever runs inside this
        # container, not on Modal's GPU pool.
        "sentence-transformers==2.6.1",
        "rank-bm25==0.2.2",
        # sentence-transformers pulls torch transitively, but pinning the
        # same version as the GPU app keeps behaviour identical.
        "torch==2.3.1",
    )
    .add_local_dir("retriver",     remote_path=f"{APP_DIR}/retriver")
    .add_local_file("streamlit_app.py", remote_path=f"{APP_DIR}/streamlit_app.py")
)

app = modal.App("text-to-sql-ui", image=image)


@app.function(
    secrets=[modal.Secret.from_name("modal-proxy-auth")],
    # CPU is fine — the UI process doesn't touch a GPU. The Modal GPU
    # endpoints are reached over HTTP.
    cpu=1.0,
    memory=2048,
    timeout=600,
    scaledown_window=300,
)
@modal.web_server(8000)
def ui():
    """Start the Streamlit server inside the container. Modal forwards
    external HTTP traffic to port 8000."""
    import subprocess
    subprocess.Popen([
        "streamlit", "run", "streamlit_app.py",
        "--server.port=8000",
        "--server.address=0.0.0.0",
        "--server.headless=true",
        "--browser.gatherUsageStats=false",
        "--server.fileWatcherType=none",
    ])


@app.local_entrypoint()
def main():
    """Print the URL the user should open."""
    print(
        "Run `modal deploy deploy/modal_streamlit.py` to deploy.\n"
        "Once deployed, Modal prints a URL like:\n"
        "    https://harsh1243--text-to-sql-ui.modal.run"
    )
