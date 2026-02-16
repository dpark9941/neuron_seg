# NeuronSeg Backend (Zarr → Neuroglancer)

FastAPI serves `backend/sessions/` as static files under `/data/` so Neuroglancer can load Zarr metadata + chunks.

## Run (macOS)


bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt


## HTTPS cert (needed for https://neuroglancer-demo.appspot.com)

brew install mkcert
mkcert -install
mkcert localhost 127.0.0.1
Start backend (HTTPS)
python -m uvicorn app:app --reload --port 8001 \
  --ssl-keyfile localhost+1-key.pem \
  --ssl-certfile localhost+1.pem

## Check (should return JSON):

https://127.0.0.1:8001/data/demo/input.zarr/volumes/raw/.zarray

Open the viewer dashboard (index.html)
cd backend
python -m http.server 3000

## Open:

http://localhost:3000/public/

## Click:

Open RAW in new tab

Open RAW + SEG overlay in new tab

Zarr sources (what Neuroglancer loads)
raw: zarr://https://127.0.0.1:8001/data/demo/input.zarr/volumes/raw

seg: zarr://https://127.0.0.1:8001/data/demo/input.zarr/volumes/labels/neuron_ids

