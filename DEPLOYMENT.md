# HeatWise AI deployment

## Public GitHub repository

This repository is safe to publish after confirming that `.env` and
`.streamlit/secrets.toml` are untracked. Processed LiDAR rasters, OSM networks,
building footprints, and boundaries required by the app are included; raw
point clouds and intermediate rasters are intentionally excluded.

## Recommended production architecture

HeatWise is a Python/Streamlit application with WebSocket sessions and native
geospatial dependencies. Cloudflare Pages and Workers cannot run it directly.
The production path is:

`Visitor -> Cloudflare DNS/proxy -> always-on Docker origin -> Streamlit app`

The included `Dockerfile` packages the app and its scientific/geospatial
runtime. The included `render.yaml` is one origin option. A paid always-on
instance avoids sleeping and cold starts.

### Deploy the origin on Render

1. Push this repository to a public GitHub repository.
2. In Render, create a **Blueprint** from the repository. Render reads
   `render.yaml` and builds the Docker image.
3. Add these encrypted environment variables in Render:

```text
FORTYGUARD_API_KEY=...
FORTYGUARD_BASE_URL=https://api.fortyguard.com
MAPBOX_ACCESS_TOKEN=...
GROQ_API_KEY=...       # optional AI assistant
```

4. Wait for `/_stcore/health` to report healthy and test routing, geocoding,
   current location, FortyGuard temperature, meteorology, PET risk, shade,
   3D layers, and the AI assistant on the Render URL.
5. In Cloudflare DNS, create a proxied CNAME such as `heatwise.example.com`
   pointing to the Render hostname. In Render, add the same hostname under
   **Custom Domains** so TLS is issued correctly.

Cloudflare Tunnel can expose the laptop for a short demo, but it is not an
always-on deployment: the laptop, Streamlit process, and tunnel must all remain
running. It is therefore not the recommended submission URL.

## Alternative: Streamlit Community Cloud

1. Push this repository to a public GitHub repository.
2. At <https://share.streamlit.io>, choose **Create app**.
3. Select the repository, default branch, and `app.py`.
4. In **Advanced settings → Secrets**, add:

```toml
FORTYGUARD_API_KEY = "..."
FORTYGUARD_BASE_URL = "https://api.fortyguard.com"
MAPBOX_ACCESS_TOKEN = "..."
GROQ_API_KEY = "..." # optional
```

5. Deploy and keep the generated `*.streamlit.app` URL for the submission.

The public URL is stable and does not depend on the developer laptop. On the
free Community Cloud tier, an inactive app may sleep and wake on the next
visit. A paid always-on host is required if zero cold starts are mandatory.

## Data provenance shown in the app

- Hyperlocal temperature: FortyGuard Temperature API®
- Humidity, wind, radiation, cloud and precipitation: Open-Meteo
- Pedestrian/cycling networks and footprints: OpenStreetMap
- Canopy/building height and shade geometry: USGS 3DEP LiDAR-derived rasters

Never commit API keys. Local keys belong in `.env`; deployed keys belong in
the hosting provider's encrypted secrets panel.
