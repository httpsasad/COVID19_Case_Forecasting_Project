# Data

Raw CSV files are intentionally not committed by default because the source files are large and the upstream archive is publicly available.

Run:

```bash
python -m src.download_data
```

The downloader uses the official archived JHU CSSE URLs defined in `src/config.py`.

For a maintained source, see the WHO COVID-19 dashboard.
