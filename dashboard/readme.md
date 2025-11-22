# Optimal Transport Dashboard

A Dash-based web application for exploring and comparing Optimal Transport solver performance across synthetic and real datasets.

## Features

- **Descriptive Analysis**  
  Violin plots, histograms, scatter plots, heatmaps and pivot tables of runtime, error, and resource usage.
- **Inferential Analysis**  
  Statistical tests (Shapiro–Wilk, Levene, paired t‑test/Wilcoxon, Friedman + Nemenyi, RM‑ANOVA, Pearson/Spearman) on solver runtimes.
- **Configurable data source** via environment variables.
- **Production-ready** Dockerfile with Gunicorn for scalable deployment.

---

## Table of Contents

1. [Prerequisites](#prerequisites)  
2. [Getting Started](#getting-started)  
3. [Configuration](#configuration)  
4. [Running Locally (Docker)](#running-locally-docker)  
5. [Running Locally (Python)](#running-locally-python)  
6. [Environment Variables](#environment-variables)  
7. [License](#license)

---

## Prerequisites

- Docker & Docker Compose (for containerized setup)  
- Python 3.9+ (for local development)  
- `git` (optional, to clone the repository)

---

## Getting Started

```bash
git clone https://github.com/your‑org/ot‑dashboard.git
cd ot‑dashboard
````

---

## Configuration

Copy the example environment file and adjust paths as needed:

```bash
cp .env.example .env
```

Edit `.env`:

```dotenv
# Path to results directory containing CSV subfolders
RESULTS_PATH=../results/local

# Comma‑separated list of subdirectories under RESULTS_PATH
SUBDIRS=.,problem_set_1,1d,2d,3d

# If set to “none” (case‑insensitive) or left empty, no source‑file filtering is applied.
# Otherwise only rows matching this filename in __source_file__ will be loaded.
SOURCE_FILE_FILTER=gen_repr.csv
```

---

## Running Locally (Docker)

1. **Build the image**

   ```bash
   docker build -t ot-dashboard .
   ```

2. **Run the container**

   ```bash
   docker run --env-file .env -p 8050:8050 ot-dashboard
   ```

3. Open your browser at [http://localhost:8050](http://localhost:8050).

---

## Running Locally (Python)

1. **Create & activate a virtual environment**

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Set environment variables**

   ```bash
   source .env
   ```

4. **Start the app**

   ```bash
   python app.py
   ```

5. Visit [http://localhost:8050](http://localhost:8050).

---

## Environment Variables

| Variable             | Default            | Description                                                                 |
| -------------------- | ------------------ | --------------------------------------------------------------------------- |
| `RESULTS_PATH`       | `../results/local` | Base path where CSV subfolders live.                                        |
| `SUBDIRS`            | `.`                | Comma‑separated list of subdirectories to scan under `RESULTS_PATH`.        |
| `SOURCE_FILE_FILTER` | `none`             | Filename to filter on `__source_file__`; set to `none` or empty to disable. |

---

