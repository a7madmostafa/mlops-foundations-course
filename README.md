# From ML to MLOps

**Software Engineering & MLOps Foundations for Machine Learning Practitioners**

You can train an ML model. This course helps you take the next step:

> I can engineer, test, package, deploy, monitor, track, automate, and operate
> an ML system.

Built around **one real project** — a US Flight Delays classification model —
that you'll grow from a fragile notebook into a production-quality ML system
one lesson at a time.

---

## Who this is for

You should already know Python, pandas, NumPy, scikit-learn, basic Streamlit,
and basic Git, and be comfortable with exploratory analysis and supervised
learning (you've trained a model or two).

You do **not** need to know software engineering, testing, Docker, CI/CD, or
any MLOps tooling — every concept is taught from the ground up, always after
the problem it solves.

## How the course works

- **8 modules**, each a phase of the journey — from a fragile notebook to an
  operated ML system.
- Each module has **lessons**. Every lesson is one reading page you open
  directly in your browser, followed by hands-on steps.
- Each lesson ends with a **milestone**: the flight-delay project gets into a
  clearly better, still-runnable state.
- Every lesson ships the project's final `CODE/` so you can run exactly what
  the lesson produced.

Start with **Module 1 — The Software Foundation**.

### The roadmap

| Module | Phase |
|--------|-------|
| 1 | The software foundation — from fragile notebook to real ML project |
| 2 | ML lifecycle & reproducibility |
| 3 | Orchestration — pipelines & workflows |
| 4 | APIs & serving |
| 5 | Containers & security |
| 6 | Model monitoring |
| 7 | Automation & CI/CD |
| 8 | Capstone & working on an ML team |

## Getting started

Clone the repository:

```bash
git clone <this-repo-url>
```

then open **`index.html`** in your browser (double-click it). No server, no
build — the whole course is static HTML. GitHub Pages hosts the same site
for online reading.

### The data

The course uses US flight-delay records from the **U.S. Bureau of
Transportation Statistics (BTS)** On-Time Performance database — public,
US-government open data.

- A small baseline month ships inside the first lesson's `CODE/data/`.
- The full 2025 corpus (12 monthly files) is on Kaggle:
  [`us-flight-delays-2025`](https://www.kaggle.com) — see `data/README.md`
  for the exact download command and for how to pull the raw data straight
  from BTS if you prefer.

## Repository layout

| Path | What it is |
|---|---|
| `index.html` | The course home (open this) |
| `assets/style.css` | The shared design system |
| `modules/module-01/` … `module-08/` | The modules, each with lessons and their runnable `CODE/` |
| `data/` | The course dataset (download from Kaggle, see `data/README.md`) |

## License

Content of this course is provided for learning purposes. The flight-delay
data is US-government public data (public domain). See `data/README.md` for
sourcing details.