# From ML to MLOps

**Software Engineering & MLOps Foundations for Machine Learning Practitioners**

You can train an ML model. This course helps you take the next step:

> I can engineer, test, package, deploy, monitor, track, automate, and operate
> an ML system.

Built around **one real project** — a US Flight Delays classification model —
that you'll grow from a fragile notebook into a production-quality ML system
one lesson at a time.

## Getting started

**Read the course online:**

> **[https://a7madmostafa.github.io/mlops-foundations-course/](https://a7madmostafa.github.io/mlops-foundations-course/)**

Start at **Module 1 — Lesson 1.1** and work forward. Every lesson leaves the
flight-delay project visibly better and still runnable.

**Or clone and read locally:**

```bash
git clone https://github.com/a7madmostafa/mlops-foundations-course.git
cd mlops-foundations-course
```

Open `index.html` in your browser — no server, no build required.

### The data

The course uses US flight-delay records from the **U.S. Bureau of
Transportation Statistics (BTS)** On-Time Performance database — public,
US-government open data.

Download the 2025 corpus (12 monthly files) from
[`us-flight-delays-2025`](https://www.kaggle.com/datasets/a7madmostafa/us-flight-delays-2025-bts-on-time-performance).
See `data/README.md` for the exact command and for how to pull the raw data
straight from BTS if you prefer.

## Who this is for

This course is for:

- ML practitioners who train models in notebooks and want to learn production engineering
- Data scientists going from experiments to production
- Python-savvy people new to software engineering
- Learners who want the "engineering" half of MLOps

## Prerequisites

You should be comfortable with:

- Python fundamentals
- NumPy and pandas
- Exploratory data analysis
- Supervised ML (classification and regression)
- scikit-learn
- Basic Streamlit
- Basic Git

You do **not** need to know software engineering, testing, Docker, CI/CD, or
any MLOps tooling — every concept is taught from the ground up, always after
the problem it solves.

New to machine learning? Start with the **[Arabic ML Bootcamp](https://github.com/a7madmostafa/Arabic_ML_Bootcamp)** — it covers the ML foundations this course builds on.

## How the course works

- **8 modules**, each a phase of the journey — from a fragile notebook to an
  operated ML system.
- Each module has **lessons**. Every lesson is one reading page you open
  directly in your browser, followed by hands-on steps.
- Every lesson ends with a **milestone**: the flight-delay project gets into a
  clearly better, still-runnable state.
- Every lesson ships the project's final `CODE/` so you can run exactly what
  the lesson produced.

**Problem** → **Concept** → **Tool** → **Apply to the Flight Delays project** → **Practice** — the rhythm of every lesson.

## Modules

| Module | Title | Lessons | What you'll learn |
|--------|-------|---------|-------------------|
| 1 | The Software Foundation | 5 | From fragile notebook to real project: structure, dependencies, configuration, robustness, tests |
| 2 | ML Lifecycle & Reproducibility | 6 | Experiment tracking, model registry, data versioning, DVC |
| 3 | Orchestration | 3 | DAGs, retries, failure handling, Airflow |
| 4 | APIs & Serving | 6 | REST, FastAPI, testing the API, performance |
| 5 | Containers & Security | 4 | Docker, compose, security basics for ML |
| 6 | Model Monitoring | 3 | Drift detection, data quality, Evidently |
| 7 | Automation & CI/CD | 4 | Pre-commit, CI, GitHub Actions, ML CD |
| 8 | Capstone & the ML Team | 2 | End-to-end operation, code review, on-call |

## The project

One binary-classification project — *will this flight arrive 15+ minutes late?* — is taken through the whole engineering journey. You download the real January 2025 BTS on-time-performance data once into the shared root `data/` directory, then reproduce every lesson from a clean clone.

- **Lesson 1.1** starts you in a deliberately fragile notebook and lets you name its problems.
- **Every later milestone** refactors, packages, tests, serves, containerizes, monitors, or automates the same project.
- **Verify every step** by diffing one lesson's `CODE/` with the previous one.

## License

Content of this course is provided for learning purposes. The flight-delay
data is US-government public data (public domain). See `data/README.md` for
sourcing details.
