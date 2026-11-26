# Genomic Ancestry Pipeline — Spec-Driven Machine Learning

## Overview

The Genomic Ancestry Pipeline is an automated machine learning workflow that predicts ancestral population from raw single-nucleotide polymorphism (SNP) genomic data. Designed using a spec-driven development (SDD) methodology, it establishes an end-to-end pipeline that translates structured biological requirements into a reproducible model training and serverless inference architecture on AWS.

---

## Live Demo

<a href="https://mattjtravers.github.io/genomic-ancestry-pipeline/"><img src="assets/open-demo-button.svg" alt="Open Demo"/></a>

Pick one of the bundled samples to see its result instantly, or upload your own single-sample chromosome-15 VCF (GRCh37) and click **Run Prediction**. Each result shows the predicted ancestral population, a confidence score, and the genomic markers that contributed most to the prediction.

> **About response times:** the demo is a static page on GitHub Pages, so it loads immediately, and sample results are precomputed. Uploaded files go to the live inference API on AWS Lambda; the page warms it up as it loads, but the first prediction after a quiet period can still take a few extra seconds.

---

## Architectural Blueprint

```text
[Raw SNP Data / VCF Input]
            │
            ▼
┌───────────────────────┐
│   Data Ingestion      │ ──► Validates file headers and chromosomal coordinates
└───────────┬───────────┘
            │ (Clean Data Stream)
            ▼
┌───────────────────────┐
│ Feature Engineering   │ ──► Executes median imputation & quality-score filtering
└───────────┬───────────┘
            │ (Feature Matrices → Amazon S3)
            ▼
┌───────────────────────┐
│  SageMaker Training   │ ──► Executes XGBoost training with cross-validation
└───────────┬───────────┘
            │ (Model Artifact → Amazon S3)
            ▼
┌───────────────────────┐
│  AWS SAM Deployment   │ ──► Configures API Gateway and AWS Lambda environment
└───────────┬───────────┘
            │ (Public HTTP API: POST /predict, GET /health)
            ▼
 [Static Site on GitHub Pages] ──► Renders ancestral population prediction, confidence, and top SNP markers

```

The system coordinates data from ingestion to deployment. Each state transition is managed by strict interface definitions, ensuring data validation occurs prior to handoffs between local compute and AWS infrastructure components.

---

## Features

* **Spec-Driven Ingestion Pipelines:** Translates structural biological specifications into deterministic Python data pipelines, using Pydantic to enforce runtime schema validation and eliminate silent data corruption.
* **Production AWS Infrastructure via IaC:** Provisions and manages the entire cloud footprint—including API Gateway, AWS Lambda, and IAM roles—using AWS SAM templates to guarantee zero configuration drift.
* **Automated Cloud Training Workflows:** Coordinates data handoffs from local environments to AWS SageMaker for structured XGBoost training, utilizing Amazon S3 as a secure feature and model artifact store.
* **Enterprise Observability & Logging:** Integrates structured logging directly into the Python application runtime, enabling native AWS CloudWatch monitoring and alerting without custom parsing overhead.

---

## Build & Health Status

The pipeline's integrity and test suites are automatically verified on every commit:

![CI Status](https://github.com/mattjtravers/genomic-ancestry-pipeline/actions/workflows/ci.yaml/badge.svg?branch=main)
