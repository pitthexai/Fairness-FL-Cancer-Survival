# Fairness-FL-Cancer-Survival

This repository contains the source code, trained models, figures, and experimental results for the study:

**Explainable and Fairness-Aware Federated Learning for Cancer Survival Prediction: Social and Environmental Heterogeneity Across Cancer Types and Stages**

## Overview

This project presents an explainable and fairness-aware federated learning framework for five-year cancer survival prediction.

The study investigates three cancer types:

- Pancreatic cancer
- Stomach cancer
- Lung cancer

Each cancer cohort is evaluated across three disease stages:

- Localized
- Regional
- Distant

The federated learning framework is designed to enable collaborative model training without sharing raw patient-level data.

## Machine Learning Models

Three machine learning models are evaluated:

- Logistic Regression
- Multilayer Perceptron (MLP)
- XGBoost

Each model is trained under two experimental settings:

- **Baseline federated learning**
- **Fairness-aware federated learning**


## Dataset

The study uses data from the **Surveillance, Epidemiology, and End Results (SEER)** database.

Cancer cases diagnosed between **2004 and 2019** are included in the analysis.

To simulate a distributed healthcare environment, each stage-specific dataset is partitioned across three non-IID virtual clients.

## Fairness-Aware Federated Learning

The fairness-aware experiments incorporate socioeconomic and environmental indicators, including:

- Yost Index
- Social Vulnerability Index (SVI)
- Air Quality Index (AQI)

Fairness is evaluated through subgroup analyses using:

- True Positive Rate (TPR)
- False Positive Rate (FPR)

## Explainability

Model explainability is investigated using:

- SHAP (SHapley Additive exPlanations)
- LIME (Local Interpretable Model-Agnostic Explanations)

These methods are used to analyze the contribution of clinical, demographic, socioeconomic, and environmental variables to cancer survival predictions.

## Evaluation Metrics

Model performance is evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- PR-AUC
- Log Loss

Bootstrap resampling is used to estimate 95% confidence intervals for the reported performance metrics.

