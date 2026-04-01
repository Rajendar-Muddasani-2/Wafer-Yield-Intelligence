<style>
h1 {
  background-color: #e6f2ff;
  padding: 10px;
  border-left: 5px solid #91caff;
}
h2 {
  background-color: #f6ffed;
  padding: 8px;
  border-left: 5px solid #b7eb8f;
}
h3 {
  background-color: #fffbe6;
  padding: 6px;
  border-left: 5px solid #ffd666;
}
</style>

# Wafer Yield Intelligence: Project Walkthrough (STAR Method)

This document provides a detailed walkthrough of the Wafer Yield Intelligence project, following the STAR (Situation, Task, Action, Result) framework.

---

## Project Overview

**Project Name:** Wafer Yield Intelligence

**One-Line Description:** An automated system that uses machine learning to predict wafer disposition decisions, reducing manual analysis time by 80% and saving over $500K annually.

---

## Situation

**Pain Point:** At a major semiconductor manufacturer, engineers were spending 4-8 hours manually analyzing every production lot of the A1G automotive-grade semiconductor family. This process was not only slow and expensive but also led to inconsistent retest decisions, as different engineers had different criteria. With production volumes increasing, this manual approach was unsustainable and was leading to revenue loss from unnecessarily scrapped lots.

**Requirement:** The business needed a scalable, automated solution to standardize and accelerate the wafer disposition process. The key requirements were to reduce analysis time by at least 80% while maintaining a decision accuracy of over 95% to minimize costs associated with both unnecessary scrap and retesting.

---

## Task

**My Role:** As the lead ML Engineer, I was responsible for the end-to-end design, development, and deployment of the Wafer Yield Intelligence system.

**What I was tasked to solve:** My primary objective was to replace the manual analysis process with a robust, data-driven system. This involved creating a full ETL pipeline to process raw test data, developing and training machine learning models to predict defect patterns and retest outcomes, and integrating the solution into the existing production workflow.

---

## Action

I developed a comprehensive solution centered around a 4-stage ETL pipeline and two distinct sets of machine learning models.

### 1. Four-Stage ETL Pipeline

I designed a modular ETL pipeline using Python and Pandas to process the raw test data and store it in an Oracle database. Each stage was an independent script, ensuring maintainability and clear data flow.

| Stage | Script | Purpose |
| :-- | :-- | :-- |
| **1. Info Extraction** | `lot_info.py` | Extracted metadata from raw test files and assigned unique task IDs. |
| **2. Yield Calculation** | `lot_data.py` | Calculated yield and offset metrics at the lot, wafer, and site levels. |
| **3. Statistical Analysis** | `lot_stats.py` | Computed 3-sigma control limits for all test parameters to establish performance thresholds. |
| **4. ML Analysis** | `lot_analysis.py` | Performed ML inference using the trained models to generate final disposition decisions. |

### 2. Two-Set Machine Learning Approach

I recognized that wafer disposition required answering two separate questions: "What is the defect pattern?" and "Should we retest this lot?" This led me to develop two independent sets of models.

#### A. Pattern Recognition Models

*   **Goal:** To classify wafer defect patterns into one of nine categories (e.g., Center, Donut, Scratch).
*   **Features:** I engineered **59 distinct features** from the wafer map images using computer vision techniques, including 13 density features, 40 Radon transform features, and 6 geometric features.
*   **Models:** I trained and evaluated several models, with a **Voting Ensemble** of Gradient Boosting and an MLP Neural Net achieving the highest accuracy of **98%**.
*   **Training:** I used transfer learning, training the models on the public WM-811K dataset (811,457 labeled wafers) and fine-tuning them on the A1G production data.

#### B. Retest Prediction Models

*   **Goal:** To predict whether a failed lot would pass if retested.
*   **Features:** I used yield metrics, SBIN patterns, and statistical offsets specific to the A1G product line.
*   **Models:** A **Voting Classifier** combining KNN, Random Forest, and an MLP achieved the best performance with **95% accuracy**.
*   **Training:** This model was trained exclusively on historical A1G production data.

### 3. System Architecture & Design

To ensure the system was robust and production-ready, I implemented several key design patterns:

*   **Configuration Management:** A central `Config` class loaded all database credentials and settings from a `.env` file, eliminating hardcoded secrets.
*   **Database Connection Pooling:** I used SQLAlchemy to manage a pool of database connections, improving performance and preventing connection leaks.
*   **Modular Codebase:** The entire system was designed with modularity in mind, with reusable utilities for database connections and configuration.

---

## Result

### Quantitative Impact

| Metric | Result |
| :-- | :-- |
| **Time Savings** | **80% reduction** in analysis time (from 4-8 hours to < 1 hour). |
| **Cost Savings** | Over **$500,000** in estimated annual savings from optimized retest decisions and reduced scrap. |
| **Model Accuracy** | **98%** for Pattern Recognition and **95%** for Retest Prediction. |
| **Decision Consistency** | **100%** consistency, removing human subjectivity. |

### Business Value

The Wafer Yield Intelligence system transformed the disposition process from a manual bottleneck into a strategic advantage. It not only freed up hundreds of engineering hours but also provided valuable insights into systematic manufacturing issues through automated pattern recognition. The system's scalability ensures that it can handle significant future increases in production volume without requiring additional headcount.

---

## Technical Stack Summary

| Category | Technologies |
| :--- | :--- |
| **Languages** | Python |
| **Data Processing** | Pandas, NumPy |
| **Database** | Oracle, SQLAlchemy, cx_Oracle |
| **Machine Learning** | Scikit-learn (Gradient Boosting, Random Forest, KNN), TensorFlow |
| **Computer Vision** | Scikit-image |
| **Visualization** | Matplotlib, Seaborn |
