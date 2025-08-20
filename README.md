# Permission-Aware RAG

This repository contains the official implementation for the paper:  
**Permission-Aware RAG: Identity and Access Management (IAM)-Based Access Filtering in Multi-Resource Environments**  

---

## Architecture

The framework is designed to perform retrieval from a centralized vector store while delegating access control to the native IAM systems of each data provider (e.g., GCP, AWS, Keycloak). This preserves the policy autonomy of each source and avoids the risks of manual policy merging.  

The overall architecture is detailed in our paper (Figure 3).  

---

## Key Features

- **Fine-Grained Access Control**: Enforces document-level permissions for RAG pipelines.  
- **Native IAM Integration**: Delegates permission checks to provider-native IAM systems like GCP IAM, AWS IAM, and Keycloak in real-time.  
- **Heterogeneous Environments**: Supports multiple cloud providers and policy models (RBAC and ABAC) simultaneously.  
- **No Policy Merging**: Avoids the complexity and security risks associated with merging diverse access policies.  

---

---

## Repository Structure

- `/scripts`: End-to-end Python scripts for running the pipeline, from data preparation to evaluation.  
- `/src`: Core modules of the framework, including the `PermissionRetriever`, `CredentialManager`, and `VectorStore`.  
- `/artifacts`: Stores input data, credentials, FAISS index, and evaluation results.  

---

## Dataset

This project uses the **HotpotQA** dataset for multi-hop question answering experiments.  
Specifically, we rely on the *distractor* split (`hotpot_dev_distractor_v1.json`) for evaluation.  

### Download

You can download the dataset using:

```bash
curl -Lo artifacts/hotpot_dev_distractor_v1.json https://hotpotqa.s3.amazonaws.com/hotpot_dev_distractor_v1.json
```
The dataset file should be placed in the artifacts/ directory.

## Setup & Installation

### 1. Prerequisites

- Python 3.10+ and Conda  
- A Google Cloud Platform (GCP) project with billing enabled  
- An Amazon Web Services (AWS) account  

### 2. Clone Repository

```bash
git clone https://github.com/your-username/permission-aware-rag.git
cd permission-aware-rag
```

### 3. Set Up Environment and Dependencies

We recommend using Conda for environment management.

```bash
conda create -n permission-aware-rag python=3.10.18
conda activate permission-aware-rag

# Install dependencies
pip install -r requirements.txt
```

**requirements.txt**
```txt
python-dotenv
google-api-python-client
oauth2client
boto3
tqdm
google-cloud-storage
numpy
faiss-cpu
sentence-transformers
langchain
langchain-openai
ujson
```

### 4. Cloud Provider Setup

- **GCP**  
  1. Create a service account with **Owner** and **Storage Admin** roles.  
  2. Enable the **Identity and Access Management (IAM) API**.  
  3. Download the service account JSON key file.  

- **AWS**  
  1. Create an IAM user with `AmazonS3FullAccess` and `IAMFullAccess`.  
  2. Generate an access key and secret key.  

### 5. Configure Environment Variables

Create a `.env` file in the project root with the following content:

```env
# OpenAI
OPENAI_API_KEY="sk-..."

# GCP
GCP_PROJECT_ID="your-gcp-project-id"
GCP_ADMIN_KEY_PATH="./artifacts/your-gcp-service-account-key.json"
GCP_BUCKET_LOCATION="asia-northeast3"

# AWS
AWS_REGION="YOUR_REGION" # ex.ap-northeast-2
AWS_ADMIN_ACCESS_KEY="YOUR_AWS_ACCESS_KEY"
AWS_ADMIN_SECRET_KEY="YOUR_AWS_SECRET_KEY"
```

---

## Running the Pipeline

The scripts in `/scripts` should be run **in order from 01 to 09**.

```bash
# Activate environment
conda activate permission-aware-rag
~permission-aware-rag$ export PYTHONPATH=$(pwd)
# Run the pipeline
python scripts/01_generate_access_model.py
python scripts/02_gcp_aws_resource_creator.py
python scripts/03_prepare_documents.py
python scripts/04_upload_resources.py
python scripts/05_generate_user_accessible_file_list.py
python scripts/06_generate_question_list.py
python scripts/07_run_ingestion_manager.py
python scripts/08_run_quantitative_evaluation.py
python scripts/09_run_latency_evaluation.py
```

### Script Descriptions

- **01_generate_access_model.py**  
  Defines IAM rules, users, and resources in a JSON access model.  

- **02_gcp_aws_resource_creator.py**  
  Provisions buckets and IAM users on GCP and AWS.  

- **03_prepare_documents.py**  
  Processes HotpotQA dataset into document files.  

- **04_upload_resources.py**  
  Uploads documents into cloud storage.  

- **05_generate_user_accessible_file_list.py**  
  Generates ground-truth user-to-document access mapping.  

- **06_generate_question_list.py**  
  Extracts multi-hop questions from HotpotQA for evaluation.  

- **07_run_ingestion_manager.py**  
  Embeds documents, builds FAISS index, and stores metadata.  

- **08_run_quantitative_evaluation.py**  
  Runs the **large-scale quantitative evaluation** over HotpotQA, measuring EM/F1 and their relationship to permission coverage.  
  This script can also be adapted for **qualitative scenario-based evaluation** by modifying the IAM settings and evaluation cases.

- **09_run_latency_evaluation.py**  
  Evaluates latency overhead of real-time IAM permission checks.  

---

## License

This project is licensed under the Apache License 2.0.  
See the [LICENSE](./LICENSE) file for details.


## Acknowledgements
This project makes use of the [HotpotQA dataset](https://github.com/hotpotqa/hotpot).  
We thank the authors for providing this resource.
