# ⚡ LAPSPEC AI: LAPTOP PRICE VALUATION ENGINE

An end-to-end Machine Learning web application that predicts laptop market prices based on hardware specifications. The project covers the complete data science lifecycle—from raw Kaggle data processing to model serialization and live cloud deployment via a premium dark-steel interactive dashboard.

🚀 Live Dashboard Link: [LINK](https://lapspec.streamlit.app/)

---

## 📊 Dataset & Features
The baseline dataset was sourced from **[Kaggle] It contains structural and performance details for over 1,300 laptops.

### Target Variable
* `Price` (Actual market cost of the configuration)

### Features Processed
* **Hardware Identifiers:** `Company`, `TypeName`, `OpSys`
* **Performance Chips:** `Cpu` (Processor Series), `Gpu` (Graphics Card Core)
* **Memory Blocks:** `Ram` (GB Memory Volume), `Memory` (Storage Volume Drive Class)
* **Form Factor:** `Inches` (Display Size), `ScreenResolution` (Native Pixel Matrix array), `Weight` (kg)

---

## 🛠️ My Data Engineering & Core Machine Learning Workflow

### 1. Data Cleaning Phase (Executed by Me)
* **Missing Value Integrity:** Audited the dataset completely using `df.isna().sum()`, removing incomplete records with null prices or missing foundational metrics.
* **String Parsing & Standardization:** Stripped text units (`GB` and `kg`) from the raw data fields to transform `Ram` and `Weight` from text objects into proper numeric arrays (`int` and `float`).
* **Logical Filtering:** Cleaned structural anomalies and filtered out impossible zero values (e.g., zero laptop weight/screen inches) that would skew the algorithm.
* **Deduplication:** Audited data strings, confirming zero structural duplicates across the matrix array to prevent training data leakage.

### 2. Categorical Encoding & Asset Serialization (Executed by Me)
* To prepare high-cardinality text columns for mathematical model training without writing manual dictionaries, I created individual instances of `LabelEncoder` from `scikit-learn` to transform `Company`, `TypeName`, `ScreenResolution`, `Cpu`, `Memory`, `Gpu`, and `OpSys`.
* To prevent state loss upon notebook termination, I permanently serialized each encoder's state directly to local disk arrays using `joblib.dump()`.

### 3. Model Training & Optimization Phase (Executed by Me)
* **Baseline Training:** Programmed and trained a standard `RandomForestRegressor()`, establishing an initial baseline accuracy score of **81.00%**.
* **Production Optimization:** To push past the 90% threshold, I upgraded the pipeline to a sequential **`GradientBoostingRegressor`** engine. By tuning tree iteration parameters (`n_estimators=300`) and configuring the learning rate, the model effectively captured non-linear hardware interactions to maximize accuracy.
* **Feature Alignment Automation:** I engineered a self-healing data alignment rule inside the application script that queries the trained model's `.feature_names_in_` memory to auto-sort incoming browser inputs dynamically before computing prices.

---

## 💻 Web App Architecture

The final user dashboard is designed with a premium, low-latency **Cyber-Steel & Neon Amber Gold** tech aesthetic using **Streamlit**.

### Core Visual Enhancements
* **Zero-Lag Celebration Blast:** Bypasses slower cloud-based dependencies by rendering lightweight JavaScript-canvas confetti streams natively inside the client's host frame.
* **Clean Layout Constraints:** Completely removed empty placeholder shapes and streamlined text structures to present input specifications in modern visual blocks.

---

## 📂 Project Repository Tree

```text
📦 laptop-price-valuation
 ┣ 📂 .streamlit
 ┃ ┗ 📜 config.toml          # Custom theme configuration parameters
 ┣ 📜 app.py                 # Core production dashboard script
 ┣ 📜 laptop_notebook.ipynb  # Dev notebook (Cleaning, Training, Serialization)
 ┣ 📜 requirements.txt       # Global system engine package dependencies
 ┣ 📜 README.md              # Documentation portal
 ┗ 📜 *_encoder.lib          # Serialized label encoder binaries
```

---

## 🚀 Local Installation & Deployment

Clone the directory path and configure the environment dependencies locally:

1. **Clone the repository:**
   ```bash
   git clone https://github.com
   cd YOUR_REPO_NAME
   ```

2. **Install requirements:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch the engine:**
   ```bash
   streamlit run app.py
   ```
