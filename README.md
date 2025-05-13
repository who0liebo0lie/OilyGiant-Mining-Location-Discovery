🛢️ OilyGiant Mining Location Discovery

This notebook helps identify the most profitable region for oil extraction for OilyGiant. It uses machine learning to predict oil production across three regions and evaluates economic viability based on predicted yields and investment risks.

# 🛢️ Optimizing Oil Well Placement for OilyGiant  
**A Machine Learning Case Study in Business Decision-Making, Linear Regression & Risk Analysis**

## 📘 Project Overview

In this project, we step into the role of a machine learning engineer for *OilyGiant*, a mining company evaluating three geographical regions for oil well development. The business objective is to **identify the most profitable region for drilling new oil wells**, under strict budget and risk constraints.

Leveraging **linear regression**, **bootstrap risk analysis**, and domain-specific financial modeling, we develop a solution that balances **profit maximization** with **risk minimization**—key considerations in production-scale machine learning deployments.

## 🎯 Objective

- Use machine learning to **predict oil reserves** based on geo-features.
- Select **200 oil wells** (from 500 samples per region) that maximize revenue.
- Respect budget of **$100 million** and a **loss risk threshold under 2.5%**.
- Recommend the most **financially sound region** for development.

## 💡 Business Constraints

- **Only linear regression** is permitted for reserve prediction.
- Development budget: **$100 million** for 200 wells.
- Profit per 1,000 barrels: **$4,500**
- Break-even point: **22,222 barrels per well**
- Average needed: **111.11 barrels per well**

---

## 🧪 Technical Approach

### 🔍 Data Preprocessing & EDA

- Handled each `geo_data_0/1/2.csv` file separately due to varying formats.
- Dropped duplicate and irrelevant columns (e.g., `id`).
- Verified feature consistency across regions.
- Defined financial constants for simulation:

```python
revenue_total = 100_000_000
wells = 200
revenue_per_well = 4500
break_even_volume = revenue_total / revenue_per_well
avg_volume_per_well = break_even_volume / wells  # 111.11 barrels
```

### 📈 Modeling & Evaluation

- Built **linear regression** models for each region.
- Calculated **Root Mean Squared Error (RMSE)** to assess model precision.
- Defined `calculate_rmse()` and `profit_calculation()` to estimate profitability and filter top 200 wells.
- Used **bootstrapping** to estimate **risk of loss** and **confidence intervals** on predicted profit.

---

## 📊 Results Summary

| Site | RMSE | Avg Predicted Volume (barrels) | Total Profit ($) | Avg Profit ($) | 2.5% Quantile ($) | 97.5% Quantile ($) | Risk of Loss |
|------|------|-------------------------------|------------------|----------------|-------------------|--------------------|---------------|
| **0** | 37.85 | 92.79 | 33,651,872 | 7,513,523 | 7,513,523 | 7,513,523 | **0.0%** |
| **1** | 0.89 | 68.73 | 24,150,867 | 2,993,594 | 2,993,594 | 2,993,594 | **0.0%** |
| **2** | 40.03 | 94.97 | 26,887,005 | -950,442 | -950,442 | -950,442 | **100.0%** |

---

## ✅ Recommendation

Despite having the **lowest average predicted reserves**, **Site 1** is the optimal choice for development:

- **Zero risk of loss**
- **Positive profit at all confidence levels**
- **Most stable return**, making it the most viable business decision

Though **Site 0** yields higher average profit, its volatility makes it riskier. **Site 2** is eliminated due to a guaranteed loss scenario (100% risk).

---

## 🔧 Tools & Technologies

- **Python** (pandas, NumPy, sklearn, matplotlib)
- **Linear Regression**
- **Bootstrapping**
- **EDA & Data Cleansing**
- **Business Modeling & Risk Analysis**

---

## 📂 Files

- `geo_data_0.csv`, `geo_data_1.csv`, `geo_data_2.csv`: Raw input data
- `profit_calculation()`: Function to simulate and sort profitable wells
- `calculate_rmse()`: RMSE error evaluation
- `.ipynb`: Full Jupyter Notebook with code, visuals, and findings

---

## 🧠 Takeaway

This project demonstrates the power of combining **machine learning** with **domain-aware business logic** to make confident, data-driven decisions. It highlights key ML competencies: model evaluation, real-world constraint handling, and stakeholder-oriented analysis.

🛠 Installation

Clone this repository or download the .ipynb file

Install required libraries:

bash
Copy
Edit
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
Run the notebook:

bash
Copy
Edit
jupyter notebook

🚀 Usage

Open the OilyGiant Mining Location Discovery.ipynb notebook and run all cells in order. The notebook performs the following:
Loads and visualizes oil production data for each region
Trains and evaluates a regression model to predict oil yield
Calculates risk (standard deviation, confidence intervals)
Compares ROI and loss probability under a fixed budget

📁 Project Structure
bash
Copy
Edit
OilyGiant Mining Location Discovery.ipynb   # Main analysis notebook
README.md                                   # Project documentation
images/                                     # Optional screenshots folder
⚙️ Technologies Used
Python 3.8+
Pandas
NumPy
Scikit-learn
Matplotlib
Seaborn
Jupyter Notebook

🤝 Contributing

Want to contribute improvements or explore alternative modeling techniques? Fork the repo and submit a pull request — all contributions are welcome!

🪪 License
This project is available under the MIT License.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Platform](https://img.shields.io/badge/Platform-JupyterLab%20%7C%20Notebook-lightgrey.svg)
![Status](https://img.shields.io/badge/Status-Exploratory-blueviolet.svg)
![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)

