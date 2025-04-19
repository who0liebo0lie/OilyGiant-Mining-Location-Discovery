🛢️ OilyGiant Mining Location Discovery

This notebook helps identify the most profitable region for oil extraction for OilyGiant. It uses machine learning to predict oil production across three regions and evaluates economic viability based on predicted yields and investment risks.

📚 Table of Contents
About the Project

Installation
Usage
Project Structure
Technologies Used
Results & Insights
Screenshots
Contributing
License

📌 About the Project
OilyGiant needs to choose one of three potential regions to open a new oil well. This notebook analyzes the historical oil production data from each region using regression modeling, risk analysis, and expected profitability calculations to guide decision-making.

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

📊 Results & Insights
Cleaning of each dataframe had to be performed independently.  Once completed tasks could be grouped together.  Each dataframe needed the "id" column dropped before running through the model.  
All other tasks were able to be placed in a loop where each dataframe was looped through. 

In order to break-even 22,222.22 barrels of oil need to be generated at a site.  Average volume required per well to break-even is 111.11 barrells. 

Site 1 has the lowest risk of loss among all the evaluated sites.  Site 1 has the largest average profit at 5 million dollars.  
Site 1 is also the only site where the lower quantile is a positive number.  It is notebwrothy that Site 1 has the lowest predicted 
average volume of reserves by almost 30 barrels.  However Site 1 remains the best business decision because its extreme low risk of failure.  
The expected 24 million dollars generation from this site could fund exploration of sites besides site 0 and site 2 for Oily Giant.   

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

