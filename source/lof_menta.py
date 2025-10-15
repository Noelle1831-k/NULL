import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
import os
from sklearn.metrics import classification_report
import joblib

train_sample = 350

test_data_path = [
  "/root/autodl-tmp/project/mentamodel_results/mbpp_layer/layerAIE.npy",
  "/root/autodl-tmp/project/mentamodel_results/humaneval_layer/layerAIE.npy",
  "/root/autodl-tmp/project/mentamodel_results/pmc_5000_layer/layerAIE.npy",
  "/root/autodl-tmp/project/mentamodel_results/menta_layer/layerAIE.npy",
  "/root/autodl-tmp/project/mentamodel_results/fin_layer/layerAIE.npy",
  "/root/autodl-tmp/project/mentamodel_results/wikidata_layer/layerAIE.npy",
  "/root/autodl-tmp/project/mentamodel_results/sciq_layer/layerAIE.npy",
  "/root/autodl-tmp/project/mentamodel_results/mathematicalproblems_layer/layerAIE.npy",
]

save_path = '/root/autodl-tmp/project/menta_llmscan_result'
if not os.path.exists(save_path):
    os.makedirs(save_path)

def load_data_n_train ():
    data_combined = np.load("/root/autodl-tmp/project/mentamodel_results/menta_layer/layerAIE.npy")

    # Standardize data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_combined[:train_sample])

    #init isolation-forest
    lof = LocalOutlierFactor(n_neighbors=10, novelty=True, contamination=0.05)
    
    # Train
    lof.fit(data_scaled)

    return scaler, lof

def predict_n_plot (scaler, lof, test_data_path, save_path):
    # Load and predict on a different dataset
    test_data = np.load(test_data_path)

    parent_dir = os.path.dirname(test_data_path)
    test_data_name = os.path.basename(parent_dir)
    
    if 'menta' in test_data_name:
        test_data = test_data[train_sample:]

    test_data_scaled = scaler.transform(test_data)
    predictions = lof.predict(test_data_scaled)

    # Analyze the predictions
    num_normals = np.sum(predictions == 1)
    num_anomalies = np.sum(predictions == -1)
    total = len(predictions)
    percent_normals = (num_normals / total) * 100
    percent_anomalies = (num_anomalies / total) * 100
    
    return percent_anomalies

def main(test_paths, save_path):
    scaler, lof = load_data_n_train()

    results = []
    for test_data_path in test_paths:
        results.append(round(predict_n_plot(scaler, lof, test_data_path, save_path),2))
    print(results)
    with open(os.path.join(save_path, f"results.txt"), 'w', encoding='utf-8') as f:
        for item in results:
            f.write(str(item) + '\n')


main (test_data_path, save_path)