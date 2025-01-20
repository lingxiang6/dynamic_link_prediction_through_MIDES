import json
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.init as init
from sklearn.preprocessing import StandardScaler
import argparse

device = 'cuda:1'

parser = argparse.ArgumentParser(description='NHPE method')
parser.add_argument('--fitting_function', default='linear', help='[linear, gaussian, sigmoid]')
args = parser.parse_args()

with open('dynamic_link_prediction_through_MIDES/adjacency_matrix/edge_status.json', 'r') as f:
    edge_status = json.load(f)

dataset = pd.DataFrame.from_dict(edge_status, orient='index')
dataset = pd.DataFrame(dataset.iloc[:, :16].values, index=dataset.index)
dataset = pd.DataFrame(dataset[~(dataset == 0).all(axis=1)])

def gaussian_pdf_vectorized(lambd_t, u, sigma):
            return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-((lambd_t - u) ** 2) / (2 * sigma ** 2))

def poisson_counting_process(dataset):
    if args.fitting_function == 'linear':
        dataset = dataset.apply(lambda x: (x == 1).cumsum(), axis=1)
        scaler = StandardScaler()
        data = pd.DataFrame(scaler.fit_transform(dataset), columns=dataset.columns, index = dataset.index)
    elif args.fitting_function == 'sigmoid':
        data = dataset.apply(lambda x: (x == 1).cumsum(), axis=1)
        data = data.applymap(lambda x: 1 / (1 + np.exp(-x)))
    else:
        pro_dataset = dataset.apply(lambda x: (x == 1).cumsum(), axis=1)
        T = 15
        tau = 1.5
        u_t = np.arange(0, T) * tau / (T - 1)
        sigma = tau / (np.arange(1, T + 1))  
        df_array = pro_dataset.values
        result_array = np.zeros_like(df_array, dtype=float)
        for t in range(T):
            result_array[:, t] = gaussian_pdf_vectorized(df_array[:, t], u_t[t], sigma[t])
        data = pd.DataFrame(result_array, columns=pro_dataset.columns)
    return data



    
data = poisson_counting_process(dataset)



X = data.iloc[:, :15].values
y = data.iloc[:, 15].values



X_tensor = torch.tensor(X, dtype = torch.float32).to(device)
y_tensor = torch.tensor(y, dtype = torch.float32).to(device)


class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        if args.fitting_function == 'linear':
            self.linear = nn.Linear(15, 1)
            init.uniform_(self.linear.weight, a=0.0, b=1.0)
            init.uniform_(self.linear.bias, a=0.0, b=1.0)
        elif args.fitting_function == 'sigmoid':
            self.linear = nn.Linear(15, 1, bias=False)
            init.uniform_(self.linear.weight, a=0.0, b=1.0)
        else:
            self.linear = nn.Linear(15, 1, bias=False)
            init.uniform_(self.linear.weight, a=0.0, b=1.0)
              
        
        
    def forward(self, x):
        x = self.linear(x)
        return x


model = LinearRegressionModel().to(device)





criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001) 


epochs = 1000
for epoch in range(epochs):
 
    y_pred = model(X_tensor).squeeze()
    loss_predict = criterion(y_pred, y_tensor)
    optimizer.zero_grad() 
    loss_predict.backward()        
    optimizer.step()      
    
    if (epoch + 1) % 10 == 0:  
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss_predict.item():.4f}')



with torch.no_grad():
    y_test = model(X_tensor).cpu().numpy().flatten()  
    y_test = torch.sigmoid(torch.from_numpy(y_test))


data['predicted_value'] = y_test
data.reset_index(inplace=True)
lambda_16_predict = data










    
    