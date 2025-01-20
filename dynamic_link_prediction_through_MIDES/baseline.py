import json
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.init as init
from sklearn.preprocessing import StandardScaler
import argparse
from sklearn import metrics

device = 'cuda:1'

torch.manual_seed(42)


torch.cuda.manual_seed(42)


parser = argparse.ArgumentParser(description='baseline model')
parser.add_argument('--fitting_function', default='linear', help='[linear, gaussian]')
args = parser.parse_args()

with open('dynamic_link_prediction_through_MIDES/adjence_matrix/edge_status.json', 'r') as f:
    edge_status = json.load(f)

dataset = pd.DataFrame.from_dict(edge_status, orient='index')


def gaussian_pdf_vectorized(lambd_t, u, sigma):
            return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-((lambd_t - u) ** 2) / (2 * sigma ** 2))

def poisson_counting_process(dataset):
    if args.fitting_function == 'linear':
        data = dataset.apply(lambda x: (x == 1).cumsum(), axis=1)
        scaler = StandardScaler()
        data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns, index = data.index)
    else:
        pro_dataset = dataset.apply(lambda x: (x == 1).cumsum(), axis=1)
        T = 16
        eta = 1.5
        u_t = np.arange(0, T) * eta / (T - 1)
        sigma = eta / (np.arange(1, T + 1))  
        df_array = pro_dataset.values
        result_array = np.zeros_like(df_array, dtype=float)
        for t in range(T):
            result_array[:, t] = gaussian_pdf_vectorized(df_array[:, t], u_t[t], sigma[t])
        data = pd.DataFrame(result_array, columns=pro_dataset.columns)
    return data



    
data = poisson_counting_process(dataset)



X = data.iloc[:, :16].values
y = dataset.iloc[:, 16].values



X_tensor = torch.tensor(X, dtype = torch.float32).to(device)
y_tensor = torch.tensor(y, dtype = torch.float32).to(device)


class Baseline(nn.Module):
    def __init__(self):
        super(Baseline, self).__init__()
        if args.fitting_function == 'linear':
            self.linear = nn.Linear(16, 1)
            init.uniform_(self.linear.weight, a=0.0, b=1.0)
            init.uniform_(self.linear.bias, a=0.0, b=1.0)
        else:
            self.linear = nn.Linear(16, 1, bias=False)
            init.uniform_(self.linear.weight, a=0.0, b=1.0)
                     
    def forward(self, x):
        x = self.linear(x)
        return x


model = Baseline().to(device)



criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001) 


epochs = 1000
for epoch in range(epochs):
 
    y_train = model(X_tensor).squeeze()
    train_loss = criterion(y_train, y_tensor)
    predicted_classes = (y_train >= 0.57).float()
    correct_predictions = (predicted_classes == y_tensor).float()
    accuracy = correct_predictions.mean().item()  
    optimizer.zero_grad() 
    train_loss.backward()        
    optimizer.step()      
    
    if (epoch + 1) % 10 == 0:  
        print(f'Epoch: {epoch}, train loss: {train_loss:.4f}, train accuracy: {accuracy:.4f}')



with torch.no_grad():
    y_test = model(X_tensor).squeeze()
    test_loss = criterion(y_test, y_tensor)    
    predicted_classes = (y_test >= 0.57).float()
    TP = ((y_tensor == 1) & (predicted_classes == 1)).sum().float()
    FP = ((y_tensor == 0) & (predicted_classes == 1)).sum().float()
    precision = TP / (TP + FP) 
    FN = ((y_tensor == 1) & (predicted_classes == 0)).sum().float()
    recall = TP / (TP + FN) if (TP + FN) > 0 else torch.tensor(0.0)
    f1 = 2 * (precision * recall) / (precision + recall)  
    auc = metrics.roc_auc_score(y_tensor.cpu().numpy(), y_test.cpu().numpy())
    correct_predictions = (predicted_classes == y_tensor).float()
    accuracy = correct_predictions.mean().item() 
    print(f'test accuracy: {accuracy:.4f}, test loss: {test_loss:.4f}, test_precision: {precision:.4f}, test_recall:{recall:.4f}, test_f1:{f1:.4f}, test_auc:{auc:.4f}')













    
    