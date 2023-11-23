import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss


class EarlyStopping:
    def __init__(self, patience=7, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

class StockPricePredictor:
    def __init__(self, model, train_df, test_df, selected_features, sequence_length=10, batch_size=32, threshold=0.01):
        self.model = model
        self.train_df = train_df
        self.test_df = test_df
        self.selected_features = selected_features
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.threshold = threshold
        self.train_loader, self.test_loader = self.prepare_data()
    
    def create_labels(close_prices, sequence_length=50):
        labels = []
        for i in range(len(close_prices)):
            if i < sequence_length:
                # 对于序列开始的前sequence_length天，使用其之前的数据来计算平均值
                mean_previous = np.mean(close_prices[:i]) if i > 0 else close_prices[0]
            else:
                # 对于序列的其他部分，使用过去sequence_length天的数据来计算平均值
                mean_previous = np.mean(close_prices[i-sequence_length:i])

            if close_prices[i] > mean_previous:
                labels.append("Increase")
            else:
                labels.append("Decrease")
        return np.array(labels)



    def create_sequences(input_data, sequence_length):
        """
        Create sequences from input data starting from the second day.
        
        :param input_data: Scaled feature data.
        :param sequence_length: Length of the sequence.
        :return: Array of sequences.
        """
        sequences = []
        for i in range(1, len(input_data) - sequence_length + 1):
            seq = input_data[i:i + sequence_length]
            sequences.append(seq)
        return np.array(sequences)





    def prepare_data(self):
        # Encode labels
        label_encoder = LabelEncoder()
        train_detailed_labels = self.create_three_class_labels(self.train_df['$close'])
        test_detailed_labels = self.create_three_class_labels(self.test_df['$close'])
        train_labels_encoded = label_encoder.fit_transform(train_detailed_labels)
        test_labels_encoded = label_encoder.transform(test_detailed_labels)

        # Scale data
        scaler = StandardScaler()
        X_train = self.train_df[self.selected_features].values
        X_test = self.test_df[self.selected_features].values
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Create sequences
        X_train_sequences = self.create_sequences(X_train_scaled)
        X_test_sequences = self.create_sequences(X_test_scaled)

        # Adjust labels for sequences
        y_train_sequences = train_labels_encoded[self.sequence_length:]
        y_test_sequences = test_labels_encoded[self.sequence_length:]

        # Convert to PyTorch tensors
        X_train_tensor = torch.tensor(X_train_sequences, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train_sequences, dtype=torch.long)
        X_test_tensor = torch.tensor(X_test_sequences, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test_sequences, dtype=torch.long)

        # Create DataLoader instances
        train_data = TensorDataset(X_train_tensor, y_train_tensor)
        test_data = TensorDataset(X_test_tensor, y_test_tensor)
        train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=self.batch_size, shuffle=False)

        return train_loader, test_loader

    # You can include your train_model and evaluate_accuracy_and_loss methods here



def train_model(model, train_loader, val_loader, test_loader, criterion, optimizer, num_epochs):
    best_model_state = None
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    train_metrics = []
    val_metrics = []

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for sequences, labels in train_loader:
            if torch.cuda.is_available():
                sequences = sequences.cuda()
                labels = labels.cuda()
            optimizer.zero_grad()
            outputs = model(sequences)
            outputs = outputs.squeeze()
            loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # 评估在训练集上的表现
        # train_accuracy, train_precision, train_recall, train_f1, train_auc_roc = ...

        # 评估在验证集上的表现
        model.eval()
        val_loss = 0
        for sequences, labels in val_loader:
            if torch.cuda.is_available():
                sequences = sequences.cuda()
                labels = labels.cuda()
            outputs = model(sequences)
            outputs = outputs.squeeze()
            loss = criterion(outputs, labels.float())
            val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        # val_accuracy, val_precision, val_recall, val_f1, val_auc_roc = ...

        # print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')

        # 更新最佳模型（基于验证集损失）
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict()

    return best_model_state, train_losses, val_losses, train_metrics, val_metrics








def evaluate_accuracy_and_loss(model, data_loader, criterion):
    model.eval()  # Set model to evaluation mode
    total_loss = 0
    total = 0
    predicted_labels_list = []
    true_labels_list = []
    predicted_probs_list = []

    with torch.no_grad():  # No gradient calculation during evaluation
        for sequences, labels in data_loader:
            if torch.cuda.is_available():
                sequences = sequences.cuda()  # 将数据移到 GPU
                labels = labels.cuda()  # 将标签移到 GPU

            labels = labels.float()
            outputs = model(sequences)
            outputs = outputs.squeeze()
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            # For binary classification, get probabilities and predicted labels
            predicted_probs = torch.sigmoid(outputs)
            predicted_labels = (predicted_probs > 0.5).float()

            # Append predictions and labels for later metrics calculation
            predicted_probs_list.extend(predicted_probs.cpu().numpy())
            predicted_labels_list.extend(predicted_labels.cpu().numpy())
            true_labels_list.extend(labels.cpu().numpy())

            total += labels.size(0)

    # Calculate metrics
    avg_loss = total_loss / len(data_loader)
    accuracy = 100 * (np.array(predicted_labels_list) == np.array(true_labels_list)).mean()
    precision = precision_score(true_labels_list, predicted_labels_list)
    recall = recall_score(true_labels_list, predicted_labels_list)
    f1 = f1_score(true_labels_list, predicted_labels_list)
    auc_roc = roc_auc_score(true_labels_list, predicted_probs_list)

    return accuracy, avg_loss, precision, recall, f1, auc_roc





class StockRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout_rate=0.5):
        super(StockRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.batch_norm = nn.BatchNorm1d(hidden_size)  # Batch normalization layer
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_size, 1)  # Output layer for binary classification

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.rnn(x, h0)
        out = out[:, -1, :]
        out = self.batch_norm(out)  # Apply batch normalization
        out = self.dropout(out)
        out = self.fc(out)
        return out



class StockLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout_rate=0.5):
        super(StockLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.batch_norm = nn.BatchNorm1d(hidden_size)  # Batch normalization layer
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        out = self.batch_norm(out)  # Apply batch normalization
        out = self.dropout(out)
        out = self.fc(out)
        return out


class StockGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout_rate):
        super(StockGRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.batch_norm = nn.BatchNorm1d(hidden_size)  # Batch normalization layer
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.gru(x, h0)
        out = out[:, -1, :]
        out = self.batch_norm(out)  # Apply batch normalization
        out = self.dropout(out)
        out = self.fc(out)
        return out

