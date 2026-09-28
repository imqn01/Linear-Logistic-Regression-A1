import numpy as np
import pandas as pd #spør  
class LinearRegression():
    def __init__(self, lr=0.0001, n_iterations=1000): 
        self.lr = lr
        self.n_iterations = n_iterations
        self.loss_history = []
        self.weights = None 
        self.bias = None
                    
    def compute_loss(self, y, y_pred):
        MSE = np.mean((y - y_pred)**2) 
        self.loss_history.append(MSE)
    
    def compute_gradients(self, X, y: pd.Series, y_pred: np.ndarray): 
        grad_w = np.dot(X.T, (y_pred-y)) / len(y)
        grad_b = np.mean(y_pred-y)
        return grad_w, grad_b
        
    def update_parameters(self, grad_w: np.ndarray , grad_b: float): 
        self.weights -= self.lr*grad_w  
        self.bias -= self.lr*grad_b
        return self.weights, self.bias
    
    def fit(self, X, y):
        """
        Estimates parameters for the classifier
        Args:
            X (array<m,n>): a matrix of floats with
                m rows (#samples) and n columns (#features)
            y (array<m>): a vector of floats
        """
        # ====================================
        # YOUR CODE GOES HERE
        if isinstance(X, pd.Series):
            X = X.to_frame()
        elif isinstance(X, np.ndarray) and X.ndim == 1:
            X=X.reshape(-1,1)
            
        _, cols = X.shape
        y = y.to_numpy(float)
        self.weights = np.ones(cols)
      
        self.bias = 0 
     
        
        for _ in range(self.n_iterations): #git lol
            y_pred = np.dot(X, self.weights) + self.bias
            grad_w, grad_b = self.compute_gradients( X, y, y_pred)
            self.update_parameters(grad_w, grad_b)
            self.compute_loss(y, y_pred)
    
        return self
        # ====================================
        raise NotImplementedError("LinearRegression.fit is not implemented yet.")
    
    def predict(self, X): 
        """
        Generates predictions
        Note: should be called after .fit()
        Args:
            X (array<m,n>): a matrix of floats with 
                m rows (#samples) and n columns (#features)
        Returns:
            A length m array of floats: 1000 ulike elementer
        """
        # ====================================
        # YOUR CODE GOES HERE
        if isinstance(X, pd.Series):
            X = X.to_frame()
        elif isinstance(X, np.ndarray) and X.ndim == 1:
            X=X.reshape(-1,1)
            
        y_pred= np.dot(X, self.weights) + self.bias        
        return y_pred
        # ====================================
        raise NotImplementedError("LinearRegression.predict is not implemented yet.")
    
    
    
class LogisticRegression():
    def __init__(self, lr=0.1, n_iterations=1000):
        self.weights = None
        self.bias = None
        
        self.lr = lr
        self.n_iterations = n_iterations
        
        self.loss_history, self.train_accuracies = [], []
        
        
    def compute_loss(self, y, y_pred):#TODO velg riktig loss function
        eps = 1e-15  # avoid log(0) 
        y_pred = np.clip(y_pred, eps, 1 - eps) 
        loss= -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
        #−yln σ(wx + b)−(1−y) ln (1−σ(wx + b)). fra forelesningen 
        self.loss_history.append(loss)
        
    def compute_gradients(self, X, y: pd.Series, y_pred: np.ndarray): #Kopiert fra LinearR
        grad_w = np.dot(X.T, (y_pred-y)) / len(y)
        grad_b = np.mean(y_pred-y)
        return grad_w, grad_b
        
    def update_parameters(self, grad_w: np.ndarray , grad_b: float): #Kopiert fra LinearR
        self.weights -= self.lr*grad_w  
        self.bias -= self.lr*grad_b
        return self.weights, self.bias
    
    def accuracy(self, y, pred_to_class):#plug and play
        acc = np.mean(y == pred_to_class)
        return acc
    
    def fit(self, X, y):#plug and play
        # ====================================
        # YOUR CODE GOES HERE
        if isinstance(X, pd.Series):
            X = X.to_frame()
        elif isinstance(X, np.ndarray) and X.ndim == 1:
            X=X.reshape(-1,1)
        
        
        self.weights = np.zeros(X.shape[1]) 
        self.bias = 0
        # Gradient Descent
        for _ in range(self.n_iterations):
            z = np.matmul(X, self.weights) + self.bias
            y_pred = self.sigmoid(z)
            grad_w, grad_b = self.compute_gradients(X, y, y_pred)
            self.update_parameters(grad_w, grad_b)
            self.compute_loss(y, y_pred)
            pred_to_class = [1 if _y > 0.5 else 0 for _y in y_pred]
            self.train_accuracies.append(self.accuracy(y, pred_to_class)) #TODO endre
        return self
        # ====================================
        raise NotImplementedError("LogisticRegression.fit is not implemented yet.")
    
    def predict_proba(self, X):
        # ====================================
        # YOUR CODE GOES HERE
        if isinstance(X, pd.Series):
            X = X.to_frame()
        elif isinstance(X, np.ndarray) and X.ndim == 1:
            X=X.reshape(-1,1)
            
        z = np.matmul(X, self.weights) + self.bias
        y_pred = self.sigmoid(z)
        return y_pred
        
        # ====================================
        raise NotImplementedError("LogisticRegression.predict_proba is not implemented yet.")
        
    def predict(self, X):#plug and play
        # ====================================
        # YOUR CODE GOES HERE
        y_pred = self.predict_proba(X)
        return [1 if _y > 0.5 else 0 for _y in y_pred]
        # ====================================
        raise NotImplementedError("LogisticRegression.predict is not implemented yet.")
    
    def sigmoid(self, z):
        # ====================================
        # YOUR CODE GOES HERE
        return 1 / (1+ np.exp(-z))
        # ====================================
        raise NotImplementedError("LogisticRegression.sigmoid is not implemented yet.")