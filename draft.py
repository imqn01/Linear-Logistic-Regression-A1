import numpy as np
import pandas as pd #spør  

class LinearRegression():
    def __init__(self, lr=0.01, n_iterations=100):
        
        #Finne y_pred -> loss funksjon -> derivere -> gradient decent 
        #df = pd.read_csv('regression.csv') #fjerne fordi modellen skal kunne passe til flere. 
        #self.X = df['Net_Activity'].tolist()#Nå er X features i Net_Activety kolonnen. 
        #self.y = df['Energy'].tolist() #Liste av Energy verdier, dette er target? 
        self.lr = lr
        self.n_iterations = n_iterations
        self.loss_history = []#lagre en verdi her per iterasjon, det kan være average loss
        self.weights = None 
        self.bias = None
        
    # def y_predictions(self,X):# trenger bare en
    #     #y_preds = np.matmul(self.weights, X.transpose()) + self.bias #Transpose => snur matrisen. kolonne til rad, rad til kolonne. 
    #     #y_preds = X*self.weights + self.bias
    #     X = np.asarray(X) #gjør 
    #     if X.ndim == 1:
    #         X = X.reshape(-1, 1)
    #     y_preds = np.matmul(X, self.weights) + self.bias #todo trenger denne 
    #     return y_preds
        
                    
    def compute_loss(self, y, y_pred):
        #MSE = np.square(np.subtract(y, y_pred)).mean()
        MSE = np.mean((y - y_pred)**2) 
        self.loss_history.append(MSE)
        #return MSE #trenger ikke å returnere. 
    
    def compute_gradients(self, X, y: pd.Series, y_pred: np.ndarray): #Ådne sa at her er feilen mest sansynlig. https://www.geeksforgeeks.org/machine-learning/gradient-descent-in-linear-regression/ Hvorfor er det feil? 
        #X = np.asarray(X)#Er y ogsp en liste liksom? 
        #y = np.asarray(y)

        #if X.ndim == 1:
        #    X = X.reshape(-1, 1)   
        
        #grad_w = np.mean((-2 *X *(y-y_pred))) 
        #grad_b = np.mean((-2 *(y-y_pred)))
        
        
        
        
        grad_w = np.dot(X.T, (y_pred-y)) / len(y)
        grad_b = np.mean(y_pred-y)
            
        #Tester noe over. Det under funket før
        #grad_w = np.mean((-2 *X *(y-y_pred))) #X er en matrise,Hvorfor er X en matrise og ikke en vektor? 
        #grad_b = np.mean((-2 *(y-y_pred))) #TODO alle steder hvor jeg ganger X med noe annet må endres  @ operatoren 
        return grad_w, grad_b
        
    def update_parameters(self, grad_w: np.ndarray , grad_b: float): #endre til at input er compute grad, ikke regn ut to ganger
        # = self.compute_gradients(self.X, self.y, self.y_predictions(self,self.X))
        self.weights -= self.lr*grad_w  #vektene liste, feature = kolonne
        #for i in range(len(self.weights)):
        #    self.weights[i] -= self.lr*grad_w 
        self.bias -= self.lr*grad_b
        return self.weights, self.bias
    
    
    def fit(self, X, y): #hva må jeg gjøre for å rette feilen
        """
        Estimates parameters for the classifier
        Args:
            X (array<m,n>): a matrix of floats with
                m rows (#samples) and n columns (#features)
            y (array<m>): a vector of floats
        """
        # ====================================
        if isinstance(X, pd.Series):
            X = X.to_frame()
        elif isinstance(X, np.ndarray) and X.ndim == 1:
            X=X.reshape(-1,1)
            
        _, cols = X.shape
        y = y.to_numpy(float)
        self.weights = np.ones(cols)
        
        # YOUR CODE GOES HERE
        #X = np.asarray(X)
        #y = np.asarray(y)
        

        #if X.ndim == 1:
        #    X = X.reshape(-1, 1)
        #m, n = X.shape
        #print(f"X har {m} rader og {n} kolonner")  # bekreft at n stemmer med antall features
        #self.weights = []
        #self.weights = np.ones(cols), må finne ut antall kolonner => _, cols = X.shape TODO #basert på lang X-en er! antall kolonner TODO
        #Bias er tall mens weights er liste
        self.bias = 0 
        #print("MATRISEN")
        #print(X)
        #print("VEKTEN")
        #print(self.weights)
        #print("her tester vi")
        
        for _ in range(self.n_iterations): #git lol
            #y_pred = self.predict(X)
            y_pred = np.dot(X, self.weights) + self.bias
                #dw, db = self.compute_gradients( X, y, self.predict(X))
            grad_w, grad_b = self.compute_gradients( X, y, y_pred)
            self.update_parameters(grad_w, grad_b)
            self.compute_loss(y, y_pred)
        #print("Y_pred")
        #print(y_pred)
        #print(type(y_pred))
        return self

            #GRADIENT DECENT, learning rate -> noe du sender inn i gradient decent
            
            #grad_w, grad_b = self.compute_gradients(X, y, y_pred)
            #self.update_parameters(grad_w, grad_b)
            
        # ====================================
        raise NotImplementedError("LinearRegression.fit is not implemented yet.")
    
    def predict(self, X): #funnet gode forslag, 
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
        #X = np.asarray(X)

        #if X.ndim == 1:
        #    X = X.reshape(-1, 1)
        if isinstance(X, pd.Series):
            X = X.to_frame()
        elif isinstance(X, np.ndarray) and X.ndim == 1:
            X=X.reshape(-1,1)
            
        y_pred= np.dot(X, self.weights) + self.bias    
            
        #Tester noe over
        #y_pred = np.matmul(X, self.weights) + self.bias #todo trenger denne 
        #dot = matmul? Nora bruker 
        # y_pred= np.dot(X, self.weights) + self.bias
        #dot kan gjøre vektor regning vanskligere i hodet. 
    
        return y_pred
        # ====================================
        raise NotImplementedError("LinearRegression.predict is not implemented yet.")
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
class LogisticRegression():
    def __init__(self, lr=0.001, n_iterations=1000):
        self.weights = None
        self.bias = None
        
        self.lr = lr
        self.n_iterations = n_iterations
        
        self.loss_history = []
    
    def fit(self, X, y):
        # ====================================
        # YOUR CODE GOES HERE
        # ====================================
        raise NotImplementedError("LogisticRegression.fit is not implemented yet.")
    
    def predict_proba(self, X):
        # ====================================
        # YOUR CODE GOES HERE
        # ====================================
        raise NotImplementedError("LogisticRegression.predict_proba is not implemented yet.")
        
    def predict(self, X):
        # ====================================
        # YOUR CODE GOES HERE
        # ====================================
        raise NotImplementedError("LogisticRegression.predict is not implemented yet.")
    
    def sigmoid(self, z):
        # ====================================
        # YOUR CODE GOES HERE
        # ====================================
        raise NotImplementedError("LogisticRegression.sigmoid is not implemented yet.")