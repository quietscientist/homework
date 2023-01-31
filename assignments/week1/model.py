import numpy as np

class LinearRegression:
    """
    A linear regression class with two functions: fit, predict. considers the analytical solution, without using machine learning libraries. 
    """   
    w: np.ndarray
    b: float


    def __init__(self):
        """
        Set class attributes w (weight), b (bias) initialized to 0. 
        """

        self.w = 0
        self.b = 0

    def fit(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Takes as input X (input data with dimensions m,n) and y (target output). Returns analytical solution to normal equation
        
        Arguments:
            X (np.ndarray): The input data
            y (np.ndarray): The target 

        Returns:
            params(np.ndarray): solution to the model (if possible)
        """    
        params = []

        m = X.shape[0] #number of training samples
        X = np.hstack((X, np.ones((m,1)))) # adding in 1's for bias
        y = y.reshape(m,1) # reshaping y to (m,1)

        """
        This section I borrowed directly from W2D1 workbook
        """
        if np.linalg.det(X.T@X) != 0:
            params = (np.linalg.inv(X.T@X)@X.T@y)
        else:
            print("LinAlgError. Matrix is Singular. No analytical solution.")
        # The Normal Equation
    
        return params

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Takes as input X and returns the vectorized linear regression model  
        
        Arguments:
            X (np.ndarray): The input data

        Returns:
            params(np.ndarray): model predictions
        """          
        w = self.w
        b = self.b

        pred = w.T@X + b
        return pred

class GradientDescentLinearRegression(LinearRegression):
    """
    A linear regression class that approximates the solution using gradient descent.
    """
    def __init__(self):
        """
        Set class attributes w (weight), b (bias) initialized to 0. Probably not necessary here since I re-initialize w in the fit function to avoid having the gradients always be 0. 
        """
 
        self.w = 0
        self.b = 0

    def fit(
        self, X: np.ndarray, y: np.ndarray, lr: float = 0.01, epochs: int = 1000
    ) -> None:
        
        """
        Finds the approximate solution by adjusting the slope of the gradient with each iteration based on the difference between y(target output) and y_hat (model prediction)

        Arguments:
            X (np.ndarray): The input data
            y (np.ndarray): The target 
            lr (float): The learning rate
            epochs (int): The number of training iterations

        Returns:
            None
        """   
        m = X.shape[0] # Number of training examples. 
        X = np.hstack((X, np.ones((m,1))))   
        
        y = y.reshape(m,1) # reshaping y to (m,1)       
        n = X.shape[1] # n features

        self.w = np.random.normal(0, 1, (n, 1)) #initialize weights scaled according to number of features
        b = self.b

        for i in range(epochs):
            """
            full disclosure I was having trouble writing this correctly so I asked ChatGPT to help me format (I was getting a dimension mismatch for y_hat and y).
            The solution was to use X instead of X.T when computing y_hat, but I still don't fully understand why. 
            """

            y_hat = X@self.w + b 
            dw = (2/m)*X.T@(y_hat - y) 
            db = (2/m)*np.sum(y_hat - y)
            
            self.w -= lr * dw #update parameters based on the error, and scaled by the learning rate. 
            self.b -= lr * db #update parameters based on the error, and scaled by the learning rate. 

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts the output for the given input (code partially derived from W2D1 workbook)

        Arguments:
            X (np.ndarray): The input data

        Returns:
            np.ndarray: The predicted output

        """

        m = X.shape[0] # Number of training examples. 
        X = np.hstack((X, np.ones((m,1))))   


        pred = X@self.w + self.b
        return pred

