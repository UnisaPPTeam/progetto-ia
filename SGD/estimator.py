# does the estimation stuff
from sklearn.linear_model import SGDRegressor
import numpy as np

class Estimator():
    # init the estimator, we create one model for each possible action, the model we picked is an SGDRegressor, models will be the list of regressors
    def __init__(self, action_count, initial_state):
        self.models = []
        for i in range(action_count):
            # Create the regressor
            model = SGDRegressor(learning_rate="constant")
            array = np.asarray(initial_state)
            tables, nx, ny = array.shape
            d2_train_dataset = array.reshape((1, tables * nx * ny))
            model.partial_fit(d2_train_dataset, [0])
            self.models.append(model)
    
    # Given a state return the q values for the actions
    def predict(self, state, a = None):
        if not a:
            # Cycle between all models, encoding all the actions, getting the expected reward in this state where one to take said action, add all of this data into one np array
            array = np.asarray(state)
            tables, nx, ny = array.shape
            d2_train_dataset = array.reshape((tables * nx * ny))
            return np.array([m.predict([d2_train_dataset])[0] for m in self.models])
        else:
            # If we recieved an action as input, either because we entered the above loop, or because this function is being used as a state-action function, return the prediction of the model that encodes said action
            return self.models[a].predict([state])[0]
        
    # Update the estimator to better fit the TD target y for the state-action pair    
    def update(self, state, action, target):
        # Cycle between all models, encoding all the actions, getting the expected reward in this state where one to take said action, add all of this data into one np array
        array = np.asarray(state)
        tables, nx, ny = array.shape
        d2_train_dataset = array.reshape((1, tables * nx * ny))
        self.models[action].partial_fit(d2_train_dataset, [target])