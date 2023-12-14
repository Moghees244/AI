import pandas as pd

class Naive_Bayes:
    def __init__(self):
        self.data = None
        self.classes = {}

        self.prior_probabilities = {}
        self.probabilities = {}

    def read_data(self):
        # reading file
        self.data = pd.read_excel("NaiveBayes_Data.xlsx")
        print("Data : \n", self.data, "\n")

    def train_model(self):
        self.calculate_prior_probabilities()

        for attribute in self.data:
            if attribute == 'Class':continue
            # get all possible values of attribute
            attribute_types = self.get_attribute_types(self.data[attribute])
            # calculate probability of each type of attribute for class
            for clas in self.classes:
                for type in attribute_types:
                    prob = self.calculate_probability(clas, type, attribute)
                    self.probabilities[clas].update({type : prob})      # Store in a 2d Dictionary

        self.display_probabilities()

    
    def predict(self):
        inputs = self.get_input()
        probs = {}
        # Predict the class
        for clas in self.classes:
            class_probs = self.probabilities[clas]
            prob = self.prior_probabilities[clas]
            for i in range(self.data.shape[1] - 1):
                prob *= class_probs[inputs[i]]

            probs.update({clas: prob})
        
        print("\nPrediction : ", probs)
        print(f"\nThe Predicted Answer is {result(probs)} \n")

    # calculate probability of attribute type
    def calculate_probability(self, clas, type, attribute):
        count = 0
        for i in range(self.data.shape[0]):
            if self.data['Class'][i] == clas and self.data[attribute][i] == type:
                count += 1
        return round(count/self.classes[clas] , 2)

    # prior probability of classes          
    def calculate_prior_probabilities(self):
        # Getting all classes
        self.classes = self.get_attribute_types(self.data['Class'])

        for clas in self.classes:
            prob = round(self.classes[clas] / self.data.shape[0], 2)        # prior probability of class
            self.prior_probabilities.update({clas: prob})
            # just initializing attributes probability dictionary
            self.probabilities.update({clas: {}})

        print("Prior Probabilities : ", self.prior_probabilities, "\n\n")

    # get all the possible values of an attribute
    def get_attribute_types(self, attribute):
        unique_value = {}
        for atr in attribute:
            if atr not in unique_value:
                unique_value.update({atr: 1})
            else:
                unique_value[atr] += 1
        return unique_value
    
    # get input from user
    def get_input(self):
        inputs = []
        value = None

        for attribute in self.data:
            if attribute == 'Class': continue
            valid_values = self.get_attribute_types(self.data[attribute]).keys()

            while value not in valid_values:
                value = input(f"\nEnter {attribute} {valid_values}: ")
            inputs.append(value)
        
        return inputs
    
    def display_probabilities(self):
        print("\nProbabilities of Each Attribute Type \n")
        for clas in self.classes:
            print(f"Class '{clas}' : {self.probabilities[clas]}")
        print()


def result(probs):
    values = list(probs.values())
    max_index = values.index(max(values))
    keys = list(probs.keys())
    return keys[max_index]
    

if __name__ == "__main__":
    nb = Naive_Bayes()
    nb.read_data()
    nb.train_model()
    nb.predict()
