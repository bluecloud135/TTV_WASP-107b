import pandas as pd

class Parameter():
    def __init__(self, name, value, reference):
        self.name = name
        self.value = value
        self.reference = reference
    def values(self):
        return self.name, self.value, self.reference
    
class Planet():
    def __init__(self, name,parameters):
        self.name = name
        self.parameters = parameters
    
    def getData(self):
        data = []
        for par in self.parameters:
            data.append(par.values())
        return data
    
    def printParameters(self, path=None):
        data = pd.DataFrame(self.getData(),columns=['Paramter', 'Value', 'Source'])
        display(data)
        if path!=None:
            data.to_csv(path+self.name+'_parameters.csv',index=False)
            data.to_latex(path+self.name+'_parameters.tex',index=False)