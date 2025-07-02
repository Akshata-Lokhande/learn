from akshatalearn.model import Model
from akshatalearn.data import MyDataset

def train():
    dataset = MyDataset("/Users/akshata/Documents/Githublearn/learn/corruptmnist_v1")
    model = Model()
    # add rest of your training code here

if __name__ == "__main__":
    train()
