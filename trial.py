import pickle


with open('REPLAYMEM', 'rb') as f:
	rep = pickle.load(f)


print(rep)
