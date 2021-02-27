a = {'one':1, 'two':2, 'three':3}
b = {'one':1, 'two':2}

def printdict(two,one,three=None):
    print(one,two,three)
    print(three,two,one)

printdict(**a)
printdict(**b)
