def mapping(y):
    a = y[0]
    for i in range(len(y)):
        if y[i]==a:
            y[i] = 0
        else:
            y[i] = 1
    return y

