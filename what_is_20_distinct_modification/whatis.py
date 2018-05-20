from helper import strategy

instance = strategy()
instance.check_data('./origin.txt', './modified.txt')
print('if you can see this, this mean all good')