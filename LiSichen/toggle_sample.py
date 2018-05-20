from shutil import copyfile

def toggle(txt_file):
    origin_file = './' + txt_file
    exchange_file = './sample/' + txt_file + '.tmp'
    toggled_file = './sample/' + txt_file + '.toggle'
    copyfile(toggled_file, exchange_file)
    copyfile(origin_file, toggled_file)
    copyfile(exchange_file, origin_file)
    print('toggle ', origin_file, ' with ', toggled_file)


toggle('class-0.txt')
toggle('class-1.txt')
toggle('test_data.txt')