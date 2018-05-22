ORIGIN = './test_data.txt'
MODIFIED = './modified_data.txt'
RESULT = './what_changed.txt'

with open(ORIGIN,'r') as fh:
    origin_corpus=[line.strip().split(' ') for line in fh]
with open(MODIFIED,'r') as fh:
    modified_corpus=[line.strip().split(' ') for line in fh]

assert(len(origin_corpus) == len(modified_corpus))

# origin 1 , 2
# modified 2, 3
# o - m = 1 removed
# m - 0 = 3 added

what_changed = []
for i in range(len(origin_corpus)):
    origin_para = set(origin_corpus[i])
    modified_para = set(modified_corpus[i])
    removed_words = sorted(list(origin_para - modified_para))
    nb_removed = len(removed_words)
    added_words = sorted(list(modified_para - origin_para))
    nb_added = len(added_words)

    line_changed = str(nb_removed + nb_added) + '='\
            +str(nb_removed) + '-' + str(nb_added) + '+ '
    line_changed += 'RM: ' + ' '.join(removed_words) + '         '
    line_changed += 'ADD: ' + ' '.join(added_words) + '\n'
    what_changed.append(line_changed)

with open(RESULT, 'w+') as fh:
    for line in what_changed:
        fh.write(line)