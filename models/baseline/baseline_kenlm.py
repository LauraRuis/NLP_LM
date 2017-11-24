import kenlm

model = kenlm.Model('4gram.arpa')

test_data = ""
test_file = open("../../data/penn/test.txt.UNK", "r")

for word in test_file.read().split():
    test_data += word + " "

print(model.perplexity(test_data))

