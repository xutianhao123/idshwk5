from sklearn.ensemble import RandomForestClassifier
import math

domainList = []
clf = RandomForestClassifier(random_state=0)

def num_digit_in_string(s):
    num = 0
    for i in s:
        if i.isdigit():
            num = num + 1
    return num


def get_entropy(s):
    entropy = 0
    times = {}
    for i in s:
        times[i] = s.count(i)
        entropy -= 1 * (times[i] / len(s)) * math.log(times[i] / len(s))
    return entropy

class Domain:

    def __init__(self, _name, _label):
        self.name = _name
        self.label = _label
        self.length = len(_name)
        self.digit_num = num_digit_in_string(_name)
        self.entropy = get_entropy(_name)

    def return_feature(self):
        return [self.length, self.digit_num, self.entropy]

    def return_label(self):
        if self.label == 'notdga':
            return 0
        else:
            return 1
        
def train_from_data(filename):
    global clf, domainList

    with open(filename) as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or line == '':
                continue
            tokens = line.split(',')
            name = tokens[0]
            label = tokens[1]
            domainList.append(Domain(name, label))

    featureMatrix = []
    labelList = []
    for i in domainList:
        featureMatrix.append(i.return_feature())
        labelList.append(i.return_label())
    clf.fit(featureMatrix, labelList)


def predict_data(test_file, result_file):
    global clf
    with open(test_file) as f_test, open(result_file, 'w') as f_result:
        lines = f_test.readlines()
        for line in lines:
            line = line.strip()
            if line.startswith('#') or line == '':
                continue
            temp = clf.predict([[len(line), num_digit_in_string(line), get_entropy(line)]])
            if temp == [0]:
                f_result.write(line + ',notdga\n')
            else:
                f_result.write(line + ',dga\n')




def main():
    global clf,domainList
    train_from_data('train.txt')
    predict_data('test.txt', 'result.txt')

if __name__ == '__main__':
    main()
