class FTC():
    def __init__(self):
        
        import os

        try:
            from nltk.stem.lancaster import LancasterStemmer
            import nltk
        except:
            os.system("pip3 install nltk")
            from nltk.stem.lancaster import LancasterStemmer
            import nltk

        try:
            from tensorflow.python.framework import ops
            import tensorflow
        except:
            os.system("pip3 install tensorflow")
            from tensorflow.python.framework import ops
            import tensorflow

        try:
            import tflearn
        except:
            os.system("pip3 install tflearn")
            import tflearn

        try:
            import pickle
        except:
            os.system("pip3 install pickle")
            import pickle

        try:
            import numpy
        except:
            os.system("pip3 install numpy")
            import numpy

        self.RunningPath = os.getcwd()

        self.stemmer = LancasterStemmer()

        self.data = {"intents":[]}

    def CreateGroup(self, Thetag, Thepatterns):

        self.data["intents"].append({
                            "tag": Thetag, 
                            "patterns": Thepatterns, 
                        })
        
    def ShowClassification(self):
        print("{\"intents\": [")
        for i in range(len(self.data["intents"])):
            print()
            print("\t{"+"\"tag\": {The_tag},\n\t\t\"patterns\": {The_pat}".format(The_tag = self.data["intents"][i]["tag"], The_pat = self.data["intents"][i]["patterns"]))
        print("\n\t]")
        print("\n}")

    def StartUp(self, TrainModel=False, epoch=1000, batch=8):

        self.ReFlashModel = False
        self.TrainModel = TrainModel
        
        try:
            if self.TrainModel == False:
                with open(self.RunningPath+r"\data.pickle", "rb") as f:
                    self.words, self.labels, self.training, self.output = pickle.load(f)
                self.checkList = []
                for self.checkingIntent in self.data["intents"]:
                    if self.checkingIntent["tag"] not in self.checkList:
                        self.checkList.append(self.checkingIntent["tag"])
                if len(self.checkList) != len(self.labels):
                    self.ReFlashModel = True
                    raise BaseException()
            else:
                raise BaseException()
        except BaseException:

            self.words = []
            self.labels = []
            self.docs_x = []
            self.docs_y = []

            for self.intent in self.data["intents"]:
                for self.pattern in self.intent["patterns"]:
                    self.wrds = nltk.word_tokenize(self.pattern)
                    self.words.extend(self.wrds)
                    self.docs_x.append(self.wrds)
                    self.docs_y.append(self.intent["tag"])

                if self.intent["tag"] not in self.labels:
                    self.labels.append(self.intent["tag"])

            self.words = sorted(list(set([self.stemmer.stem(w.lower()) for w in self.words if w != "?"])))

            self.labels = sorted(self.labels)

            self.training = []
            self.output = []

            self.out_empty = [0 for _ in range(len(self.labels))]

            for self.x, self.doc in enumerate(self.docs_x):
                self.bag = []

                self.wrds = [self.stemmer.stem(w.lower()) for w in self.doc]

                for w in self.words:
                    if w in self.wrds:
                        self.bag.append(1)
                    else:
                        self.bag.append(0)

                self.output_row = self.out_empty[:]
                self.output_row[self.labels.index(self.docs_y[self.x])] = 1

                self.training.append(self.bag)
                self.output.append(self.output_row)


            self.training = numpy.array(self.training)
            self.output = numpy.array(self.output)

            with open(self.RunningPath+r"\data.pickle", "wb") as f:
                pickle.dump((self.words, self.labels, self.training, self.output), f)

        if self.ReFlashModel == True:
            shutil.rmtree(os.getcwd()+r"\FTCmodels", ignore_errors=True)

        ops.reset_default_graph()

        self.model = tflearn.DNN(
                    tflearn.regression(
                        tflearn.fully_connected(
                            tflearn.fully_connected(
                                tflearn.fully_connected(
                                    tflearn.input_data(shape=[None, len(self.training[0])]), 
                                8), 
                            8), 
                        len(self.output[0]), activation="softmax"
                        )
                    )
                )

        self.epoch = epoch
        self.batch = batch
        
        if os.path.exists(self.RunningPath+r"\FTCmodels") and self.TrainModel == False:
            self.model.load(self.RunningPath+r"\FTCmodels\model.tflearn")
        else:
            shutil.rmtree(os.getcwd()+r"\FTCmodels", ignore_errors=True)
            os.mkdir(self.RunningPath+r"\FTCmodels")
            self.model.fit(self.training, self.output, n_epoch=self.epoch, batch_size=self.batch, show_metric=True)
            self.model.save(self.RunningPath + r"\FTCmodels\model.tflearn")

    def bag_of_words(self, s, words):

        self.words = words
        self.s = s
        
        self.bag = [0 for _ in range(len(self.words))]
        self.s_words = [self.stemmer.stem(word.lower()) for word in nltk.word_tokenize(self.s)]

        for se in self.s_words:
            for t, w in enumerate(self.words):
                if w == se:
                    self.bag[t] = 1
                
        return numpy.array(self.bag)


    def TestingAccuracy(self, AccuracyFilter=0):
        if type(AccuracyFilter) == float or type(AccuracyFilter) == int:
            if AccuracyFilter <= 1 and AccuracyFilter >=0:

                self.AccuracyFilter = AccuracyFilter
                
                print("Type QUIT to quit testing.")
                
                while True:
                    inp = input("Input: ")
                    if inp.lower() == "quit":
                        break

                    self.results = self.model.predict([self.bag_of_words(inp, self.words)])
                    self.results_index = numpy.argmax(self.results)
                    self.tag = self.labels[self.results_index]

                    if self.results[0][self.results_index] > self.AccuracyFilter:
                        print(self.tag)
                    else:
                        print("--Prediction Accuracy lower then {}, (In GetResponse function will return <None>".format(self.AccuracyFilter))
            else:
                raise BaseException("Prediction-Accuracy-Filter needs to be in between 0~1")
        else:
            raise BaseException("Prediction-Accuracy-Filter needs to be type <int>")

    def Predict(self, text, AccuracyFilter=0):
        if type(AccuracyFilter) == float or type(AccuracyFilter) == int:
            if AccuracyFilter <= 1 and AccuracyFilter >=0:

                self.AccuracyFilter = AccuracyFilter

                self.results = self.model.predict([self.bag_of_words(text, self.words)])
                self.results_index = numpy.argmax(self.results)
                self.tag = self.labels[self.results_index]

                if self.results[0][self.results_index] > self.AccuracyFilter:
                    return self.tag
                else:
                    return None
            else:
                raise BaseException("Prediction-Accuracy-Filter needs to be in between 0~1")
        else:
            raise BaseException("Prediction-Accuracy-Filter needs to be type <int>")