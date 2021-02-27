def api(r,p):  
    Item1 = p
    import pandas as pd
    dataset2 = pd.DataFrame(r)
    import re
    import nltk
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    from nltk.corpus import stopwords
    from nltk.stem.porter import PorterStemmer
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import word_tokenize
    import contractions
    corpus2 = []
    for i in dataset2['Reviews']:
        review = re.sub('[^a-zA-Z]', ' ', i) 
        review = review.lower()
        review = contractions.fix(review)
        review = review.split()
        lemmatizer = WordNetLemmatizer()
        ps = PorterStemmer() 
        all_stopwords = stopwords.words('english')
        all_stopwords.remove('not')
        review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
        review = [lemmatizer.lemmatize(word) for word in review if not word in set(all_stopwords)]
        review = ' '.join(review)
        corpus2.append(review)

    import joblib
    filename = 'finalized_model.sav'
    loaded_model = joblib.load(open(filename, 'rb'))

    Phones = ['iphone 7 plus','iphone 6','huawei y9']
    Laptops = ['hp i7','hp i5','dell i5']

    Feature1 = []
    Feature2 = []
    Feature3 = []

    if(Item1 in Phones):
        device = Phones
        feature1 = 'display'
        feature2 = 'batteri'
        feature3 = 'speaker'
        featureOne = 'battery'
        featureTwo = 'desplay'
        featureThree = 'speakers'
    else:
        device = Laptops
        feature1 = 'display'
        feature2 = 'batteri'
        feature3 = 'charger'
        featureOne = 'battery'
        featureTwo = 'desplay'
        featureThree = 'charger'

    import re
    for comment in corpus2:
        x = re.search(feature1, comment)
        y = re.search(feature2, comment)
        z = re.search(feature3, comment)
        if x:
            Feature1.append(comment)
        if y:
            Feature2.append(comment)
        if z:
            Feature3.append(comment)
 
    if(Feature1 and Feature2 and Feature3):
        from sklearn.feature_extraction.text import CountVectorizer
        cv2 = CountVectorizer(max_features = 1500)

        Overall = cv2.fit_transform(corpus2).toarray()
        Feature1 = cv2.fit_transform(Feature1).toarray()
        Feature2 = cv2.fit_transform(Feature2).toarray()
        Feature3 = cv2.fit_transform(Feature3).toarray()
        PredictOverall = loaded_model.predict(Overall)
        PredictFeature1 = loaded_model.predict(Feature1)
        PredictFeature2 = loaded_model.predict(Feature2)
        PredictFeature3 = loaded_model.predict(Feature3)
       
        def p(x):
            pos_list = []
            pos_count = 0
            for i in x:
                if (i == 'pos'):
                    pos_list.append(i)
                    pos_count += 1 
            return pos_count
        def n(y):
            neg_list = []
            neg_count = 0
            for i in y:
                if (i == 'neg'):
                    neg_list.append(i)
                    neg_count += 1 
            return neg_count
            
        pos_count_Overall = p(PredictOverall)
        neg_count_Overall = n(PredictOverall) 
        pos_count_Feature1 = p(PredictFeature1)
        neg_count_Feature1 = n(PredictFeature1)
        pos_count_Feature2 = p(PredictFeature2)
        neg_count_Feature2 = n(PredictFeature2)
        pos_count_Feature3 = p(PredictFeature3)
        neg_count_Feature3 = n(PredictFeature3) 
        
        total_count_Overall = pos_count_Overall + neg_count_Overall
        pos_ratio_Overall = int((pos_count_Overall / total_count_Overall) * 100)
        neg_ratio_Overall = int((neg_count_Overall / total_count_Overall) * 100)

        total_count1 = pos_count_Feature1 + neg_count_Feature1
        pos_ratio1 = int((pos_count_Feature1 / total_count1) * 100)
        neg_ratio1 = int((neg_count_Feature1 / total_count1) * 100)

        total_count2 = pos_count_Feature2 + neg_count_Feature2
        pos_ratio2 = int((pos_count_Feature2 / total_count2) * 100)
        neg_ratio2 = int((neg_count_Feature2 / total_count2) * 100)

        total_count3 = pos_count_Feature3 + neg_count_Feature3
        pos_ratio3 = int((pos_count_Feature3 / total_count3) * 100)
        neg_ratio3 = int((neg_count_Feature3 / total_count3) * 100)

        print(pos_ratio_Overall ,neg_ratio_Overall)
        print(pos_ratio1 ,neg_ratio1)
        print(pos_ratio2 ,neg_ratio2)
        print(pos_ratio3 ,neg_ratio3)

    
        import pymongo
        client = pymongo.MongoClient("mongodb+srv://RH:1234@cluster0.7zbx6.mongodb.net/test?retryWrites=true&w=majority")
        #client = pymongo.MongoClient("mongodb+srv://user01:XGZoO4znMtBJJ2mY@cluster0.d2ddd.mongodb.net/itemsDB?retryWrites=true&w=majority")
        db = client['test']
        collection = db["test"]
        
        import datetime
        post = {
            "item": Item1,
            "positive": pos_ratio_Overall,
            "negative": neg_ratio_Overall,
            "positiveCount": pos_count_Overall,
            "negativeCount": neg_count_Overall,
            "featuresCount": 3,
            "totalCount": total_count_Overall,
            "features":{
            "featureOne":{
                "name":featureOne,
                "positive":pos_ratio1,
                "Nagative":neg_ratio1,
            },
            "featureTwo":{
                "name":featureTwo,
                "positive":pos_ratio2,
                "negative":neg_ratio2,
            },
            "featureThree":{
                "name":featureThree,
                "positive":pos_ratio3,
                "negative":neg_ratio3,
            }
            }
            
        }
        #"date": datetime.datetime.utcnow()
        A = collection.insert_one(post).inserted_id
    else:
        print("Features are not found")
        


            

                