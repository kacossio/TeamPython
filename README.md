# TeamPython
This repository contains our model, which includes a model to pull Twitter data using the API, and a model for classifying Twitter's users
location from information such as profile images, banner images, bio information, profile name, and recent user tweets. 

# Documentation
## Normalization stage:
```python 
def import_clean_data(dataset_name):
  data=pd.read_csv(dataset_name)
  # clean dataset
  # remove URLs
  no_url_data = data.tweet
  no_url_data = no_url_data.str.replace('http\S+|www.\S+', '', case=False)
  # remove emoji 
  no_emoji_data = no_url_data.str.replace('[^\w\s#@/:%.,_-]', '', flags=re.UNICODE)
  #remove punctuation characters except hashtag
  no_punc_data = no_emoji_data.str.replace('[^\w\s#]', '', flags=re.UNICODE)
  no_punc_data.head()
  # delete previous tweet column
  del data["tweet"]
  # add a cleaned tweets column into dataset
  data['tweets']=no_punc_data
  return(data)

def normalize_data(data):
  #Account info normalization
  # log1p normalization
  numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
  for c in [c for c in data.columns if data[c].dtype in numerics]:
      data[c] = np.log1p(data[c])

  # min-max normalization
  column_names_to_normalize = ['favourites_count','followers_count','statuses_count','friends_count','listed_count']
  x = data[column_names_to_normalize].values
  min_max_scaler = preprocessing.MinMaxScaler()
  x_scaled = min_max_scaler.fit_transform(x)
  df_temp = pd.DataFrame(x_scaled, columns=column_names_to_normalize, index = data.index)
  df_temp["screen_name"] = data['screen_name']
  return(df_temp)

```

## Tweet Embedding Stage:

Get tweets from dataframe
```python
def get_tweets_list(df):
  all_tweets = []
  s_name = []
  for num in range(len(df)):
    all_tweets.append(df.iloc[num]['tweets'])
    s_name.append(df.iloc[num]['screen_name'])
  print('tweets extracted')
  return all_tweets,s_name
```
Input list of tweets with structure as [[tweet, screen_name],[tweet, screen_name],[tweet, screen_name],...]
```python
def get_bert_embeddings(tweet_list):
  model_path = "/content/drive/My Drive/dataset/multilingual_model"
  embeddings = keras_bert.extract_embeddings(model_path, tweet_list)
  print('embeddings complete')
  return(embeddings)
```
Mean pool the the embeddings to return 768 embeddings per sentence
```python
def avg_pooling(embed_array):
  embeddings_pooled = []
  for sentence in embed_array:
    sentence = np.expand_dims((sentence),axis = 0)
    sentence = tf.keras.layers.GlobalAveragePooling1D()(sentence)
    embeddings_pooled.append(np.squeeze(sentence))
  return(embeddings_pooled)
```
## Image Embedding Stage:
Reading images from URLs
```python
def read_images(dataframe):
    counter =list(range(0,len(dataframe), int(len(dataframe)/3)))
    working_sname = []
    for i in range(3): 
        count = counter[1]
        pop = counter.pop(0)
        print(f"Range for this iteration is {pop}, {count}")
        profile = []
        banner = []
        success = True
        start = time.time()
        for num in tqdm(range(pop,count)):
            image_url = dataframe.iloc[num]["image_url"]
            banner_url = dataframe.iloc[num]["banner_url"]
            try:
                profile.append(np.asarray(Image.open(requests.get(image_url, stream=True).raw).convert('RGB')))
                print(getsizeof(profile))
                success = True
            except:
                success = False
                pass
            if success:
                try:
                    banner.append(np.asarray(Image.open(requests.get(banner_url, stream=True).raw).convert('RGB')))
                    print(getsizeof(banner))
                    working_sname.append(dataframe.iloc[num]["screen_name"])
                except:
                    del profile[-1]
                    success = False
                    pass
        np.save("profile"+str(i), profile)
        np.save("banner"+str(i), banner)

    print(f"{len(working_sname)} images have been read")
    return(working_sname,banner,profile)
```
Preparing the embedding model
```
def get_image_model():
    #define new feature embedding model
    source_model = EfficientNetB0(weights='imagenet')
    test_model = source_model.layers[-3].output
    predictions = keras.layers.Dense(1280)(test_model)
    image_embedding_model = keras.Model(inputs = source_model.input, outputs = predictions)
    return(image_embedding_model)
```
Extract image embeddings
```python 
def image_embeddings_keras(image_list,model):
    image_size = model.input_shape[1]
    cursor = 0
    embed_list = []
    for i in tqdm(range(len(image_list))):
        image =np.asarray(image_list[i])
        x = center_crop_and_resize(image, image_size=image_size)
        x = preprocess_input(x)
        embed_list.append(x)
    print("preprocessing complete")
    #complete all embeddings 
    start = time.time()
    final_array = model.predict(np.array(embed_list[0:]))
    print(f"embedding extracted in {time.time()-start}")
    return(final_array)
```
## Fully connected model:
Function for model flow
```
def model_flow(model_name, num_countries, input_shape):
    inputs = keras.Input(shape=(input_shape), name="Combined_inputs")
    x = layers.Dense(64, activation="relu", name="dense_1")(inputs)
    x = layers.BatchNormalization(name="normalization_1")(x)
    x = layers.Dense(32, activation="relu",name="dense_2")(x)
    x = layers.Dense(16, activation=tf.keras.layers.LeakyReLU(alpha=0.2), name="dense_3")(x)
    x = layers.Dropout(0.15)(x)
    outputs = layers.Dense(num_countries, activation="softmax",name="predictions")(x)
    model = keras.Model(inputs=inputs, outputs=outputs, name=model_name)
    return model
```
