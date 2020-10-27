# TeamPython
This repository contains our model, which includes a model to pull Tweeter data using the API, and a model for classifying Twitter's users
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

get tweets from dataframe
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
input list of tweets with structure as [[tweet, screen_name],[tweet, screen_name],[tweet, screen_name],...]
```python
def get_bert_embeddings(tweet_list):
  model_path = "/content/drive/My Drive/dataset/multilingual_model"
  embeddings = keras_bert.extract_embeddings(model_path, tweet_list)
  print('embeddings complete')
  return(embeddings)
```
mean pool the the embeddings to return 768 embeddings per sentence
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
  profile = []
  banner = []
  working_sname = []
  success = True
  start = time.time()
  for num in tqdm(range(2000)):
    image_url = dataframe.iloc[num]["image_url"]
    banner_url = dataframe.iloc[num]["banner_url"]
    try:
      profile.append((Image.open(requests.get(image_url, stream=True).raw).convert('RGB')))
      success = True
    except:
      success = False
      pass
    if success:
      try:
        banner.append((Image.open(requests.get(banner_url, stream=True).raw).convert('RGB')))
        working_sname.append(dataframe.iloc[num]["screen_name"])
      except:
        del profile[-1]
        success = False
        pass

  print(f"/n {len(working_sname)} images have been read")
  return(profile, banner,working_sname)
```
extract image embeddings
```python 
def image_embeddings(model_name, image_list):
  model_name = 'efficientnet-b6'
  image_size = EfficientNet.get_image_size(model_name)
  tfms = transforms.Compose([transforms.Resize(image_size), transforms.CenterCrop(image_size), 
                          transforms.ToTensor(),
                          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])
  model = EfficientNet.from_pretrained(model_name)
  embedding_list = []
  for image in tqdm(image_list):
    image = tfms(image).unsqueeze(0)
    with torch.no_grad():
        features = model.extract_features(image)
        features = model._avg_pooling(features)
        features = torch.squeeze(features)
        features = np.asarray(features)
        embedding_list.append(features)
  print("embeddings complete")
  return(embedding_list)
```
