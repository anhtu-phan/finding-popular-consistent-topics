# finding-popular-consistent-topics
Covid19 is a global epidemic that affects many aspects of society and is discussed a lot on social network such as Facebook, Twitter. What topics become popular together? And these are topics that are discussed frequently or only appear in a certain period of time? The purpose of this project is finding popular consistent
topics in each time period from the [COVID19 Tweets dataset](https://www.kaggle.com/gpreda/covid19-tweets). This project will implement two approaches to finding popular consistent topics: 
 - The first approach is trying some finding frequent items set algorithms to find words appear together in many tweets in a time period. 
 - The second approach is apply some topic modeling methods to find latent topic in each tweets and then find topics become together in a time period by applying some finding frequent items set algorithms.

## How to run
### Requirements
    python3.6

    pip install -r requirements.txt

### Run finding similar item sets (first approach)

    python finding_similar_items.py [--pre_process_type remove_twitter_account] [--alg fpg] [--min_sup 0.01] [--min_conf 0.01]

- `pre_process_type`: method when applying pre-process tweet content
  + `remove_url` 
  + `remove_twitter_account`
  + `remove_url_replace_twitter_account`
  + `remove_twitter_account_replace_url`
  + `replace_twitter_account_and_url`
- `alg`: finding similar item set algorithm
  + `fpg`: Frequent Pattern Growth algorithm 
  + `pcy`: PCY algorithm
  + `apriori`: A-Priori algrithm
- `min_sup`, `min_conf`: support and confidence parameter of fiding similar item set algorithm
- The result in `output` folder

### Run finding popular topics (second approach)

    python finding_popular_topic.py [--pre_process_type remove_twitter_account] [--alg_similar fpg] [--alg_topic LDA] [--num_topic 20]

- `pre_process_type`: same as finding similar item sets 
- `alg_similar`: finding similar topics algorithms
  + `fpg`: Frequent Pattern Growth algorithm 
  + `pcy`: PCY algorithm
  + `apriori`: A-Priori algrithm
- `alg_topic`: algorithm for finding latent topics from each tweets
  + `LDA`: Latent Dirichlet Allocation
  + `BTM`: Biterm topic model
- `num_topic`: number of topic
- The results in `output` folder
