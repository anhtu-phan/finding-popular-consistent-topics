# finding-popular-consistent-topics

## How to run
### Requirements

    pip install -r requirements.txt

### Run finding similar item sets

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

### Run finding popular topics

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