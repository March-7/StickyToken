import os
import re
import itertools
from tqdm import tqdm
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch
import pandas as pd
import matplotlib.pyplot as plt
from datasets import load_dataset,concatenate_datasets
from stickytoken.utils import distance_metrics


def output_dataset_name(model_id, tag, extension):
    model_id_alphanum = re.sub(r"[^a-zA-Z0-9]", "_", model_id)
    filename = f"/root/StickyToken/data/{model_id_alphanum}/{tag}.{extension}"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    return filename

class SentencePair:
    '''
    SentencePair is a class for sentence pair embedding.
    '''
    def __init__(self, 
                 dataset_paths: list[str] | str, 
                 model,
                 model_name:str,
                get_sentence_pairs_num:int=1000,
                 split:str='test',
                ):
        self.datasets = []
        if isinstance(dataset_paths, str):
            dataset_paths = [dataset_paths]
        
        for path in dataset_paths:
            try:
                if split == 'all':
                    # If split is 'all', load all subsets
                    dataset = load_dataset(path=path, name='en')
                    # Merge all subsets
                    all_splits = []
                    for split_name in dataset.keys():
                        all_splits.append(dataset[split_name])
                    dataset = concatenate_datasets(all_splits)
                else:
                    # Try to load the "en" subset of the dataset if it exists
                    dataset = load_dataset(path=path, name='en', split=split)
            except:
                try:
                    if split == 'all':
                        # If split is 'all', load all subsets
                        dataset = load_dataset(path=path, name='en-en')
                        # Merge all subsets
                        all_splits = []
                        for split_name in dataset.keys():
                            all_splits.append(dataset[split_name])
                        dataset = concatenate_datasets(all_splits)
                    else:
                        dataset = load_dataset(path=path, name='en-en', split=split)
                except:
                    try:
                        if split == 'all':
                            # If split is 'all', load all subsets
                            dataset = load_dataset(path=path)
                            # Merge all subsets
                            all_splits = []
                            for split_name in dataset.keys():
                                all_splits.append(dataset[split_name])
                            dataset = concatenate_datasets(all_splits)
                        else:
                            # If no "en" subset exists, try loading the specified split of the entire dataset
                            dataset = load_dataset(path=path, split=split)
                    except:
                        raise Exception(f'Please check the dataset path or subset name for {path}.')
            self.datasets.append(dataset)
        
        self.get_sentence_pairs(model, model_name,num = get_sentence_pairs_num)
        self.get_sampled_sentence_pairs(model_name)

    def get_sentence_pairs(self,
                          embedding_model,
                          model_name:str,
                          num:int=1000,
                          batch_size:int=128,
                          ):
        '''
        Get sentence pairs from multiple datasets: first combine two columns from all datasets into one column, 
        then sample, then calculate similarity between pairs, and save results to csv file
        '''
        # Define output file path
        self.sentence_pairs_path = output_dataset_name(model_name, 'sentence_pairs', 'csv')
        if os.path.exists(self.sentence_pairs_path):
            print(f"File {self.sentence_pairs_path} already exists.")
            # Read existing file
            existing_df = pd.read_csv(self.sentence_pairs_path)
            # Print file information
            print(f"sentence_pairs file contains {len(existing_df)} sentence pairs.")
            unique_sentences = set(existing_df['sentence1']).union(set(existing_df['sentence2']))
            print(f"sentence_pairs file contains {len(unique_sentences)} unique sentences.")
            # Calculate and print similarity range
            min_similarity_existing = existing_df['similarity'].min()
            max_similarity_existing = existing_df['similarity'].max()
            print(f"Similarity range in sentence_pairs file: min similarity {min_similarity_existing}, max similarity {max_similarity_existing}.")
            return

        # Combine all sentences from 'sentence1' and 'sentence2' columns of all datasets into a list
        all_sentences = []
        for dataset in self.datasets:
            all_sentences.extend(dataset['sentence1'])
            all_sentences.extend(dataset['sentence2'])

        # Remove duplicates from all_sentences
        all_sentences = list(set(all_sentences))
        print(f"Total {len(all_sentences)} unique sentences")
        # Randomly sample num sentences from deduplicated sentences
        np.random.seed(42)
        if len(all_sentences) > num:
            sampled_sentences = np.random.choice(all_sentences, num, replace=False)
        else:
            sampled_sentences = all_sentences

        print(f"Sampled total of {len(sampled_sentences)} sentences")
        print("First 5 example sentences:")
        for i, sentence in enumerate(sampled_sentences[:5]):
            print(f"{i+1}. {sentence}")

        # Calculate all possible sentence pair combinations
        sentence_pairs = list(itertools.combinations(sampled_sentences, 2))

        print(f"Total {len(sentence_pairs)} unique sentence pair combinations")

        # Initialize results list
        results_batch = []

        embedding_model.eval()
        # Use tqdm to show progress (using batches)
        for i in tqdm(range(0, len(sentence_pairs), batch_size), desc="Calculating sentence pair similarities in batches",miniters=10):
            # Get current batch of sentence pairs
            batch_pairs = sentence_pairs[i:i + batch_size]
            
            # Extract sentences from batch
            batch_sentence1 = [pair[0] for pair in batch_pairs]
            batch_sentence2 = [pair[1] for pair in batch_pairs]

            # Calculate embeddings for all sentences in batch
            embeddings1 = embedding_model.encode(batch_sentence1)
            embeddings2 = embedding_model.encode(batch_sentence2)
            
            # Calculate cosine similarities for all sentences in batch
            similarities = cosine_similarity(embeddings1, embeddings2)

            # Add results to list
            for j, (sentence1, sentence2) in enumerate(batch_pairs):
                if sentence1 != sentence2:
                    results_batch.append({
                        'sentence1': sentence1,
                        'sentence2': sentence2,
                        'similarity': similarities[j][j]
                    })

        # Convert results to DataFrame
        result_df = pd.DataFrame(results_batch)

        # Remove duplicates from results
        result_df = result_df.drop_duplicates(subset=['sentence1', 'sentence2'])

        # Sort by similarity in ascending order
        result_df = result_df.sort_values('similarity', ascending=True)

        # Save results to CSV file
        result_df.to_csv(self.sentence_pairs_path, index=False)

        print(f"\nAll results sorted by cosine similarity in ascending order, saved to {self.sentence_pairs_path}.")
        print(f"File data size (number of sentence pairs): {len(result_df)}") # 460k
        
    def get_sampled_sentence_pairs(self,
                                   model_name:str,
                                   sample_size:int=1000,
                                   ):
        '''
        Read sentence pair similarity data from file, sample uniformly across similarity ranges, 
        remove duplicates and save to csv file
        '''
        # Define output file path
        self.sampled_sentence_pairs_path = output_dataset_name(model_name, 'sampled_sentence_pairs', 'csv')
        if os.path.exists(self.sampled_sentence_pairs_path):
            print(f"File {self.sampled_sentence_pairs_path} already exists.")
            # Read existing file
            existing_df = pd.read_csv(self.sampled_sentence_pairs_path)
            # Print file information
            print(f"sampled_sentence_pairs file contains {len(existing_df)} sentence pairs.")
            unique_sentences = set(existing_df['sentence1']).union(set(existing_df['sentence2']))
            print(f"sampled_sentence_pairs file contains {len(unique_sentences)} unique sentences.")
            # Calculate and print similarity range
            min_similarity_existing = existing_df['similarity'].min()
            max_similarity_existing = existing_df['similarity'].max()
            print(f"Similarity range in sampled_sentence_pairs file: min similarity {min_similarity_existing}, max similarity {max_similarity_existing}.")
            return

        sentence_pairs_df = pd.read_csv(self.sentence_pairs_path)
        # Calculate similarity range
        min_similarity = sentence_pairs_df['similarity'].min()
        max_similarity = sentence_pairs_df['similarity'].max()

        # Calculate similarity step size
        similarity_step = (max_similarity - min_similarity) / (sample_size - 1)

        # Perform approximate uniform sampling
        sampled_df = pd.DataFrame()
        selected_pairs = set()
        selected_sentences = set()
        for i in range(sample_size):
            target_similarity = min_similarity + i * similarity_step
            sorted_indices = (sentence_pairs_df['similarity'] - target_similarity).abs().argsort()
            for idx in sorted_indices:
                candidate_row = sentence_pairs_df.iloc[[idx]]
                candidate_pair = (candidate_row['sentence1'].values[0], candidate_row['sentence2'].values[0])
                if candidate_pair not in selected_pairs:
                    sampled_df = pd.concat([sampled_df, candidate_row])
                    selected_pairs.add(candidate_pair)
                    break

        # Sort in ascending order
        sampled_df = sampled_df.sort_values('similarity', ascending=True)

        # Count total number of unique individual sentences used
        unique_sentences = set(sampled_df['sentence1']).union(set(sampled_df['sentence2']))
        print(f"Total {len(unique_sentences)} unique sentences used.")

        # Save sampling results
        sampled_df.to_csv(self.sampled_sentence_pairs_path, index=False)
        print(f"\nAll sampling results saved to {self.sampled_sentence_pairs_path}.")
        print(f"Sampled total of {len(sampled_df)} unique sentence pairs, uniformly sampled between min similarity {min_similarity} and max similarity {max_similarity}.")



class Dataset:
    '''
    Dataset is a class for dataset
    '''
    def __init__(self, 
                sentence_pairs_path:str,
                moda,
                num:int,
                ):
        self.dataset = load_dataset('csv', data_files=sentence_pairs_path, split='train')

        # First get moda.vocab_embeddings_mean_cosine_similarity, use this value as boundary,
        # uniformly sample num sentence pairs from pairs with similarity less than this boundary.
        # Specifically, sample num pairs from the first 80% of the range between min similarity and mean similarity,
        # using uniform step sizes based on similarity
        import json
        json_file = os.path.join('/root/StickyToken/magicembed', 'model_record.json')

        mean_similarity = None
        try:
            mean_similarity = moda.vocab_embeddings_mean_cosine_similarity
        except AttributeError:
            print('moda object has no vocab_embeddings_mean_cosine_similarity attribute, trying to get from JSON file.')

        if mean_similarity is None:
            try:
                print('moda object vocab_embeddings_mean_cosine_similarity attribute is None, trying to get from JSON file.')
                with open(json_file, 'r') as f:
                    records = json.load(f)
                
                # Assume records is a list of dictionaries with model name and vocab_embeddings_mean_cosine_similarity
                model_name = moda.model_name  # Assume moda has model_name attribute
                mean_similarity = next((np.float32(record["vocab_embeddings_mean_cosine_similarity"]) for record in records if record['model_name'] == model_name), None)

                if mean_similarity is None:
                    print(f'Could not find vocab_embeddings_mean_cosine_similarity for model {model_name} in JSON file.')
            except FileNotFoundError:
                print('JSON file not found, please check path.')
            except json.JSONDecodeError:
                print('JSON file format error.')
            except Exception as e:
                print('Unknown error occurred, please ensure moda.check_is_anisotropic() has been run to generate vocab_embeddings_mean_cosine_similarity.', e)

        filtered_dataset = self.dataset.filter(lambda x: x['similarity'] < mean_similarity)
        
        # Calculate similarity range
        min_similarity = min(filtered_dataset['similarity'])
        max_similarity = min_similarity + (mean_similarity - min_similarity) * 0.8
        print('Sampling similarity range: [',min_similarity,',', max_similarity,']')
        print('mean_similarity:',mean_similarity)
        
        # Check if enough sentence pairs available
        if len(filtered_dataset) < num:
            print(f"Only {len(filtered_dataset)} sentence pairs available, cannot sample {num} pairs. Suggest sampling {len(filtered_dataset)/2}. Consider increasing SentencePair.get_sampled_sentence_pairs() sample_size parameter > sample_size * {num/len(filtered_dataset)}")
            num = len(filtered_dataset)

        # Calculate similarity step size
        similarity_step = (max_similarity - min_similarity) / (num - 1)

        # Perform approximate uniform sampling
        sampled_indices = []
        for i in range(num):
            target_similarity = min_similarity + i * similarity_step
            closest_index = np.argsort(np.abs(filtered_dataset['similarity'] - target_similarity))[0]
            sampled_indices.append(closest_index)

        sampled_dataset = filtered_dataset.select(sampled_indices)
        self.gt_texts = sampled_dataset['sentence1']
        self.contract_texts = sampled_dataset['sentence2']

        self.gt_embs = moda.model.encode(self.gt_texts)
        self.contract_embs = moda.model.encode(self.contract_texts)
        self.gt_metrics = distance_metrics(self.gt_embs, self.contract_embs)

        if num > 10:
            print('Partial gt_texts:', self.gt_texts[:10])
            print('Partial contract_texts:', self.contract_texts[:10])
            print('gt_embs.shape:', self.gt_embs.shape)
            print('contract_embs.shape:', self.contract_embs.shape)
            print("Partial gt_metrics:")
            print("cosine_distance:", self.gt_metrics.cosine_distance[:10])
            print("euclidean_distance:", self.gt_metrics.euclidean_distance[:10])
            print("manhattan_distance:", self.gt_metrics.manhattan_distance[:10])
        else:
            print('Complete gt_texts:', self.gt_texts)
            print('Complete contract_texts:', self.contract_texts)
            print('gt_embs.shape:', self.gt_embs.shape)
            print('contract_embs.shape:', self.contract_embs.shape)
            print("Complete gt_metrics:")
            print(self.gt_metrics)
