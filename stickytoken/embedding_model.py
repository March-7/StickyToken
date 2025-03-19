import os
import random
import json

import torch

random.seed(42)
from time import time
from typing import Optional
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from stickytoken.utils import check_vectors_on_unit_sphere,check_embeddings_is_anisotropic,\
    calculate_neighbor_distances, load_verification_candidates, magic_token_test_metric_multi_token,plot_neighbor_distances_distribution,magic_token_test_metric, record_experiment_info\
    ,write_magic_tokens_within_threshold,magic_token_verification,\
    write_verification_results,update_experiment_record,write_vocab_token_magic_scores,save_vocab_token_magic_scores_all_results,\
    write_verification_candidates, load_vocab_token_magic_scores,load_vocab_verifications

DEFAULT_THRESHOLD_PERCENTILE = 2.0

def candidates_for_verification(token_infos, threshold_ratio=DEFAULT_THRESHOLD_PERCENTILE, threshold=None):
    if threshold is None:
        threshold = np.percentile(
            [tc.get("main_metric", float('-inf')) for tc in token_infos.values() if "main_metric" in tc or "metrics" in tc],
            100 - threshold_ratio,
        )
        print(
            f"Using threshold {threshold:.3f} as {threshold_ratio:.1f}% of tokens to verify with vocab size {len(token_infos)}."
        )
    candidates = sorted(
        [tc for tc in token_infos.values() if tc.get("main_metric", float('-inf')) >= threshold and ("main_metric" in tc or "metrics" in tc)],
        key=lambda tc: tc.get("main_metric", float('-inf')),
        reverse=True
    )
    return candidates, threshold

def classify_verification(token_info: dict) -> float:
        """Classify a token based on its verification results."""
        token_info["max_prob"] = np.max(token_info["verification"])
        # encodeable
        if token_info["max_prob"] > 0.877:      # 0.877 = 1-1/(3*e)
            token_info["magic"] = "strong_verified"
        elif token_info["max_prob"] > 0.816:    # 0.816 = 1-1/(2*e)
            token_info["magic"] = "weak_verified"
        elif token_info["max_prob"] < 0.632:    # 0.632 = 1-1/(e)
            token_info["magic"] = "strong_rejected"
        else:  # 0.5 <= max_prob <= 0.8
            token_info["magic"] = "weak_verified"
        return token_info["max_prob"]

class ModelAnalyzer:
    def __init__(self, 
                 model_name_or_path:str, 
                 trust_remote_code:bool = False,
                 use_flash_attn:bool = False  
                 ,**kwargs
                ):
        self.model_name =  os.path.basename(model_name_or_path)
        if self.model_name in ["gte-base-en-v1.5","gte-large-en-v1.5","nomic-embed-text-v1","nomic-embed-text-v1.5"]:
            trust_remote_code = True
        if use_flash_attn:
            self.model = SentenceTransformer(
                model_name_or_path=model_name_or_path,
                device='cuda',
                trust_remote_code=trust_remote_code,
                model_kwargs={
                    "torch_dtype": torch.bfloat16,  
                    "attn_implementation": "flash_attention_2"}
            )
        else:
            self.model = SentenceTransformer(
                model_name_or_path=model_name_or_path,
                trust_remote_code=trust_remote_code,
                device='cuda',
                
            )
        self.tokenizer = self.model.tokenizer
        self.transformer_model = self.model._first_module().auto_model
        if self.model_name in ["gte-Qwen2-1.5B-instruct", "gte-Qwen2-7B-instruct"]:
            self.model.max_seq_length = 8192
            self.tokenizer.padding_side = 'left'
        self.vocab_size = self.tokenizer.vocab_size
        # Get model parameter count
        self.num_parameters = sum(p.numel() for p in self.model.parameters()) 
        print(f"Model name: {self.model_name}")
        print(f"Number of parameters: {self.num_parameters}")
        print(f"Vocabulary size: {self.vocab_size}")
        try:
            self.wte = self.transformer_model.encoder.embed_tokens.weight
            self.wte = self.wte.detach().cpu().to(torch.float32).numpy()
            self.wte = self.wte[0:self.vocab_size]
        except:
            try:
                self.wte = self.transformer_model.embeddings.word_embeddings.weight.data
                self.wte = self.wte.detach().cpu().to(torch.float32).numpy()
                self.wte = self.wte[0:self.vocab_size]
            except:
                try:
                    #model._first_module().auto_model.embed_tokens.weight.detach().cpu().numpy()
                    self.wte = self.transformer_model.embed_tokens.weight.data
                    self.wte = self.wte.detach().cpu().to(torch.float32).numpy()
                    self.wte = self.wte[0:self.vocab_size]
                except:
                    print('Unable to get word embedding weights')
    
        self.vocab = [self.tokenizer.convert_ids_to_tokens(i) for i in range(self.vocab_size)]
        self.vocab_embeddings = self.model.encode(self.vocab,batch_size=512,show_progress_bar=True)

        # Initialize attributes that will be computed later
        self.vocab_embeddings_is_on_unit_sphere = None
        self.wte_is_on_unit_sphere = None
        self.vocab_embeddings_is_anisotropic = None
        self.wte_is_anisotropic = None 
        self.vocab_embeddings_mean_cosine_similarity = None 


    def check_on_unit_sphere(self) -> bool:
        '''
        Check if all token vectors on output side are on unit hypersphere,
        and check if all token weights in wte are on unit hypersphere
        '''
        self.vocab_embeddings_is_on_unit_sphere = check_vectors_on_unit_sphere(self.vocab_embeddings)
        print(f"All token vectors on output side are on unit hypersphere: {self.vocab_embeddings_is_on_unit_sphere}")
        print('------------------------------------------------------')
        self.wte_is_on_unit_sphere = check_vectors_on_unit_sphere(self.wte)  
        print(f"All token weights in wte are on unit hypersphere: {self.wte_is_on_unit_sphere}")

    def check_is_anisotropic(self,
                             plot:bool = False) -> bool:
        '''
        Check if all token vectors on output side are anisotropic,
        and check if all token weights in wte are anisotropic
        '''
        self.vocab_embeddings_is_anisotropic,self.vocab_embeddings_mean_cosine_similarity = check_embeddings_is_anisotropic(self.vocab_embeddings,
                                                                                                                            plot=plot)
        print(f"All token vectors on output side are anisotropic: {self.vocab_embeddings_is_anisotropic}")
        print('------------------------------------')
        self.wte_is_anisotropic,_ = check_embeddings_is_anisotropic(self.wte)
        print(f"All token weights in wte are anisotropic: {self.wte_is_anisotropic}")
        # Optimize saving vocab_embeddings_mean_cosine_similarity to json file
        json_file = os.path.join('/root/StickyToken/magicembed', 'model_record.json')
        update_experiment_record(self.model_name, self.vocab_embeddings_mean_cosine_similarity, json_file)

    def neighbor_distances_statistics(self, mode = 'nearest') -> None:
        '''
        mode: 'nearest' or 'mean'
        '''
        self.nearest_neighbor_distances = calculate_neighbor_distances(self.vocab_embeddings,mode=mode)
        print('Neighbor distance statistics for all token vectors on output side:')
        print(self.nearest_neighbor_distances)

        # self.mean_neighbor_distances = calculate_neighbor_distances(self.vocab_embeddings,mode='mean')
        plot_neighbor_distances_distribution(self.nearest_neighbor_distances,mode=mode)

        print('statistic:')
        for distance_type, values in self.nearest_neighbor_distances.items():
            mean = np.mean(values)
            std = np.std(values)
            median = np.median(values)
            min_value = np.min(values)
            max_value = np.max(values)
            print(f"{distance_type} distance:")
            print(f"  Mean: {mean:.4f}")
            print(f"  Standard deviation: {std:.4f}")
            print(f"  Median: {median:.4f}")
            print(f"  Min: {min_value:.4f}")
            print(f"  Max: {max_value:.4f}")
            if mode == 'nearest':
                threshold = median
                threshold_name = f"{distance_type}_threshold"
                # locals()[threshold_name] = threshold
                setattr(self, threshold_name, threshold)
                print(f"  Threshold {threshold_name} (mean): {threshold:.4f}")
            print()

    def magic_token_test(self,
                        token: str,
                        token_id: Optional[int],
                        dataset,
                        num :int,
                        ) -> None:
        '''
        token: str  
        token_id: int
        dataset: Dataset
        num: int
        '''
        if token_id is not None:
            # token = self.tokenizer.decode([token_id])
            token = self.tokenizer.convert_ids_to_tokens(token_id)
            print('token_id:',token_id)
            print('token:',token)
        self.magic_results, self.magic_score = magic_token_test_metric(token,
                                                                dataset.gt_texts, 
                                                                dataset.contract_texts,
                                                                dataset.gt_embs, 
                                                                dataset.contract_embs, 
                                                                dataset.gt_metrics,
                                                                num,
                                                                self.model)
        print('magic token test metric:')
        print(self.magic_results)
        print('------------------------------------')
        print('magic score:')
        print(self.magic_score)
    
    def caculate_vocab_token_magic_score(self,
                        token_infos,
                        dataset,
                        num ,
                        metric_ix: int = 0,
                        do_sample = False,
                        sample_num = 2000,
                        )-> dict:
        
        '''
        1. Calculate metrics (magic_score) for tokens starting with 'ok'
        2. Add metric_names to first row of token_infos, add metrics to all rows
        3. Add main_metric to all rows
        '''
        # Filter tokens starting with 'ok'
        ok_tokens = [tc for tc in token_infos.values() if tc["category"].startswith("OK")]
        self.ok_tokens_num = len(ok_tokens)
        self.ok_tokens_percent = self.ok_tokens_num/len(token_infos)*100
        self.exlusion_tokens_num = len(token_infos) - self.ok_tokens_num
        self.exlusion_tokens_percent = (len(token_infos)-self.ok_tokens_num)/len(token_infos)*100
        print(f"Found {self.ok_tokens_num} tokens starting with 'ok' in {len(token_infos)} total tokens.")
        print(f'OK_tokens_percent:{self.ok_tokens_percent}%')
        print(f"exlusion_tokens_percent:{self.exlusion_tokens_percent}%")
        print('Then caculate magic score for these tokens.')
        # Calculate magic_score
        self.vocab_token_all_results = []
        record_metric_names = False
        start = time()

        if do_sample:
            ok_tokens = random.sample(ok_tokens,sample_num)
        for token_info in tqdm(ok_tokens, desc='Processing vocab magic score(OK)', total=len(ok_tokens), miniters=10):
            token_id = token_info['i']
            token = token_info['decoded']
            results, score = magic_token_test_metric(token, 
                                                    dataset.gt_texts, 
                                                    dataset.contract_texts,
                                                    dataset.gt_embs, 
                                                    dataset.contract_embs, 
                                                    dataset.gt_metrics,
                                                    num,
                                                    self.model)
            self.vocab_token_all_results.append(results)
            token_infos[token_id]["metrics"] = [float(v) for v in score.values()]
            token_infos[token_id]["main_metric"] = token_infos[token_id]["metrics"][metric_ix]
            if not record_metric_names:
                token_infos[0]["metric_names"] = list(score.keys())
                token_infos[0]["main_metric_name"] = token_infos[0]["metric_names"][metric_ix]
                record_metric_names = True


        self.caculate_vocab_token_magic_score_time = time() - start
        print(
            f'Time Cost:{self.caculate_vocab_token_magic_score_time}'
            )
        save_vocab_token_magic_scores_all_results(self.vocab_token_all_results,self.model_name)
        write_vocab_token_magic_scores(token_infos,self.model_name,compress=False)
        return token_infos

    def caculate_vocab_token_magic_score_multi_token(self,
                        token_infos,
                        dataset,
                        num ,
                        metric_ix: int = 0,
                        do_sample = False,
                        sample_num = 2000,
                        batch_size = 64,
                        )-> dict:
        
        '''
        1. Calculate metrics (magic_score) for tokens starting with 'ok'
        2. Add metric_names to first row of token_infos, add metrics to all rows
        3. Add main_metric to all rows
        '''
        # Filter tokens starting with 'ok'
        ok_tokens = [tc for tc in token_infos.values() if tc["category"].startswith("OK")]
        self.ok_tokens_num = len(ok_tokens)
        self.ok_tokens_percent = self.ok_tokens_num/len(token_infos)*100
        self.exlusion_tokens_num = len(token_infos) - self.ok_tokens_num
        self.exlusion_tokens_percent = (len(token_infos)-self.ok_tokens_num)/len(token_infos)*100
        print(f"Found {self.ok_tokens_num} tokens starting with 'ok' in {len(token_infos)} total tokens.")
        print(f'OK_tokens_percent:{self.ok_tokens_percent}%')
        print(f"exlusion_tokens_percent:{self.exlusion_tokens_percent}%")
        print('Then caculate magic score for these tokens.')
        # Calculate magic_score
        self.vocab_token_all_results = []
        record_metric_names = False
        start = time()

        if do_sample:
            ok_tokens = random.sample(ok_tokens, sample_num)
        
        # Extract token ids and decoded values
        token_ids = [token_info['i'] for token_info in ok_tokens]
        tokens = [token_info['decoded'] for token_info in ok_tokens]

        # Process tokens in batches
        for i in tqdm(range(0, len(tokens), batch_size), desc="Processing vocab magic score(OK) multi-token", total=(len(tokens) + batch_size - 1) // batch_size):
            batch_tokens = tokens[i:i + batch_size]
            batch_token_ids = token_ids[i:i + batch_size]

            # Use magic_token_test_metric_multi_token to process multiple tokens
            results, scores = magic_token_test_metric_multi_token(batch_tokens, 
                                                                  dataset.gt_texts, 
                                                                  dataset.contract_texts,
                                                                  dataset.gt_embs, 
                                                                  dataset.contract_embs, 
                                                                  dataset.gt_metrics,
                                                                  num,
                                                                  self.model,
                                                                  )

            for token_id, token in zip(batch_token_ids, batch_tokens):
                self.vocab_token_all_results.append(results[token])
                token_infos[token_id]["metrics"] = [float(v) for v in scores[token].values()]
                token_infos[token_id]["main_metric"] = token_infos[token_id]["metrics"][metric_ix]
                if not record_metric_names:
                    token_infos[0]["metric_names"] = list(scores[token].keys())
                    token_infos[0]["main_metric_name"] = token_infos[0]["metric_names"][metric_ix]
                    record_metric_names = True

        self.caculate_vocab_token_magic_score_time = time() - start
        print(
            f'Time Cost:{self.caculate_vocab_token_magic_score_time}'
            )
        save_vocab_token_magic_scores_all_results(self.vocab_token_all_results,self.model_name)
        write_vocab_token_magic_scores(token_infos,self.model_name,compress=False)
        return token_infos

    def magic_token_verify(self,
                        token,
                        token_id,
                        verification_dataset,
                        num ,
                        ) -> None:
        # print('Run neighbor_distances_statistics() function before running this function')
        if token_id is not None:
            token = self.tokenizer.decode([token_id])
            print('token_id:',token_id)
            print('token:',token)
        self.verify_results,self.mean_metrics_change = magic_token_verification(token,
                                                                verification_dataset.gt_texts, 
                                                                verification_dataset.gt_embs, 
                                                                verification_dataset.contract_texts,
                                                                verification_dataset.contract_embs, 
                                                                verification_dataset.gt_metrics,
                                                                num,
                                                                self.model)
                                                                # self.cosine_threshold,
                                                                # self.euclidean_threshold,
                                                                # self.manhattan_threshold)
        print('magic token verification:')
        print(self.verify_results)
        print('------------------------------------')
        print('mean metrics change:')
        print(self.mean_metrics_change)

    def adaptive_threshold_verification(self,
                        verification_dataset,
                        num ,
                        ) -> None:
        record_changes = {
            'cosine':[],
            'euclidean':[],
            'manhattan':[],
        }
        if isinstance(self.cosine_top_percent_tokens, list):
            sampled_tokens = random.sample(self.cosine_top_percent_tokens, len(self.cosine_top_percent_tokens) // 2)
        else:
            raise TypeError("self.cosine_top_percent_tokens should be a list")
        
        # Use tqdm to show progress bar
        for token_id, _ in tqdm(sampled_tokens, desc="Processing Sampled Tokens"):
            token = self.tokenizer.convert_ids_to_tokens(token_id)
            _, mean_metrics_change, _ = magic_token_verification(token,
                                                                verification_dataset.gt_texts, 
                                                                verification_dataset.gt_embs, 
                                                                verification_dataset.contract_texts,
                                                                verification_dataset.contract_embs, 
                                                                verification_dataset.gt_metrics,
                                                                num,
                                                                self.model,
                                                                0,
                                                                0,
                                                                0)

            record_changes['cosine'].append(mean_metrics_change.cosine_distance)
            record_changes['euclidean'].append(mean_metrics_change.euclidean_distance)
            record_changes['manhattan'].append(mean_metrics_change.manhattan_distance)
        # Update thresholds by dividing by negative floor division of mean and multiplying by original threshold
        self.cosine_threshold = self.cosine_threshold/(-self.cosine_threshold//np.mean(record_changes['cosine']))
        self.euclidean_threshold = self.euclidean_threshold/(-self.euclidean_threshold//np.mean(record_changes['euclidean']))
        self.manhattan_threshold = self.manhattan_threshold/(-self.manhattan_threshold//np.mean(record_changes['manhattan']))
        print('adaptive_threshold:')
        print('cosine_threshold:',self.cosine_threshold)
        print('euclidean_threshold:',self.euclidean_threshold)
        print('manhattan_threshold:',self.manhattan_threshold)

        from collections import OrderedDict

        self.verification_results = OrderedDict()
        
        for token_id, _ in tqdm(self.cosine_top_percent_tokens, desc="Verifying Tokens"):
            token = self.tokenizer.convert_ids_to_tokens(token_id)
            verify_results, mean_metrics_change,verify_flag = magic_token_verification(token,
                                                                verification_dataset.gt_texts, 
                                                                verification_dataset.gt_embs, 
                                                                verification_dataset.contract_texts,
                                                                verification_dataset.contract_embs, 
                                                                verification_dataset.gt_metrics,
                                                                num,
                                                                self.model,
                                                                self.cosine_threshold,
                                                                self.euclidean_threshold,
                                                                self.manhattan_threshold)
            token_info = dict(i=token_id,
                            raw_vocab=self.tokenizer.convert_ids_to_tokens(token_id) ,
                            verify_flag = verify_flag,
            )
            self.verification_results[token_id] = token_info
        write_verification_results(self.verification_results,self.model_name,compress=False)

    def final_verification(self,
                        token_infos,
                        verification_dataset,
                        num ,
                        threshold_ratio: float = DEFAULT_THRESHOLD_PERCENTILE,  # % of tokens to verify
                        threshold: Optional[float] = None,
                        batch_size: int = 256,
                        reload: bool = False,
                        ) -> None:
        '''
        1. candidates_for_verification
        2. Verify tokens in candidates_for_verification
        3. Save verification results
        '''
        if reload:
            token_infos = load_vocab_verifications(self.model_name)
        candidates, self.candidates_for_verification_threshold = candidates_for_verification(
                token_infos, threshold_ratio=threshold_ratio, threshold=threshold
            )
        remaining_candidates = [tc for tc in candidates if "verification" not in tc]
        print(
            "Candidates for verification, wrote to",
            write_verification_candidates(candidates,self.model_name,compress=False),
        )
        main_metric_name = token_infos[0]["main_metric_name"]
        print(
            f"Verifying {len(remaining_candidates)} of total {len(candidates)} candidates above threshold {self.candidates_for_verification_threshold:.3f} of {main_metric_name} for model {self.model_name} with vocab size {len(token_infos)}."
        )
        self.candidates_for_verification_num = len(remaining_candidates)

        start = time()
        for ii, token_info in enumerate(remaining_candidates):
            token = token_info['decoded']

            verify_results, mean_metrics_count = magic_token_verification(
                token,
                verification_dataset.gt_texts,
                verification_dataset.gt_embs,
                verification_dataset.contract_texts,
                verification_dataset.contract_embs,
                verification_dataset.gt_metrics,
                num,
                self.model,
                batch_size=batch_size,
                
            )
            token_info["verification"] = mean_metrics_count

            max_prob = classify_verification(token_info)
            print(
                f"[{time()-start:.0f}s, {ii+1}/{len(remaining_candidates)}] {token_info['magic']} with max_prob = {max_prob:.2e} token {token_info['i']}: {self.tokenizer.convert_ids_to_tokens(token_info['i'])!r} verification info: {token_info['verification']}"
            )
            if ii % 10 == 0:
                write_verification_results(token_infos,self.model_name,compress=False)
        self.final_verification_time = time()-start
        print(
            "Finished verification, wrote results to",
            write_verification_results(token_infos,self.model_name,compress=False),
        )


    def record_all(self,
                   EXP,
                  ):
        try:
            record_experiment_info(self.model_name,
                                self.vocab_size,
                                self.num_parameters,
                                EXP,
                                json_file='experiment_information.json',
                                ok_tokens_num = self.ok_tokens_num,
                                ok_tokens_percent = self.ok_tokens_percent,
                                exlusion_tokens_num = self.exlusion_tokens_num,
                                    exlusion_tokens_percent = self.exlusion_tokens_percent,
                                caculate_vocab_token_magic_score_time = self.caculate_vocab_token_magic_score_time,
                                final_verification_time = self.final_verification_time,
                                vocab_embeddings_is_on_unit_sphere = self.vocab_embeddings_is_on_unit_sphere,
                                    wte_is_on_unit_sphere = self.wte_is_on_unit_sphere ,
                                    vocab_embeddings_is_anisotropic = self.vocab_embeddings_is_anisotropic,
                                    wte_is_anisotropic = self.wte_is_anisotropic,
                                    vocab_embeddings_mean_cosine_similarity =self.vocab_embeddings_mean_cosine_similarity,
                                    candidates_for_verification_percentile = DEFAULT_THRESHOLD_PERCENTILE,
                                    candidates_for_verification_threshold = self.candidates_for_verification_threshold,
                                    candidates_for_verification_num = self.candidates_for_verification_num,
                                    )
        except:
            record_experiment_info(self.model_name,
                                self.vocab_size,
                                self.num_parameters,
                                EXP,
                                json_file='experiment_information.json',
                                ok_tokens_num = 0,
                                ok_tokens_percent = 0,
                                exlusion_tokens_num = 0,
                                exlusion_tokens_percent = 0,
                                caculate_vocab_token_magic_score_time = 0,
                                final_verification_time = self.final_verification_time,
                                    vocab_embeddings_is_on_unit_sphere = self.vocab_embeddings_is_on_unit_sphere,
                                    wte_is_on_unit_sphere = self.wte_is_on_unit_sphere ,
                                    vocab_embeddings_is_anisotropic = self.vocab_embeddings_is_anisotropic,
                                    wte_is_anisotropic = self.wte_is_anisotropic,
                                    vocab_embeddings_mean_cosine_similarity =self.vocab_embeddings_mean_cosine_similarity,
                                    candidates_for_verification_percentile = DEFAULT_THRESHOLD_PERCENTILE,
                                    candidates_for_verification_threshold = self.candidates_for_verification_threshold,
                                    candidates_for_verification_num = self.candidates_for_verification_num,
                                    )

