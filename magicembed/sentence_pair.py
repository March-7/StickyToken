import os
import re
import itertools
from tqdm import tqdm
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch
import pandas as pd
import matplotlib.pyplot as plt
from datasets import load_dataset
from magicembed.utils import distance_metrics


def output_dataset_name(model_id, tag, extension):
    model_id_alphanum = re.sub(r"[^a-zA-Z0-9]", "_", model_id)
    filename = f"G:\juchiyun2024-11-14/ckx_ws/MagicEmbed/data/{model_id_alphanum}/{tag}.{extension}"
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
                # 尝试加载数据集的"en"子集，如果存在的话
                dataset = load_dataset(path=path, name='en', split=split)
            except:
                try:
                    dataset = load_dataset(path=path, name='en-en', split=split)
                except:
                    try:
                        # 如果没有"en"子集，尝试加载整个数据集的指定split
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
        从多个dataset中获取sentence pair：先将所有数据集的两列组成一列，再采样，再两两组合计算相似度，结果保存到csv文件中
        '''
        # 定义输出文件路径
        self.sentence_pairs_path = output_dataset_name(model_name, 'sentence_pairs', 'csv')
        if os.path.exists(self.sentence_pairs_path):
            print(f"文件 {self.sentence_pairs_path} 已经存在。")
            # 读取已存在的文件
            existing_df = pd.read_csv(self.sentence_pairs_path)
            # 打印文件中的信息
            print(f"sentence_pairs文件中包含 {len(existing_df)} 个句子对。")
            unique_sentences = set(existing_df['sentence1']).union(set(existing_df['sentence2']))
            print(f"sentence_pairs文件中包含 {len(unique_sentences)} 个不重复的句子。")
            # 计算并打印相似度范围
            min_similarity_existing = existing_df['similarity'].min()
            max_similarity_existing = existing_df['similarity'].max()
            print(f"sentence_pairs文件中的相似度范围：最小相似度 {min_similarity_existing}，最大相似度 {max_similarity_existing}。")
            return

        # 将所有数据集的'sentence1'和'sentence2'中的所有句子组合成一个列表
        all_sentences = []
        for dataset in self.datasets:
            all_sentences.extend(dataset['sentence1'])
            all_sentences.extend(dataset['sentence2'])

        # 对all_sentences去重
        all_sentences = list(set(all_sentences))
        print(f"总共有 {len(all_sentences)} 个非重复句子")
        # 从去重后的句子中随机采样num个句子
        np.random.seed(42)
        if len(all_sentences) > num:
            sampled_sentences = np.random.choice(all_sentences, num, replace=False)
        else:
            sampled_sentences = all_sentences

        print(f"总共采样了 {len(sampled_sentences)} 个句子")
        print("其中前5个句子示例:")
        for i, 句子 in enumerate(sampled_sentences[:5]):
            print(f"{i+1}. {句子}")

        # 计算所有可能的句子对组合
        sentence_pairs = list(itertools.combinations(sampled_sentences, 2))

        print(f"总共有 {len(sentence_pairs)} 个非重复句子对组合")

        # 初始化结果列表
        results_batch = []

        embedding_model.eval()
        # 使用tqdm显示进度（使用batch）
        for i in tqdm(range(0, len(sentence_pairs), batch_size), desc="分batch计算句子对相似度",miniters=10):
            # 获取当前批次的句子对
            batch_pairs = sentence_pairs[i:i + batch_size]
            
            # 分别提取批次中的句子
            batch_sentence1 = [pair[0] for pair in batch_pairs]
            batch_sentence2 = [pair[1] for pair in batch_pairs]

            # 计算批次中所有句子的嵌入向量
            embeddings1 = embedding_model.encode(batch_sentence1)
            embeddings2 = embedding_model.encode(batch_sentence2)
            
            # 计算批次中所有句子的余弦相似度
            similarities = cosine_similarity(embeddings1, embeddings2)

            # 将结果添加到列表中
            for j, (sentence1, sentence2) in enumerate(batch_pairs):
                if sentence1 != sentence2:
                    results_batch.append({
                        'sentence1': sentence1,
                        'sentence2': sentence2,
                        'similarity': similarities[j][j]
                    })

        # 将结果转换为DataFrame
        result_df = pd.DataFrame(results_batch)

        # 对结果进行去重
        result_df = result_df.drop_duplicates(subset=['sentence1', 'sentence2'])

        # 按相似度升序排序
        result_df = result_df.sort_values('similarity', ascending=True)

        # 将结果保存到CSV文件
        result_df.to_csv(self.sentence_pairs_path, index=False)

        print(f"\n所有结果已按照余弦相似度升序排列，保存到 {self.sentence_pairs_path} 文件中。")
        print(f"文件数据量大小（句子对数量）：{len(result_df)}") # 460k
        
    def get_sampled_sentence_pairs(self,
                                   model_name:str,
                                   sample_size:int=1000,
                                   ):
        '''
        从文件中读取sentence pair的相似度数据,并且按照similarity的范围，均匀分割进行采样,最后去重保存到csv文件中
        '''
        # 定义输出文件路径
        self.sampled_sentence_pairs_path = output_dataset_name(model_name, 'sampled_sentence_pairs', 'csv')
        if os.path.exists(self.sampled_sentence_pairs_path):
            print(f"文件 {self.sampled_sentence_pairs_path} 已经存在。")
            # 读取已存在的文件
            existing_df = pd.read_csv(self.sampled_sentence_pairs_path)
            # 打印文件中的信息
            print(f"sampled_sentence_pairs文件中包含 {len(existing_df)} 个句子对。")
            unique_sentences = set(existing_df['sentence1']).union(set(existing_df['sentence2']))
            print(f"sampled_sentence_pairs文件中包含 {len(unique_sentences)} 个不重复的句子。")
            # 计算并打印相似度范围
            min_similarity_existing = existing_df['similarity'].min()
            max_similarity_existing = existing_df['similarity'].max()
            print(f"sampled_sentence_pairs文件中的相似度范围：最小相似度 {min_similarity_existing}，最大相似度 {max_similarity_existing}。")
            return

        sentence_pairs_df = pd.read_csv(self.sentence_pairs_path)
        # 计算相似度范围
        min_similarity = sentence_pairs_df['similarity'].min()
        max_similarity = sentence_pairs_df['similarity'].max()

        # 计算相似度步长
        similarity_step = (max_similarity - min_similarity) / (sample_size - 1)

        # 进行近似均匀采样
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

        # print('去重前',len(sampled_df))
        # 去除可能的重复行并重置索引
        # 仅使用非NumPy数组列进行去重
        # columns_to_check = [col for col in sampled_df.columns if not isinstance(sampled_df[col].iloc[0], np.ndarray)]
        # sampled_df = sampled_df.drop_duplicates(subset=['sentence1', 'sentence2']).reset_index(drop=True)

        # 升序排序
        sampled_df = sampled_df.sort_values('similarity', ascending=True)

        # 统计总共使用了多少个不重复的单个句子
        unique_sentences = set(sampled_df['sentence1']).union(set(sampled_df['sentence2']))
        print(f"总共使用了 {len(unique_sentences)} 个不重复的句子。")

        # 保存采样结果
        sampled_df.to_csv(self.sampled_sentence_pairs_path, index=False)
        print(f"\n所有采样结果已保存到 {self.sampled_sentence_pairs_path} 文件中。")
        print(f"总共采样了 {len(sampled_df)} 个不重复的句子对，在最小相似度 {min_similarity} 和最大相似度 {max_similarity} 之间均匀采样。")



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

        # 首先得到moda.vocab_embeddings_mean_cosine_similarity，以该值作为分界，从小于该分界的句子对中，均匀采样num个句子对，
        # 具体做法为从similarity最小值到mean_similarity之一区间的前80%的句子对中，按照similarity均匀设置步长，采样num个句子对
        import json
        json_file = os.path.join('G:\juchiyun2024-11-14/ckx_ws/MagicEmbed/magicembed', 'model_record.json')

        mean_similarity = None
        try:
            mean_similarity = moda.vocab_embeddings_mean_cosine_similarity
        except AttributeError:
            print('moda对象没有vocab_embeddings_mean_cosine_similarity属性，尝试从JSON文件中获取。')

        if mean_similarity is None:
            try:
                print('moda对象vocab_embeddings_mean_cosine_similarity属性为None，尝试从JSON文件中获取。')
                with open(json_file, 'r') as f:
                    records = json.load(f)
                
                # 假设records是一个列表，包含多个字典，每个字典有模型名称和vocab_embeddings_mean_cosine_similarity
                model_name = moda.model_name  # 假设moda有一个属性model_name
                mean_similarity = next((np.float32(record["vocab_embeddings_mean_cosine_similarity"]) for record in records if record['model_name'] == model_name), None)

                if mean_similarity is None:
                    print(f'在JSON文件中未找到模型 {model_name} 的vocab_embeddings_mean_cosine_similarity。')
            except FileNotFoundError:
                print('JSON文件未找到，请检查路径。')
            except json.JSONDecodeError:
                print('JSON文件格式错误。')
            except Exception as e:
                print('发生未知错误，请确保moda.check_is_anisotropic()已运行以生成vocab_embeddings_mean_cosine_similarity。', e)

        filtered_dataset = self.dataset.filter(lambda x: x['similarity'] < mean_similarity)
        
        # 计算相似度范围
        min_similarity = min(filtered_dataset['similarity'])
        max_similarity = min_similarity + (mean_similarity - min_similarity) * 0.8
        print('采样相似度范围：[',min_similarity,',', max_similarity,']')
        print('mean_similarity:',mean_similarity)
        
        # 判断是否有足够的句子对
        if len(filtered_dataset) < num:
            print(f"只有 {len(filtered_dataset)} 个句子对，无法采样 {num} 个。建议采样{len(filtered_dataset)/2}。可以适当增加SentencePair.get_sampled_sentence_pairs()的sample_size参数 > sample_size * {num/len(filtered_dataset)}")
            num = len(filtered_dataset)

        # 计算相似度步长
        similarity_step = (max_similarity - min_similarity) / (num - 1)

        # 进行近似均匀采样
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
            print('部分gt_texts:', self.gt_texts[:10])
            print('部分contract_texts:', self.contract_texts[:10])
            print('gt_embs.shape:', self.gt_embs.shape)
            print('contract_embs.shape:', self.contract_embs.shape)
            print("部分gt_metrics:")
            print("cosine_distance:", self.gt_metrics.cosine_distance[:10])
            print("euclidean_distance:", self.gt_metrics.euclidean_distance[:10])
            print("manhattan_distance:", self.gt_metrics.manhattan_distance[:10])
        else:
            print('完整gt_texts:', self.gt_texts)
            print('完整contract_texts:', self.contract_texts)
            print('gt_embs.shape:', self.gt_embs.shape)
            print('contract_embs.shape:', self.contract_embs.shape)
            print("完整gt_metrics:")
            print(self.gt_metrics)
