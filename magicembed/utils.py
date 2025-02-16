import gzip
import itertools
import json, jsonlines
import os
import re
import csv
from collections import Counter, namedtuple
from sklearn.metrics.pairwise import cosine_similarity,cosine_distances, euclidean_distances, manhattan_distances
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import List
import random
import numpy as np
from sklearn.decomposition import PCA


# --- results saving/loading utils ---
def update_experiment_record(model_name, mean_cosine_similarity, file_path):
    experiment_record = {
        "model_name": model_name,
        "vocab_embeddings_mean_cosine_similarity": float(mean_cosine_similarity),
    }
            
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
            data = []

    for record in data:
        if record['model_name'] == model_name:
            record['vocab_embeddings_mean_cosine_similarity'] = experiment_record['vocab_embeddings_mean_cosine_similarity']
            break
    else:
        data.append(experiment_record)

    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print(f"已更新模型数据文件: {file_path}")

def record_experiment_info(model_name,
                           vocab_size,
                           num_parameters,
                           EXP,
                           json_file='experiment_information.json',
                           **kwargs):

    from datetime import datetime
    import pytz

    # 获取当前北京时间
    beijing_tz = pytz.timezone('Asia/Shanghai')
    current_time = datetime.now(beijing_tz).strftime('%Y-%m-%d %H:%M:%S')

    experiment_record = {
        "record_time": current_time,  # 添加记录时间
        "model_name": model_name,
        "vocab_size": vocab_size,
        "num_parameters": num_parameters,
    }
    for attr in dir(EXP):
        if not attr.startswith("__") and not callable(getattr(EXP, attr)):
            experiment_record[attr.lower()] = getattr(EXP, attr)
    
    # 记录kwargs中的额外属性
    for key, value in kwargs.items():
        # if isinstance(value, (bool, int, float, str, list, dict)) or value is None:
        if isinstance(value, np.float32):
            experiment_record[key] = float(value)
        else:
            experiment_record[key] = value

    # 读取现有的JSON文件内容，如果文件不存在则初始化为空列表
    output_path = os.path.join('/root/MagicEmbed/results', json_file)
    try:
        with open(output_path, 'r', encoding='utf-8') as f:
            file_content = f.read().strip()
            data = json.loads(file_content) if file_content else []
    except (FileNotFoundError, json.JSONDecodeError):
        data = []

    # 将新记录添加到现有数据中
    data.append(experiment_record)

    # 保存更新后的数据到JSON文件
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)    

    # 输出保存路径
    print(f"实验信息已保存到: {output_path}")

def output_name(model_id, tag, extension):
    model_id_alphanum = re.sub(r"[^a-zA-Z0-9]", "_", model_id)
    filename = f"/root/MagicEmbed/results/{tag}/{model_id_alphanum}.{extension}"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    return filename

def write_tokenizer_analysis_results(token_infos, model_id, compress=True) -> str:
    output_file = output_name(model_id, "tokenizer_analysis", "jsonl")
    open_fn_with_formats = [(open, "")]
    if compress:  # write both compressed and uncompressed versions, with uncompressed never committed
        open_fn_with_formats.append((gzip.open, ".gz"))
    for open_func, gzext in open_fn_with_formats:
        with open_func(output_file + gzext, "wt", encoding='utf-8') as f:
            for _, token_info in sorted(token_infos.items()):
                print(json.dumps(token_info, ensure_ascii=False), file=f)
    print(f"已将{model_id=}的tokenizer analysis结果保存到文件: {output_file}")
    return output_file

def write_vocab_token_magic_scores(token_infos, model_id, compress=True) -> str:
    output_file = output_name(model_id, "vocab_token_magic_scores", "jsonl")
    open_fn_with_formats = [(open, "")]
    if compress:  # write both compressed and uncompressed versions, with uncompressed never committed
        open_fn_with_formats.append((gzip.open, ".gz"))
    for open_func, gzext in open_fn_with_formats:
        with open_func(output_file + gzext, "wt", encoding='utf-8') as f:
            for _, token_info in sorted(token_infos.items()):
                print(json.dumps(token_info, ensure_ascii=False), file=f)
    print(f"已将{model_id=}的vocab_token_magic_scores结果保存到文件: {output_file}")
    return output_file

def load_vocab_token_magic_scores(model_id) -> dict:
    token_infos = {}
    for open_fn, ext in [(open, ""), (gzip.open, ".gz")]:
        try:
            with open_fn(output_name(model_id, "vocab_token_magic_scores", "jsonl" + ext), "rt", encoding='utf-8') as f:
                for line in f:
                    token_info = json.loads(line.strip())
                    token_id = int(token_info.get("i")  )
                    token_infos[token_id] = token_info
                return token_infos
        except FileNotFoundError:
            pass
        except json.JSONDecodeError:
            print(f"文件解码错误: {output_name(model_id, 'vocab_token_magic_scores', 'jsonl' + ext)}")
    return token_infos

def load_vocab_verifications(model_id) -> dict:
    token_infos = {}
    for open_fn, ext in [(open, ""), (gzip.open, ".gz")]:
        try:
            with open_fn(output_name(model_id, "verifications", "jsonl" + ext), "rt", encoding='utf-8') as f:
                for line in f:
                    token_info = json.loads(line.strip())
                    token_id = int(token_info.get("i")  )
                    token_infos[token_id] = token_info
                return token_infos
        except FileNotFoundError:
            pass
        except json.JSONDecodeError:
            print(f"文件解码错误: {output_name(model_id, 'verifications', 'jsonl' + ext)}")
    return token_infos

def save_vocab_token_magic_scores_all_results(all_results, model_name, compress=True):
    def convert_np_types(data):
        if isinstance(data, dict):
            return {key: convert_np_types(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [convert_np_types(item) for item in data]
        elif isinstance(data, np.generic):
            return data.item()
        else:
            return data

    results_output_path = output_name(model_name, "magic_tokens_all_results", "jsonl")
    compressed_output_path = output_name(model_name, "magic_tokens_all_results", "jsonl.gz")

    if compress:
        with gzip.open(compressed_output_path, 'wt', encoding='utf-8') as f:
            for item in convert_np_types(all_results):
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        print(f"已将{model_name=}的magic_tokens_all_results结果保存到压缩文件: {compressed_output_path}")
    else:
        with jsonlines.open(results_output_path, mode='w') as writer:
            for item in convert_np_types(all_results):
                writer.write(item)
        print(f"已将{model_name=}的magic_tokens_all_results结果保存到文件: {results_output_path}")

def write_magic_tokens_within_threshold(token_scores, model_name,tokenizer, score_name, threshold=0.02,reverse=True) -> list:
    '''
    根据指标score_name，输出指标前threshold的token

    参数:
    token_scores (dict): token的分数字典
    model_name (str): 模型名称
    tokenizer: 分词器
    score_name (str): 指标名称
    threshold (float): 阈值
    reverse (bool): 是否降序排列
    '''
    sorted_token_scores = sorted(token_scores.items(), key=lambda item: item[1][f'{score_name}'], reverse=reverse) # 是个列表
    top_percent_count = max(1, int(len(sorted_token_scores) * threshold))
    top_percent_tokens = sorted_token_scores[:top_percent_count]
    # top_percent_tokens.sort(key=lambda item: token_scores[item[0]][f'{score_name}'], reverse=reverse)  # 保留原本的序号顺序
    print(f"根据指标{score_name}前{threshold*100}%的 token:", top_percent_tokens)
    # 修改文件名生成逻辑，避免替换小数点和百分号
    output_path = output_name(f"{model_name}_{score_name}_{threshold}", "magic_tokens_within_threshold", "csv")
    with open(output_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Token ID",'Token', f"{score_name}"]) 
        for token_id, score in top_percent_tokens:
            writer.writerow([token_id, tokenizer.convert_ids_to_tokens(token_id),score[f'{score_name}']])
    print(f"按照{score_name}前{threshold*100}%的token已保存到文件: {output_path}")
    return top_percent_tokens

def write_verification_candidates(token_infos: list, model_id, compress=True) -> str:
    output_file = output_name(model_id, "magic_tokens_within_threshold", "jsonl")
    open_fn_with_formats = [(open, "")]
    if compress:  # write both compressed and uncompressed versions, with uncompressed never committed
        open_fn_with_formats.append((gzip.open, ".gz"))
    for open_func, gzext in open_fn_with_formats:
        with open_func(output_file + gzext, "wt", encoding='utf-8') as f:
            for token_info in token_infos:
                print(json.dumps(token_info, ensure_ascii=False), file=f)
    return output_file

def load_verification_candidates(model_id):
    for open_fn, ext in [(open, ""), (gzip.open, ".gz")]:
        try:
            with open_fn(output_name(model_id, "magic_tokens_within_threshold", "jsonl" + ext), "r") as f:
                return [json.loads(line) for line in f]
        except FileNotFoundError:
            pass
    return None

def write_verification_results(token_infos, model_id, compress=True) -> str:
    output_file = output_name(model_id, "verifications", "jsonl")
    open_fn_with_formats = [(open, "")]
    if compress:  # write both compressed and uncompressed versions, with uncompressed never committed
        open_fn_with_formats.append((gzip.open, ".gz"))
    for open_func, gzext in open_fn_with_formats:
        with open_func(output_file + gzext, "wt", encoding='utf-8') as f:
            for _, token_info in token_infos.items():
                print(json.dumps(token_info, ensure_ascii=False), file=f)
    return output_file


def load_verification_results(model_id):
    for open_fn, ext in [(open, ""), (gzip.open, ".gz")]:
        try:
            with open_fn(output_name(model_id, "verifications", "jsonl" + ext), "r") as f:
                return {token_info["i"]: token_info for token_info in [json.loads(line) for line in f]}
        except FileNotFoundError:
            pass
    return None

# --- model analyzer utils ---
def check_vectors_on_unit_sphere(embeddings: np.ndarray) -> bool:
    """
    检查所有向量是否在单位超球体上

    参数:
    embeddings (np.ndarray): 所有token的嵌入向量,或者词嵌入权重矩阵

    返回:
    bool: 如果所有向量都在单位超球体上，返回True；否则返回False
    """
    # 计算每个向量的模长
    vector_norms = np.linalg.norm(embeddings, axis=1)
    print('向量模长大小：',vector_norms.shape)
    print(vector_norms)

    # 判断是否所有向量的模长都接近1，考虑浮点数误差，设置一个阈值
    threshold = 1e-5  # 设置一个小的阈值
    is_on_unit_sphere = np.allclose(vector_norms, 1, atol=threshold)
    if not is_on_unit_sphere:
        # 统计不在单位超球体上的向量数量
        num_not_on_unit_sphere = np.sum(~np.isclose(vector_norms, 1))
        print(f"不在单位超球体上的向量数量: {num_not_on_unit_sphere}")
        
        # 计算这些向量的模长的平均值和方差
        not_on_unit_sphere_norms = vector_norms[~np.isclose(vector_norms, 1)]
        mean_norm = np.mean(not_on_unit_sphere_norms)
        variance_norm = np.var(not_on_unit_sphere_norms)
        print(f"不在单位超球体上的向量模长的平均值: {mean_norm}")
        print(f"不在单位超球体上的向量模长的方差: {variance_norm}")

    return is_on_unit_sphere

def check_embeddings_is_anisotropic(all_embeddings: np.ndarray,
                                     batch_size=128,
                                     plot=False)->bool:
    """
    分析嵌入向量，计算余弦相似度并判断是否具有各向异性

    参数:
    all_embeddings (np.ndarray): 所有token的嵌入向量
    batch_size (int): 批处理大小，默认为128
    plot (bool): 是否画出余弦相似度的分布图，默认为False

    返回:
    bool: 如果所有token的向量具有各向异性，返回True；否则返回False
    """

    # 将嵌入移动到GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_embeddings = torch.tensor(all_embeddings).to(device)

    # 计算向量之间的模长
    vector_norms = torch.norm(all_embeddings, dim=1)

    # 计算向量之间的余弦相似度
    cosine_similarities = []

    for i in tqdm(range(0, all_embeddings.shape[0], batch_size), desc="计算余弦相似度",miniters=10):
        batch_embeddings = all_embeddings[i:i+batch_size]
        batch_norms = vector_norms[i:i+batch_size]
        for j in range(batch_embeddings.shape[0]):
            cosine_similarity = torch.matmul(batch_embeddings[j], all_embeddings.T) / (batch_norms[j] * vector_norms)
            cosine_similarities.append(cosine_similarity.cpu().numpy())

    # 将cosine_similarities展平成一个numpy数组
    cosine_similarities_np = np.concatenate(cosine_similarities).flatten() #展平成一维数组(vocab_size^2,)

    if plot:
        # 画出余弦距离的分布图和密度估计曲线
        plt.figure(figsize=(8, 4))
        plt.hist(cosine_similarities_np, bins=50, alpha=0.75, color='blue', edgecolor='black', density=True)
        
        # 添加密度估计曲线
        # import seaborn as sns
        # # sns.kdeplot(cosine_similarities_np, color='red', linewidth=2) #运行很慢,cosine_similarities_np.shape=(vocab_size^2,)，这个数量级太大了
        # sample_size = min(100, len(cosine_similarities_np))  # 取样以加快速度
        # sample_indices = np.random.choice(len(cosine_similarities_np), sample_size, replace=False)
        # sampled_cosine_similarities = cosine_similarities_np[sample_indices]
        # sns.kdeplot(sampled_cosine_similarities, color='red', linewidth=2)

        plt.title('Distribution of Cosine Similarity between Token Embeddings')
        plt.xlabel('Cosine Similarity')
        plt.ylabel('Density/Frequency')
        plt.grid(True)
        plt.show()

    # 对结果进行统计分析
    mean_similarity = np.mean(cosine_similarities_np)
    median_similarity = np.median(cosine_similarities_np)
    std_deviation = np.std(cosine_similarities_np)
    min_similarity = np.min(cosine_similarities_np)
    max_similarity = np.max(cosine_similarities_np)

    # 打印统计结果
    print(f"平均余弦相似度: {mean_similarity}")
    print(f"中位数余弦相似度: {median_similarity}")
    print(f"余弦相似度标准差: {std_deviation}")
    print(f"最小余弦相似度: {min_similarity}")
    print(f"最大余弦相似度: {max_similarity}")

    # # 使用Kolmogorov-Smirnov检验来判断cosine_similarities是否为均匀分布
    # from scipy.stats import kstest
    # from scipy.stats import uniform
    # ks_statistic, p_value = kstest(cosine_similarities_np, uniform(loc=0, scale=1).cdf)

    # # 如果p值小于0.05，则拒绝原假设，认为cosine_similarities不是均匀分布，即具有各向异性
    # is_anisotropic = p_value < 0.05
    # 使用快速方法判断cosine_similarities是否为均匀分布,均匀分布的余弦相似度应该在0附近
    is_anisotropic = not np.allclose(mean_similarity, 0, atol=0.01)

    return is_anisotropic,mean_similarity

def calculate_neighbor_distances(embeddings, batch_size=64,mode = 'nearest'):
    """
    计算词表中所有token与其最近邻token之间的距离，使用GPU加速计算
    
    返回:
    dict: 包含余弦距离、欧几里得距离和曼哈顿距离的最小值列表
    """
    distances = {
        'cosine': [],
        'euclidean': [],
        'manhattan': []
    }
    
    # 获取词表中所有token的嵌入表示
    # vocab_size = tokenizer.vocab_size
    # all_tokens = [tokenizer.convert_ids_to_tokens(i) for i in range(vocab_size)]
    # all_embeddings = model.encode(all_tokens, convert_to_tensor=True,batch_size=256)  # 转换为PyTorch张量
    # 生成一个简单的all_embeddings例子
    # 注意:这只是一个示例,实际的all_embeddings会有更多的token和更高的维度
    # all_embeddings = torch.tensor([
    #     [0.1, 0.2, 0.3],
    #     [0.4, 0.5, 0.6],
    #     [0.7, 0.8, 0.9],
    #     [1.0, 1.1, 1.2],
    #     [1.3, 1.4, 1.5],
    #     [1.6, 1.7, 1.8]
    # ])
    # print("示例 all_embeddings 形状:", all_embeddings.shape)
    # print("示例 all_embeddings 内容:\n", all_embeddings)
    
    # 注意:这里我们使用了一个小的示例
    # 实际的代码应该使用原始的all_embeddings,不要替换它
    # 将嵌入移动到GPU
    # transformer_model = model._first_module().auto_model
    # wte = transformer_model.encoder.embed_tokens.weight
    # wte = wte.detach().cpu()[:vocab_size]
    # all_embeddings = wte
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # all_embeddings = all_embeddings.to(device)
    all_embeddings = torch.tensor(embeddings).to(device)

    
    for i in tqdm(range(0, all_embeddings.shape[0], batch_size), desc="计算最近邻距离"):
        batch = all_embeddings[i:i+batch_size]

        # 计算批处理与所有嵌入之间的距离
        cosine_dist = 1 - torch.nn.functional.cosine_similarity(batch.unsqueeze(1), all_embeddings.unsqueeze(0), dim=2)  #cosine_dist.shape=[128, 32100]
        euclidean_dist = torch.cdist(batch, all_embeddings, p=2)
        manhattan_dist = torch.cdist(batch, all_embeddings, p=1)

        if mode == 'nearest':
            # 将自身距离设为无穷大
            # 将当前批次的对角线元素设置为无穷大
            for j in range(batch_size):
                if i+j < len(all_embeddings):
                    cosine_dist[j, i+j] = float('inf')
                    euclidean_dist[j, i+j] = float('inf')
                    manhattan_dist[j, i+j] = float('inf')
            # 找到每个token的最小距离
            distances['cosine'].extend(cosine_dist.min(dim=1)[0].cpu().numpy())
            distances['euclidean'].extend(euclidean_dist.min(dim=1)[0].cpu().numpy())
            distances['manhattan'].extend(manhattan_dist.min(dim=1)[0].cpu().numpy())
            # print(cosine_dist.shape)
            # print(cosine_dist.cpu().numpy().mean(axis=1).shape)
        elif mode == 'mean':
            distances['cosine'].extend(list(cosine_dist.cpu().numpy().mean(axis=1)))
            distances['euclidean'].extend(list(euclidean_dist.cpu().numpy().mean(axis=1)))
            distances['manhattan'].extend(list(manhattan_dist.cpu().numpy().mean(axis=1)))

    # 清空GPU内存
    torch.cuda.empty_cache()
    
    return distances

def plot_neighbor_distances_distribution(nearest_neighbor_distances,mode = 'nearest'):
    """
    绘制最近邻距离的分布图，包括直方图和核密度估计图

    参数:
    nearest_neighbor_distances (dict): 包含不同距离类型及其值(长度为词表大小的列表)的字典
    """
    import seaborn as sns
    # 设置图表样式
    plt.style.use('default')  # 使用默认样式而不是seaborn

    # 创建一个2x2的子图布局
    fig, axs = plt.subplots(1, 3, figsize=(24, 8))
    if mode == 'nearest':
        fig.suptitle('Nearest Neighbor Distance Distribution', fontsize=16)
    elif mode == 'mean':
        fig.suptitle('Mean Neighbor Distance Distribution', fontsize=16)
    # 扁平化axs数组以便于索引
    axs = axs.flatten()

    # 为每种距离类型绘制直方图和核密度估计图
    for i, (distance_type, values) in enumerate(nearest_neighbor_distances.items()):
        # 计算阈值，排除接近0的小值
        threshold = np.percentile(values, 0)  # 使用第1百分位数作为阈值
        filtered_values = [v for v in values if v > threshold]
        
        axs[i].hist(filtered_values, bins=50, density=True, alpha=0.7)
        axs[i].set_title(f'{distance_type.capitalize()} Distance Distribution')
        axs[i].set_xlabel('Distance')
        axs[i].set_ylabel('Frequency')
        
        # 添加核密度估计曲线
        sns.kdeplot(filtered_values, ax=axs[i], color='r')
        
        # 设置x轴的范围，排除接近0的部分
        axs[i].set_xlim(left=threshold)

    # 移除多余的子图
    # fig.delaxes(axs[3])

    # 调整子图之间的间距
    plt.tight_layout()

    # 显示图表
    plt.show()

# --- token magic score utils ---
def random_insert(text, insert_string, times):
    words = text.split()  # 将句子分割成单词列表
    for _ in range(times):
        insert_position = random.randint(0, len(words))  # 随机选择插入位置
        words.insert(insert_position, insert_string)  # 在随机位置插入字符串
    return " ".join(words)  # 将单词列表重新组合成句子

def calculate_token_distances(token, gt_embs, model):
    """
    计算单个token与所有gt_texts的余弦距离、欧氏距离和曼哈顿距离
    
    参数:
    token (str): 要计算距离的token
    gt_embss (list): 所有的ground truth文本列表的嵌入表示
    model: 用于编码的模型
    
    返回:
    tuple: 包含三个np.array，分别是token与每个gt_text的余弦距离、欧氏距离和曼哈顿距离
    """

    try:
        # 编码token
        # token_emb = model.encode(token)   #(768,)
        token_emb = model.encode([token])  #(1,768)

        # 计算token与所有gt_texts的余弦距离、欧氏距离和曼哈顿距离
        cosine_distance = cosine_distances(token_emb, gt_embs)[0]
        euclidean_distance = euclidean_distances(token_emb, gt_embs)[0]
        manhattan_distance = manhattan_distances(token_emb, gt_embs)[0]
        
        return DistanceMetrics(cosine_distance=cosine_distance,
                               euclidean_distance=euclidean_distance,
                               manhattan_distance=manhattan_distance)
    except Exception as e:
        print(f"Error occurred: {e}")
        print(f"Token: {token}")
        print(f"Token Embedding: {token_emb}")
        print(f"Ground Truth Embeddings: {gt_embs}")
        raise



def calculate_score(metrics_list, alpha=0.1, beta=0.05, gamma=0.1, smaller_is_better =True):
    """
    计算综合指标
    
    :param metrics_list: 相似度列表
    :param alpha: 上升次数的权重
    :param beta: 下降次数的权重
    :param gamma: 防止除零的常数
    :return: 综合指标
    """
    changes = np.diff(metrics_list)
    rise_amplitude = changes[changes > 0].sum()
    rise_count = (changes > 0).sum()
    fall_amplitude = (-changes[changes < 0]).sum()
    fall_count = (changes < 0).sum()
    if smaller_is_better:
        comprehensive_score = (fall_amplitude + alpha * fall_count) / (rise_amplitude + beta * rise_count + gamma)
    else:
        comprehensive_score = (rise_amplitude + alpha * rise_count) / (fall_amplitude + beta * fall_count + gamma)
    return comprehensive_score


def calculate_additive_score(similarities, w1=0.5, w2=0.3, w3=0.2):
    # Calculate Mean Rate of Change (MRC)
    changes = np.diff(similarities)
    MRC = np.mean(changes)
    
    # Calculate Variance (VAR)
    VAR = np.var(similarities)
    
    # Calculate Proportion of Increases (PI)
    PI = np.sum(changes < 0) / len(changes)
    
    # Calculate Score
    score = w1 * MRC - w2 * (1 - VAR) - w3 * PI
    
    return score

def calculate_score_with_token_distance(metrics_list, token_distance ,alpha=0.1, beta=0.05, gamma=0.01, smaller_is_better =True):
    """
    计算综合指标
    
    :param metrics_list: 相似度列表
    :param token_distance: 令牌和被比较的句子之间的距离
    :param alpha: 上升次数的权重
    :param beta: 下降次数的权重
    :param gamma: 防止除零的常数
    :return: 综合指标
    """
    changes = np.diff(metrics_list)
    rise_amplitude = changes[changes > 0].sum()
    rise_count = (changes > 0).sum()
    fall_amplitude = (-changes[changes < 0]).sum()
    fall_count = (changes < 0).sum()
    if smaller_is_better:
        comprehensive_score = (fall_amplitude + alpha * fall_count + 0.1*token_distance) / (rise_amplitude + beta * rise_count + gamma )
    else:
        comprehensive_score = (rise_amplitude + alpha * rise_count + 0.1*token_distance) / (fall_amplitude + beta * fall_count + gamma )

    return comprehensive_score

def magic_token_test_metric(token,
                            gt_texts,
                            contract_texts,
                            gt_embs,
                            contract_embs,
                            gt_metrics, 
                            num ,
                            model,
                            ):
    results = {'Prefix': [], 'Suffix': [], 'Insert': []}
    total = len(contract_texts)
    token_distances = calculate_token_distances(token, gt_embs, model)  #输出一个列表，包含token与每个gt_text的距离


    def process_texts(text_list, gt_emb, gt_metric, text_type, id):
        pred_emb = model.encode(text_list)
        
        # ret = cosine_similarity(gt_emb.reshape(1, -1), pred_emb)[0]
        ret = distance_metrics(gt_emb, pred_emb)

        result = {
            'Pair_id': id,  
            'Source text': gt_texts[id],
            'Texts to be contrasted': [contract_text] + text_list,
            'cosine_distance': [gt_metric.cosine_distance] + list(ret.cosine_distance),
            'cosine_distance_contrast': [None],
            'euclidean_distance': [gt_metric.euclidean_distance] + list(ret.euclidean_distance),
            'euclidean_distance_contrast': [None],
            'manhattan_distance': [gt_metric.manhattan_distance] + list(ret.manhattan_distance),
            'manhattan_distance_contrast': [None],
        }
        # print(token_similarities[id])
        result['cosine_distance_score'] = calculate_score_with_token_distance(result['cosine_distance'],token_distances.cosine_distance[id],smaller_is_better=True)
        result['euclidean_distance_score'] = calculate_score_with_token_distance(result['euclidean_distance'],token_distances.euclidean_distance[id],smaller_is_better=True)
        result['manhattan_distance_score'] = calculate_score_with_token_distance(result['manhattan_distance'],token_distances.manhattan_distance[id],alpha=1,beta=0.5,gamma=0.1,smaller_is_better=True)

        for i in range(num):
            cosine_distance_contrast = (result['cosine_distance'][i] > result['cosine_distance'][i + 1])
            result['cosine_distance_contrast'].append(cosine_distance_contrast)
            euclidean_distance_contrast = (result['euclidean_distance'][i] > result['euclidean_distance'][i + 1])
            result['euclidean_distance_contrast'].append(euclidean_distance_contrast)
            manhattan_distance_contrast = (result['manhattan_distance'][i] > result['manhattan_distance'][i + 1])
            result['manhattan_distance_contrast'].append(manhattan_distance_contrast)

        results[text_type].append(result)

    for id, contract_text in enumerate(contract_texts):
        text_list_prefix = [token * i + contract_text for i in range(1, num + 1)]
        text_list_suffix = [contract_text + token * i for i in range(1, num + 1)]
        text_list_insert = []
        tem_text = contract_text
        for i in range(1, num + 1):
            new_text = random_insert(tem_text, token, 1)
            tem_text = new_text
            text_list_insert.append(new_text)
        
        for text_list, text_type in [(text_list_prefix, 'Prefix'), (text_list_suffix, 'Suffix'), (text_list_insert, 'Insert')]:
            process_texts(text_list, gt_embs[id],
                           DistanceMetrics(
                                gt_metrics.cosine_distance[id],
                                gt_metrics.euclidean_distance[id],
                                gt_metrics.manhattan_distance[id]
                            )
                            , text_type, id)


    weights = {'Prefix': 0.35, 'Suffix': 0.35, 'Insert': 0.3}
    token_score_aggregation = {
        'cosine_distance_score': 0,
        'euclidean_distance_score': 0,
        'manhattan_distance_score': 0
    }
    for text_type, weight in weights.items():
        for result in results[text_type]:
            token_score_aggregation['cosine_distance_score'] += result['cosine_distance_score'] * weight
            token_score_aggregation['euclidean_distance_score'] += result['euclidean_distance_score'] * weight
            token_score_aggregation['manhattan_distance_score'] += result['manhattan_distance_score'] * weight
    
    token_score_aggregation = {k: round(v / total, 6) for k, v in token_score_aggregation.items()}

    return results, token_score_aggregation

def magic_token_test_metric_multi_token(tokens: list[str],
                                        gt_texts : list[str],
                                        contract_texts : list[str],
                                        gt_embs,
                                        contract_embs,
                                        gt_metrics, 
                                        num,
                                        model,
                                        batch_size=256):
    results = {token: {'Prefix': [], 'Suffix': [], 'Insert': []} for token in tokens}
    token_score_aggregation = {token: {'cosine_distance_score': 0, 'euclidean_distance_score': 0, 'manhattan_distance_score': 0} for token in tokens}
    total = len(contract_texts)
    weights = {'Prefix': 0.35, 'Suffix': 0.35, 'Insert': 0.3}

    # 计算每个token的距离
    token_distances = {token: calculate_token_distances(token, gt_embs, model) for token in tokens}

    # Prepare all text lists for encoding
    all_text_lists = []
    text_indices = []
    
    for token in tokens:
        for id, contract_text in enumerate(contract_texts):
            text_list_prefix = [token * i + contract_text for i in range(1, num + 1)]
            text_list_suffix = [contract_text + token * i for i in range(1, num + 1)]
            text_list_insert = []
            tem_text = contract_text
            for i in range(1, num + 1):
                new_text = random_insert(tem_text, token, 1)
                tem_text = new_text
                text_list_insert.append(new_text)
            
            all_text_lists.extend([text_list_prefix, text_list_suffix, text_list_insert]) 
            text_indices.extend([(token, id, 'Prefix'), (token, id, 'Suffix'), (token, id, 'Insert')])
    
    # Encode all text lists in batches
    all_pred_embs = model.encode([text for sublist in all_text_lists for text in sublist], batch_size=batch_size) # 64*3*5*8 = 7680
    
    # Process results
    index = 0
    for (token, id, text_type), text_list in zip(text_indices, all_text_lists):
        pred_emb = all_pred_embs[index:index + len(text_list)]
        index += len(text_list)
        
        gt_emb = gt_embs[id]
        gt_metric = DistanceMetrics(
            gt_metrics.cosine_distance[id],
            gt_metrics.euclidean_distance[id],
            gt_metrics.manhattan_distance[id]
        )
        
        ret = distance_metrics(gt_emb, pred_emb)
        
        result = {
            'Pair_id': id,
            'Source text': gt_texts[id],
            'Texts to be contrasted': [contract_texts[id]] + text_list,
            'cosine_distance': [gt_metric.cosine_distance] + list(ret.cosine_distance),
            'cosine_distance_contrast': [None],
            'euclidean_distance': [gt_metric.euclidean_distance] + list(ret.euclidean_distance),
            'euclidean_distance_contrast': [None],
            'manhattan_distance': [gt_metric.manhattan_distance] + list(ret.manhattan_distance),
            'manhattan_distance_contrast': [None],
        }
        
        result['cosine_distance_score'] = calculate_score_with_token_distance(result['cosine_distance'], token_distances[token].cosine_distance[id], smaller_is_better=True)
        result['euclidean_distance_score'] = calculate_score_with_token_distance(result['euclidean_distance'], token_distances[token].euclidean_distance[id], smaller_is_better=True)
        result['manhattan_distance_score'] = calculate_score_with_token_distance(result['manhattan_distance'], token_distances[token].manhattan_distance[id], alpha=1, beta=0.5, gamma=0.1, smaller_is_better=True)
        
        for i in range(num):
            cosine_distance_contrast = (result['cosine_distance'][i] > result['cosine_distance'][i + 1])
            result['cosine_distance_contrast'].append(cosine_distance_contrast)
            euclidean_distance_contrast = (result['euclidean_distance'][i] > result['euclidean_distance'][i + 1])
            result['euclidean_distance_contrast'].append(euclidean_distance_contrast)
            manhattan_distance_contrast = (result['manhattan_distance'][i] > result['manhattan_distance'][i + 1])
            result['manhattan_distance_contrast'].append(manhattan_distance_contrast)
        
        results[token][text_type].append(result)

    # 计算token_score_aggregation
    for token in tokens:
        for text_type, weight in weights.items():
            for result in results[token][text_type]:
                token_score_aggregation[token]['cosine_distance_score'] += result['cosine_distance_score'] * weight
                token_score_aggregation[token]['euclidean_distance_score'] += result['euclidean_distance_score'] * weight
                token_score_aggregation[token]['manhattan_distance_score'] += result['manhattan_distance_score'] * weight
        
        token_score_aggregation[token] = {k: round(v / total, 6) for k, v in token_score_aggregation[token].items()}

    return results, token_score_aggregation
    

def magic_token_verification(token,
                       verification_gt_texts,
                       verification_gt_embs, 
                       verification_contract_texts,
                       verification_contract_embs,
                       verification_gt_metrics,
                        add_num,
                         model,
                         batch_size=256,
                         cosine_threshold = 0,
                         euclidean_threshold = 0,
                         manhattan_threshold = 0):
    results = {'Prefix': [], 'Suffix': [], 'Insert': []}
    total = len(verification_contract_texts) #250

    # 准备所有文本列表以进行编码
    all_text_lists = []
    text_indices = []

    for id, verification_contract_text in enumerate(verification_contract_texts):
        text_list_prefix = [token * i + verification_contract_text for i in range(1, add_num + 1)]
        text_list_suffix = [verification_contract_text + token * i for i in range(1, add_num + 1)]
        text_list_insert = []
        tem_text = verification_contract_text
        for i in range(1, add_num + 1):
            new_text = random_insert(tem_text, token, 1)
            tem_text = new_text
            text_list_insert.append(new_text)
        
        all_text_lists.extend([text_list_prefix, text_list_suffix, text_list_insert]) # [ ['xx','xx',...'xx']（8个‘xx’）, ['xx','xx'], ['xx','xx'] ,['xx','xx'], ['xx','xx'], ['xx','xx'] ]
        text_indices.extend([(id, 'Prefix'), (id, 'Suffix'), (id, 'Insert')])         # [(0, 'Prefix'), (0, 'Suffix'), (0, 'Insert'), (1, 'Prefix'), (1, 'Suffix'), (1, 'Insert')...(249, 'Insert')]

    # 批量编码所有文本列表
    all_pred_embs = model.encode([text for sublist in all_text_lists for text in sublist],batch_size=batch_size) # 8*3*250 = 6000

    # 处理结果
    index = 0
    for (id, text_type), text_list in zip(text_indices, all_text_lists):
        pred_emb = all_pred_embs[index:index + len(text_list)]
        index += len(text_list)

        gt_emb = verification_gt_embs[id]
        gt_metric = DistanceMetrics(
            verification_gt_metrics.cosine_distance[id],
            verification_gt_metrics.euclidean_distance[id],
            verification_gt_metrics.manhattan_distance[id]
        )
        
        # ret = cosine_similarity(gt_emb.reshape(1, -1), pred_emb)[0]
        ret = distance_metrics(gt_emb, pred_emb)

        result = {
            'Pair_id': id,  
            'Source text': verification_gt_texts[id],
            'Texts to be contrasted': [verification_contract_texts[id]] + text_list,
            'cosine_distance': [gt_metric.cosine_distance] + list(ret.cosine_distance),
            'cosine_distance_contrast': np.mean(ret.cosine_distance) - gt_metric.cosine_distance,
            'euclidean_distance': [gt_metric.euclidean_distance] + list(ret.euclidean_distance),
            'euclidean_distance_contrast': np.mean(ret.euclidean_distance) - gt_metric.euclidean_distance,
            'manhattan_distance': [gt_metric.manhattan_distance] + list(ret.manhattan_distance),
            'manhattan_distance_contrast': np.mean(ret.manhattan_distance) - gt_metric.manhattan_distance,
        }

        results[text_type].append(result)

    # 设置阈值
    threshold =  {
        'cosine_distance': -1*cosine_threshold,
        'euclidean_distance': -1*euclidean_threshold,
        'manhattan_distance': -1*manhattan_threshold
    } # 可以根据需要调整阈值

    # total_count = 0
    cosine_distance_sum = 0
    euclidean_distance_sum = 0
    manhattan_distance_sum = 0

    # 遍历所有结果
    weights = {'Prefix': 4/11, 'Suffix': 4/11, 'Insert': 3/11} # 和为1，(4:4:3)/11 
    for text_type, weight in weights.items():
        for result in results[text_type]:
            # 如果余弦距离对比小于阈值，则加1*weight，否则加0
            cosine_distance_sum += 1 * weight if result['cosine_distance_contrast'] < threshold['cosine_distance'] else 0
            # 如果欧几里得距离对比小于阈值，则加1*weight，否则加0
            euclidean_distance_sum += 1 * weight if result['euclidean_distance_contrast'] < threshold['euclidean_distance'] else 0
            # 如果曼哈顿距离对比小于阈值，则加1*weight，否则加0
            manhattan_distance_sum += 1 * weight if result['manhattan_distance_contrast'] < threshold['manhattan_distance'] else 0

    # 计算平均值
    mean_cosine_count = cosine_distance_sum / total
    mean_euclidean_count = euclidean_distance_sum / total
    mean_manhattan_count = manhattan_distance_sum / total

    # 所有指标的平均变化量
    mean_metrics_count = DistanceMetrics(
        mean_cosine_count,
        mean_euclidean_count,
        mean_manhattan_count
    )

    # # 与阈值比较,如果大于阈值，则将对应的flag设为1
    # if mean_cosine_count > judgment_threshold:
    #     token_flag_aggregation['cosine_distance_flag'] = 1
    # if mean_euclidean_count > judgment_threshold:
    #     token_flag_aggregation['euclidean_distance_flag'] = 1
    # if mean_manhattan_count > judgment_threshold:
    #     token_flag_aggregation['manhattan_distance_flag'] = 1

    return results,mean_metrics_count


def magic_token_verification_v0(token,
                       verification_gt_texts,
                       verification_gt_embs, 
                       verification_contract_texts,
                       verification_contract_embs,
                       verification_gt_metrics,
                        add_num,
                         model,
                         cosine_threshold,
                         euclidean_threshold,
                         manhattan_threshold,):
    results = {'Prefix': [], 'Suffix': [], 'Insert': []}
    total = len(verification_contract_texts)

    def process_texts(text_list, gt_emb, gt_metric, text_type, id):
        pred_emb = model.encode(text_list)
        
        # ret = cosine_similarity(gt_emb.reshape(1, -1), pred_emb)[0]
        ret = distance_metrics(gt_emb, pred_emb)

        result = {
            'Pair_id': id,  
            'Source text': verification_gt_texts[id],
            'Texts to be contrasted': [verification_contract_text] + text_list,
            'cosine_distance': [gt_metric.cosine_distance] + list(ret.cosine_distance),
            'cosine_distance_contrast': np.mean(ret.cosine_distance) - gt_metric.cosine_distance,
            'euclidean_distance': [gt_metric.euclidean_distance] + list(ret.euclidean_distance),
            'euclidean_distance_contrast': np.mean(ret.euclidean_distance) - gt_metric.euclidean_distance,
            'manhattan_distance': [gt_metric.manhattan_distance] + list(ret.manhattan_distance),
            'manhattan_distance_contrast': np.mean(ret.manhattan_distance) - gt_metric.manhattan_distance,
        }


        # for i in range(add_num):
        #     cosine_distance_contrast = (result['cosine_distance'][i] > result['cosine_distance'][i + 1])
        #     result['cosine_distance_contrast'].append(cosine_distance_contrast)
        #     euclidean_distance_contrast = (result['euclidean_distance'][i] > result['euclidean_distance'][i + 1])
        #     result['euclidean_distance_contrast'].append(euclidean_distance_contrast)
        #     manhattan_distance_contrast = (result['manhattan_distance'][i] > result['manhattan_distance'][i + 1])
        #     result['manhattan_distance_contrast'].append(manhattan_distance_contrast)

        results[text_type].append(result)
    
    for id, verification_contract_text in enumerate(verification_contract_texts):
        text_list_prefix = [token * i + verification_contract_text for i in range(1, add_num + 1)]
        text_list_suffix = [verification_contract_text + token * i for i in range(1, add_num + 1)]
        text_list_insert = []
        tem_text = verification_contract_text
        for i in range(1, add_num + 1):
            new_text = random_insert(tem_text, token, 1)
            tem_text = new_text
            text_list_insert.append(new_text)
        
        for text_list, text_type in [(text_list_prefix, 'Prefix'), (text_list_suffix, 'Suffix'), (text_list_insert, 'Insert')]:
            process_texts(text_list, verification_gt_embs[id],
                           DistanceMetrics(
                                verification_gt_metrics.cosine_distance[id],
                                verification_gt_metrics.euclidean_distance[id],
                                verification_gt_metrics.manhattan_distance[id]
                            ),
                            text_type, id)
            
    
    # token_flag_aggregation = {
    #     'cosine_distance_flag': 0,
    #     'euclidean_distance_flag': 0,
    #     'manhattan_distance_flag': 0
    # }
    # # 设置阈值
    # threshold =  2/3  # 可以根据需要调整阈值

    # cosine_true_count = 0
    # euclidean_true_count = 0
    # manhattan_true_count = 0
    # # 遍历所有结果
    # for loc in ['Prefix', 'Suffix', 'Insert']:
    #     for result in results[loc]:
    #         # 统计每种距离度量中为True的个数
    #         cosine_true_count += sum(1 for x in result['cosine_distance_contrast'][1:] if x)
    #         euclidean_true_count += sum(1 for x in result['euclidean_distance_contrast'][1:] if x)
    #         manhattan_true_count += sum(1 for x in result['manhattan_distance_contrast'][1:] if x)
            
    # print(f'阈值:{threshold:.3%}')
    # print(f'余弦距离真值计数:{cosine_true_count}/{total}--{cosine_true_count/total:.3%},欧几里得距离真值计数:{euclidean_true_count}/{total}--{euclidean_true_count/total:.3%},曼哈顿距离真值计数:{manhattan_true_count}/{total}--{manhattan_true_count/total:.3%}')
    # # 如果True的个数超过阈值，将对应的flag设为
    # if cosine_true_count > threshold*total:
    #     token_flag_aggregation['cosine_distance_flag'] = 1
    # if euclidean_true_count > threshold*total:
    #     token_flag_aggregation['euclidean_distance_flag'] = 1
    # if manhattan_true_count > threshold*total:
    #     token_flag_aggregation['manhattan_distance_flag'] = 1

    token_flag_aggregation = {
        'cosine_distance_flag': 0,
        'euclidean_distance_flag': 0,
        'manhattan_distance_flag': 0
    }
    # 设置阈值
    threshold =  {
        'cosine_distance': -1*cosine_threshold,
        'euclidean_distance': -1*euclidean_threshold,
        'manhattan_distance': -1*manhattan_threshold
    } # 可以根据需要调整阈值

    # total_count = 0
    cosine_distance_sum = 0
    euclidean_distance_sum = 0
    manhattan_distance_sum = 0

    # 遍历所有结果
    weights = {'Prefix': 0.35, 'Suffix': 0.35, 'Insert': 0.3}
    for text_type, weight in weights.items():
        for result in results[text_type]:
            # total_count += 1
            cosine_distance_sum += result['cosine_distance_contrast']*weight
            euclidean_distance_sum += result['euclidean_distance_contrast']*weight
            manhattan_distance_sum += result['manhattan_distance_contrast']*weight
    
    # print(total_count)
    # 计算平均值
    mean_cosine_distance = cosine_distance_sum / total
    mean_euclidean_distance = euclidean_distance_sum / total
    mean_manhattan_distance = manhattan_distance_sum / total

    # 所有指标的平均变化量
    mean_metrics_change = DistanceMetrics(
        mean_cosine_distance,
        mean_euclidean_distance,
        mean_manhattan_distance
    )
    # print(f"平均距离变化量: ")
    # print(mean_metrics_change)
    # 与阈值比较
    if mean_cosine_distance < threshold['cosine_distance']:
        token_flag_aggregation['cosine_distance_flag'] = 1
    if mean_euclidean_distance < threshold['euclidean_distance']:
        token_flag_aggregation['euclidean_distance_flag'] = 1
    if mean_manhattan_distance < threshold['manhattan_distance']:
        token_flag_aggregation['manhattan_distance_flag'] = 1
    # print(f"标志聚合结果: {token_flag_aggregation}")
    # print(f'阈值:{threshold:.3%}')
    # print(f'余弦距离真值计数:{cosine_true_count}/{total}--{cosine_true_count/total:.3%},欧几里得距离真值计数:{euclidean_true_count}/{total}--{euclidean_true_count/total:.3%},曼哈顿距离真值计数:{manhattan_true_count}/{total}--{manhattan_true_count/total:.3%}')
    # # 如果True的个数超过阈值，将对应的flag设为
    # if cosine_true_count > threshold*total:
    #     token_flag_aggregation['cosine_distance_flag'] = 1
    # if euclidean_true_count > threshold*total:
    #     token_flag_aggregation['euclidean_distance_flag'] = 1
    # if manhattan_true_count > threshold*total:
    #     token_flag_aggregation['manhattan_distance_flag'] = 1
    return results,mean_metrics_change,token_flag_aggregation

# --- distance metric utils ---


def cosine_distance(mat: np.ndarray, v: np.ndarray) -> float:
    """
    Calculate the cosine distance between a matrix and a vector.

    Args:
        mat: The matrix of shape (n, m).
        v: The vector of shape (m,).

    Returns:
        The cosine distances between the each row in the matrix and the vector.
    """
    # clip here mainly needed if rows of mat are 0, e.g. in Phi-3
    return 1 - np.dot(mat, v) / (np.linalg.norm(mat, axis=1) * np.linalg.norm(v)).clip(min=1e-6)


DistanceMetrics = namedtuple("Metrics", ["cosine_distance", "euclidean_distance", "manhattan_distance"])

def distance_metrics(emb1: np.ndarray,emb2:np.ndarray ) -> DistanceMetrics:
    """
    计算两个嵌入向量之间的距离度量。

    参数:
    emb1 (np.ndarray): 第一个嵌入向量或嵌入向量矩阵
    emb2 (np.ndarray): 第二个嵌入向量或嵌入向量矩阵

    返回:
    DistanceMetrics: 包含余弦距离、欧几里得距离和曼哈顿距离的命名元组

    注意:
    - 如果emb1是1维向量而emb2是2维矩阵，函数会将emb1重塑为2维
    - 如果两个输入都是2维矩阵，函数会计算对角线上的距离
    """
    if emb1.ndim == 1 and emb2.ndim != 1:
        emb1 = emb1.reshape(1, -1)
        return DistanceMetrics(
            cosine_distances(emb1, emb2)[0],
            euclidean_distances(emb1, emb2)[0],
            manhattan_distances(emb1, emb2)[0],
        )
    elif emb1.ndim != 1 and emb2.ndim != 1:
        return DistanceMetrics(
            cosine_distances(emb1, emb2).diagonal(),
            euclidean_distances(emb1, emb2).diagonal(),
            manhattan_distances(emb1, emb2).diagonal(),
        )

def oov_distance_metrics(mat: np.ndarray, known_unused_tokens: List[int]) -> DistanceMetrics:
    """
    Calculate the distance metrics for out-of-vocabulary (OOV) tokens.

    Args:
        mat: The matrix of shape (n, m) representing the embeddings.
        known_unused_tokens: A list of token indices that are known to be unused.

    Returns:
        A named tuple containing the cosine distance metrics:
        - cosine_distance: The cosine distance between the embeddings and the mean unused vector.
        - cosine_distance_without_first_pc: The cosine distance between the embeddings and the mean unused vector without the first principal component.
        - l2_distance: The L2 distance between the embeddings and the mean unused vector.
    """
    assert mat.shape[0] > mat.shape[1], "Expected more tokens than dimensions"

    pca = PCA(n_components=1)
    pca.fit(mat)
    first_pc = pca.components_[0]

    mat_without_first_pc = mat - np.outer(mat.dot(first_pc), first_pc)

    mean_unused_vector = mat[known_unused_tokens].mean(axis=0)
    mean_unused_vector_without_first_pc = mat_without_first_pc[known_unused_tokens].mean(axis=0)
    l2_distance = np.linalg.norm(mat - mean_unused_vector, axis=1)

    return DistanceMetrics(
        cosine_distance(mat, mean_unused_vector),
        cosine_distance(mat_without_first_pc, mean_unused_vector_without_first_pc),
        l2_distance,
    )


# --- misc


def format_ranges(ixs: list[int]) -> str:
    sixs = sorted(set(ixs))
    ranges = []
    for k, g in itertools.groupby(enumerate(sixs), lambda x: x[0] - x[1]):
        g = list(g)
        if len(g) == 1:
            ranges.append(f"{g[0][1]}")
        else:
            ranges.append(f"{g[0][1]}-{g[-1][1]}")
    s = ", ".join(ranges)
    c = Counter(ixs)
    for d in range(2, 5):
        ddups = {k: v for k, v in c.items() if v == d}
        if ddups:
            ds = ",".join(str(s) for s in sorted(ddups.keys()))
            s += f" - 2x {len(ddups)} items: {ds}"
    return s

if __name__ == "__main__":
    mat = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9],[10,11,12]])
    # v = np.array([3, 4, 0])
    # print(np.linalg.norm(v))
    # print(np.linalg.norm(mat, axis=1))
    # cosine_distances = cosine_distance(mat, v)
    # print(cosine_distances)
    pca = PCA(n_components=1)
    pca.fit(mat)
    first_pc = pca.components_[0]
    print(first_pc)
    dot_product = mat.dot(first_pc)
    print(dot_product)
    outer_product = np.outer(dot_product, first_pc)
    print(outer_product)
    mat_without_first_pc = mat - outer_product
    print(mat_without_first_pc)


